"""Core CLUECL3 model: encoders, imputers, routers and classifier."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import LinearLayer, Prediction, Router
from utils import info_nce_loss, safe_l2_normalize, tensor_version_or_none


class CLUECL3(nn.Module):
    # FP16-safe replacement for -inf in topk similarity scores.
    # FP16 range is [-65504, 65504], so -1e9 overflows to NaN.
    _NEG_INF_CLAMP: float = -65000.0

    def __init__(self, in_dim, hidden_dim, num_class, dropout, prediction_dicts):
        super().__init__()
        if len(in_dim) != 3 or any(dim <= 0 for dim in in_dim):
            raise ValueError("in_dim must contain three positive feature sizes")
        if not hidden_dim or hidden_dim[0] < 2 or any(
                dim <= 0 for dim in hidden_dim):
            raise ValueError(
                "hidden_dim must contain positive sizes and start with at least 2")
        if num_class < 2:
            raise ValueError("num_class must be at least 2")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if set(prediction_dicts) != {0, 1, 2} or any(
                not dims or any(dim <= 0 for dim in dims)
                for dims in prediction_dicts.values()):
            raise ValueError(
                "prediction_dicts must define three non-empty positive branches")
        self.views = 3  # views = 3, in_dim = [1000, 503, 1000]
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.dropout = dropout

        self.att = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])  # fc [in_dim, in_dim]
        self.emb = nn.ModuleList(
            [LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])  # fc [in_dim, hidden]
        self.aux_clf = nn.ModuleList(
            [LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])  # fc [hidden, num_class]

        # ---- aux_conf: 2 层 MLP 置信度网络（含 LN + Sigmoid）----
        # 类比 GREMI 的 Conf Network，独立特征提取能力，输出直接为 [0,1] 置信度
        # 中间层维度取 hidden_dim[0] // 2，加 LayerNorm 防过拟合（BRCA 仅 ~758 训练样本）
        conf_mid = max(hidden_dim[0] // 2, 1)  # 至少为 1，防止 hidden_dim 极小时出错
        self.aux_conf = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[0], conf_mid),
                nn.LayerNorm(conf_mid),
                nn.ReLU(),
                nn.Linear(conf_mid, 1),
                nn.Sigmoid()  # 内部已 sigmoid，外部调用处不再套 sigmoid
            ) for _ in range(self.views)
        ])
        # 注意：不加 Xavier 初始化。PyTorch 默认 Kaiming uniform 对 ReLU 更友好；
        # Xavier 缩放过小会导致 sigmoid 输出集中在 0.5 附近，置信度门控失效。

        # [FIX-Issue3] Chain intermediate layers: each layer's input = previous output.
        # dims = [views*hidden_dim[0], hidden_dim[1], hidden_dim[2], ...]
        # e.g. hidden_dim=[128,64,32] → dims=[384,64,32] → 384→64→32→num_class
        _mm_dims = [self.views * hidden_dim[0]] + hidden_dim[1:]
        self.MMClasifier = []
        for i in range(len(_mm_dims) - 1):
            self.MMClasifier.append(LinearLayer(_mm_dims[i], _mm_dims[i + 1]))
            if i < len(_mm_dims) - 2:  # ReLU/Dropout between hidden layers, not before final
                self.MMClasifier.append(nn.ReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
        self.MMClasifier.append(LinearLayer(_mm_dims[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # CLUE — 输入维度必须是 hidden_dim[0]（feat_emb 的实际维度），不是 hidden_dim[-1]
        self.a2b = Prediction([hidden_dim[0]] + prediction_dicts[0])
        self.a2c = Prediction([hidden_dim[0]] + prediction_dicts[0])
        self.b2a = Prediction([hidden_dim[0]] + prediction_dicts[1])
        self.b2c = Prediction([hidden_dim[0]] + prediction_dicts[1])
        self.c2a = Prediction([hidden_dim[0]] + prediction_dicts[2])
        self.c2b = Prediction([hidden_dim[0]] + prediction_dicts[2])
        # Plain lookup table: modules are already registered by the attributes
        # above.  This avoids rebuilding the same six-entry dict at every call.
        self._predictor_map = {
            (0, 1): self.a2b, (0, 2): self.a2c,
            (1, 0): self.b2a, (1, 2): self.b2c,
            (2, 0): self.c2a, (2, 1): self.c2b,
        }

        # One shared router jointly learns expert reliability and the selective
        # action.  Action 0=skip, 1=impute the first missing view,
        # 2=impute the second missing view.
        self.router = Router(
            input_dim=12, hidden_dim=64, num_experts=4, num_actions=3)

    def forward(self, data_list, label=None, infer=False, aux_loss=False,
                lambda_al=0.05, record_loss=True):
        att_score, feat_emb, aux_logit, aux_confidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            att_score[view] = torch.sigmoid(self.att[view](data_list[view]))
            feat_emb[view] = data_list[view] * att_score[view]
            feat_emb[view] = F.dropout(F.relu(self.emb[view](feat_emb[view])), self.dropout, training=self.training)
            # Auxiliary logits are only consumed by the auxiliary loss.  Skip
            # three classifier heads during inference or when lambda_al == 0.
            if not infer and aux_loss:
                aux_logit[view] = self.aux_clf[view](feat_emb[view])
            # aux_conf MLP 内部已包含 Sigmoid，直接输出 [0,1] 置信度
            aux_confidence[view] = self.aux_conf[view](feat_emb[view])
            feat_emb[view] = feat_emb[view] * aux_confidence[view]

        MMfeature = torch.cat([i for i in feat_emb.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        loss_dict = {}
        _log = lambda value: round(value.detach().item(), 4) if record_loss else None
        # 1. Loss between classifier and gt
        MMLoss = torch.mean(self.criterion(MMlogit, label))
        loss_dict["clf"] = _log(MMLoss)

        if aux_loss:
            aux_losses = []
            for view in range(self.views):
                pred = F.softmax(aux_logit[view], dim=1)
                # TCP is a teacher target for the confidence network. Detaching
                # prevents the MSE term from moving the auxiliary classifier's
                # probability toward the confidence output.
                p_target = torch.gather(
                    input=pred, dim=1,
                    index=label.unsqueeze(dim=1)).view(-1).detach()
                # 3. 置信度损失：MSE(aux_conf, TCP) + L_Cls(aux_clf)
                #    — MSE 训练置信度网络（aux_conf MLP）去拟合 TCP（真实标签对应概率）
                #    — L_Cls 强制 aux_clf 提取有鉴别力的特征，梯度同时回传到 aux_conf 共享的输入 feat_emb
                #    — 两者联合确保置信度网络"既懂分类又知诚实"（GREMI 公式5 的设计哲学）
                confidence_loss = (
                    F.mse_loss(aux_confidence[view].view(-1), p_target)
                    + self.criterion(aux_logit[view], label).mean())
                MMLoss = MMLoss + lambda_al * confidence_loss
                aux_losses.append(_log(lambda_al * confidence_loss))

            loss_dict["aux"] = aux_losses
        return MMLoss, None, loss_dict

    def train_missing_cg(self, data_list, mask, label=None,
                         aux_loss=False, lambda_al=0.05,
                         original_data_list=None,
                         support_mask=None,
                         use_cross_sample_impute=True,
                         use_prototype_bank=True,
                         lambda_imputation=0.1,
                         cross_sample_k=5,
                         knn_base_temperature=0.2,
                         support_labels=None,
                         contrastive_loss=False,
                         temperature=0.07,
                         lambda_cil=1.0,
                         router_temperature=0.5,
                         compute_router_supervision=True,
                         use_action_router=True,
                         record_loss=True):
        # select the complete samples (all 3 views present)
        x1_train, x2_train, x3_train = data_list
        flag = (mask == 1.0).all(dim=1)  # 简化：直接判断所有视图是否为1
        complete_label = label[flag]
        has_complete = complete_label.numel() > 0

        # Encode every masked training view once in train mode and reuse the
        # result for auxiliary/contrastive/imputation losses.
        # Previously complete rows were encoded here and all rows were encoded
        # again inside _compute_impute_loss_train (three redundant passes).
        feat_emb = {}
        aux_logit, aux_confidence, aux_label = dict(), dict(), dict()
        shared_train_latent = {}
        need_shared_encoding = (
            lambda_imputation > 0 or aux_loss or
            (has_complete and contrastive_loss))
        if need_shared_encoding:
            for view in range(self.views):
                full_x = (x1_train, x2_train, x3_train)[view]
                full_att = torch.sigmoid(self.att[view](full_x))
                full_raw = full_x * full_att
                full_raw = F.dropout(
                    F.relu(self.emb[view](full_raw)), self.dropout,
                    training=self.training)
                full_conf = self.aux_conf[view](full_raw)
                full_weighted = full_raw * full_conf
                shared_train_latent[view] = full_weighted
                observed_rows = torch.where(mask[:, view].bool())[0]
                if aux_loss and observed_rows.numel() > 0:
                    # Per-view supervision can use every patient for which that
                    # view is observed; requiring all three views to be complete
                    # silently disables confidence learning at high missingness.
                    observed_raw = full_raw[observed_rows]
                    aux_logit[view] = self.aux_clf[view](observed_raw)
                    aux_confidence[view] = full_conf[observed_rows]
                    aux_label[view] = label[observed_rows]
                if has_complete and contrastive_loss:
                    feat_emb[view] = full_weighted[flag]

        loss_dict = {}
        _log = lambda value: round(value.detach().item(), 4) if record_loss else None
        # 1. Loss between classifier and gt on all samples via two-stage imputation
        # [FIX] Use the full (unsliced) data as support set. data_list has already
        # been sliced to complete samples above, but support_labels covers the
        # entire support set — using sliced data here causes a length mismatch
        # in _build_class_prototype_bank's index_add_.
        #
        # [Quality note] When original_data_list is None, the fallback uses
        # x1_train/x2_train/x3_train which are the MASKED data (zeros at missing
        # views). This degrades support quality:
        #   - KNN donors' missing-target views are zero → imputed latent is biased.
        #   - Prototype bank includes encoder bias from zero-input views.
        #   - Router teacher supervision sees erroneous support latent.
        # The Trainer always supplies the configured support set (masked by
        # default), so this fallback is only used by external callers.
        if original_data_list is None:
            import warnings
            warnings.warn(
                "original_data_list is None — support will use masked data "
                "(zeros at missing views). KNN/Prototype quality may degrade. "
                "Pass original (unmasked) data as original_data_list.",
                stacklevel=2,
            )
            original_data_list = [x1_train, x2_train, x3_train]

        # [Design note] infer_on_missing runs in eval() mode for deterministic
        # imputation (dropout disabled). This ensures stable KNN/Router/action
        # decisions. The imputation loss (_compute_impute_loss_train) is separately
        # recomputed in train() mode with dropout for proper regularization.
        was_training = self.training
        if was_training:
            self.eval()
        try:
            # [P6] AMP compatible: latent buffers now use new_zeros() to match dtype
            MMlogit_all = self.infer_on_missing(
                [x1_train, x2_train, x3_train],
                mask,
                support_data_list=original_data_list,
                support_labels=support_labels,
                support_mask=support_mask,
                use_cross_sample_impute=use_cross_sample_impute,
                use_prototype_bank=use_prototype_bank,
                cross_sample_k=cross_sample_k,
                knn_base_temperature=knn_base_temperature,
                exclude_self=True,
                return_latents=False,
                compute_router_loss=(lambda_imputation > 0 and
                                     compute_router_supervision),
                router_temperature=router_temperature,
                use_action_router=(
                    use_action_router and lambda_imputation > 0),
                build_all_candidates=(
                    compute_router_supervision and use_action_router
                    and lambda_imputation > 0 and label is not None),
                cache_teacher_latent=lambda_imputation > 0,
            )
        finally:
            if was_training:
                self.train()

        # MMlogit_all is now a single tensor (return_latents=False)
        clf_loss = torch.mean(self.criterion(MMlogit_all, label))
        MMLoss = clf_loss
        loss_dict["clf"] = _log(clf_loss)

        # One umbrella objective trains candidate reconstruction, expert
        # reliability and the classification-utility action head.  Only
        # lambda_imputation weights this complete subsystem.
        imputation_terms = []
        if original_data_list is not None and lambda_imputation > 0:
            # [OPT-③] Reuse support_latent from infer_on_missing as Teacher (saves 3 encode calls)
            _cached_teacher = getattr(self, '_last_support_latent', None)
            rec_loss = self._compute_impute_loss_train(
                [x1_train, x2_train, x3_train], mask, original_data_list,
                teacher_latent=_cached_teacher,
                source_latent=shared_train_latent)
            imputation_terms.append(rec_loss)
            loss_dict["imputation_rec_raw"] = _log(rec_loss)

        # Counterfactual expert-reliability KL produced inside infer_on_missing.
        if hasattr(self, '_last_router_loss') and self._last_router_loss is not None:
            expert_router_loss = self._last_router_loss
            imputation_terms.append(expert_router_loss)
            loss_dict["imputation_expert_router_raw"] = _log(
                expert_router_loss)
            self._last_router_loss = None

        # The action head uses the same shared Router trunk.  Its teacher is a
        # distribution over the downstream CE of skip / impute-first /
        # impute-second.  No selector weight or imputation-cost coefficient is
        # needed because every non-skip action inserts exactly one view.
        if (compute_router_supervision and use_action_router
                and lambda_imputation > 0 and label is not None):
            action_router_loss = self.compute_action_router_loss(
                mask, label, tau=router_temperature)
            imputation_terms.append(action_router_loss)
            loss_dict["imputation_action_router_raw"] = _log(
                action_router_loss)

        if imputation_terms:
            unified_imputation = torch.stack(imputation_terms).sum()
            weighted_imputation = lambda_imputation * unified_imputation
            MMLoss = MMLoss + weighted_imputation
            loss_dict["imputation"] = _log(weighted_imputation)

        if aux_loss and aux_logit:
            aux_losses = []
            for view in sorted(aux_logit):
                pred = F.softmax(aux_logit[view], dim=1)
                labels_v = aux_label[view]
                p_target = torch.gather(
                    input=pred, dim=1,
                    index=labels_v.unsqueeze(dim=1)).view(-1).detach()
                # 置信度损失：MSE(aux_conf_MLP, TCP) + L_Cls(aux_clf)
                # 与 GREMI 公式5 相同——MSE 校准置信度，L_Cls 强制判别性特征提取
                confidence_loss = (
                    F.mse_loss(aux_confidence[view].view(-1), p_target)
                    + self.criterion(aux_logit[view], labels_v).mean())
                MMLoss = MMLoss + lambda_al * confidence_loss
                aux_losses.append(_log(lambda_al * confidence_loss))
            loss_dict['aux_clf'] = aux_losses

        # [FIX-Issue2] Guard contrastive loss: feat_emb is computed only on
        # complete samples. When none exist, info_nce_loss on empty tensors
        # produces NaN.
        if contrastive_loss and lambda_cil > 0 and has_complete:
            # [PERF-8] Subsample when complete batch is large — InfoNCE builds
            # an O(N²) similarity matrix. 256 samples keep it fast while
            # maintaining gradient quality.
            # [NOTE] Random truncation alters the negative-sample set each step,
            # so this is a stochastic approximation, NOT a strictly equivalent
            # speed-up. Bias is small when _cil_max >> num_class.
            _cil_n = feat_emb[0].shape[0]
            _cil_max = 256
            if _cil_n > _cil_max:
                _cil_perm = torch.randperm(_cil_n, device=feat_emb[0].device)[:_cil_max]
                _cil_emb = [feat_emb[v][_cil_perm] for v in range(self.views)]
            else:
                _cil_emb = [feat_emb[v] for v in range(self.views)]
            loss_cil_ab = info_nce_loss(_cil_emb[0], _cil_emb[1], temperature=temperature)
            loss_cil_ac = info_nce_loss(_cil_emb[0], _cil_emb[2], temperature=temperature)
            loss_cil_bc = info_nce_loss(_cil_emb[1], _cil_emb[2], temperature=temperature)
            loss_cil = loss_cil_ab + loss_cil_ac + loss_cil_bc
            weighted_cil = lambda_cil * loss_cil
            MMLoss = MMLoss + weighted_cil
            loss_dict['cil'] = _log(weighted_cil)

        return MMLoss, None, loss_dict

    # [FIX-Bug2] encode 返回 (weighted, raw)：
    #   weighted = feat_emb * aux_confidence（用于分类、插补、KNN 等下游任务）
    #   raw = 仅经过 attention + embedding + dropout，未乘 aux_confidence
    #          （用于 router 特征构建，避免 source_conf 双重加权）
    def encode(self, data_x, f_att, f_emb, f_aux_conf):
        att_score = torch.sigmoid(f_att(data_x))
        feat_emb = data_x * att_score
        feat_emb = F.dropout(F.relu(f_emb(feat_emb)), self.dropout, training=self.training)
        raw_emb = feat_emb  # pre-confidence embedding
        # f_aux_conf 现在是 MLP（内部含 Sigmoid），直接输出 [0,1]
        aux_confidence = f_aux_conf(feat_emb)
        feat_emb = feat_emb * aux_confidence
        # [P5] 返回 confidence 避免调用方重复运行 aux_conf
        return feat_emb, raw_emb, aux_confidence

    def _compute_impute_loss_train(self, data_list, mask, original_data_list,
                                   teacher_latent=None, source_latent=None):
        """Compute pairwise-overlap imputation loss in train mode.

        Trains cross-view predictors on ALL samples where both source and target
        views are present (pairwise overlap), not just on missing subsets.
        This provides significantly more training signal than the old missing-only
        approach, especially at low missing rates.

        LayerNorm replaces BatchNorm in Prediction/aux_conf modules, so no
        eval/finally workaround is needed (LN is sample-wise, batch-size agnostic).

        Args:
            teacher_latent: optional cached eval-mode encoding from infer_on_missing's
                support_latent. If provided, skips redundant Teacher encoding (saves 3
                encode forward passes). [OPT-③]
            source_latent: optional shared train-mode encodings. If provided,
                skips three more encoder passes.
        """
        mask_version = tensor_version_or_none(mask)
        overlap_cached = (
            mask_version is not None and
            getattr(self, '_overlap_cache_mask', None) is mask and
            getattr(self, '_overlap_cache_version', None) == mask_version)
        if overlap_cached:
            ab_rows, ac_rows, bc_rows = self._overlap_cache_rows
        else:
            present = mask == 1
            ab_rows = torch.where(present[:, 0] & present[:, 1])[0]
            ac_rows = torch.where(present[:, 0] & present[:, 2])[0]
            bc_rows = torch.where(present[:, 1] & present[:, 2])[0]
            self._overlap_cache_mask = mask
            self._overlap_cache_version = mask_version
            self._overlap_cache_rows = (ab_rows, ac_rows, bc_rows)

        # [OPT-③] Reuse cached teacher if available (from infer_on_missing's support_latent)
        if teacher_latent is not None:
            teacher = [teacher_latent[v] for v in range(self.views)]
        else:
            # [FIX-Bug3] Teacher 编码必须在 eval 模式下进行（关闭 dropout），
            # 否则 dropout 随机噪声使 teacher target 不稳定，导致插补损失震荡。
            was_training = self.training
            if was_training:
                self.eval()
            try:
                with torch.no_grad():
                    teacher = [
                        self.encode(original_data_list[v], self.att[v], self.emb[v], self.aux_conf[v])[0]
                        for v in range(self.views)
                    ]
            finally:
                if was_training:
                    self.train()

        imp_losses = []

        # [OPT-A] Encode each view's full data ONCE in train mode (with gradients),
        # then index into results for pairwise losses. Saves 3 encode forward passes.
        # For pairwise overlap samples (e.g. A present & B present), the masked
        # data == original data for those views, so indexing gives identical results.
        src_enc = ([source_latent[v] for v in range(self.views)]
                   if source_latent is not None else [
                       self.encode(
                           data_list[v], self.att[v], self.emb[v],
                           self.aux_conf[v])[0]
                       for v in range(self.views)
                   ])

        # --- Pairwise overlap: A↔B ---
        if ab_rows.numel() > 0:
            imp_losses.append(F.mse_loss(
                self.a2b(src_enc[0][ab_rows])[0], teacher[1][ab_rows]))
            imp_losses.append(F.mse_loss(
                self.b2a(src_enc[1][ab_rows])[0], teacher[0][ab_rows]))

        # --- Pairwise overlap: A↔C ---
        if ac_rows.numel() > 0:
            imp_losses.append(F.mse_loss(
                self.a2c(src_enc[0][ac_rows])[0], teacher[2][ac_rows]))
            imp_losses.append(F.mse_loss(
                self.c2a(src_enc[2][ac_rows])[0], teacher[0][ac_rows]))

        # --- Pairwise overlap: B↔C ---
        if bc_rows.numel() > 0:
            imp_losses.append(F.mse_loss(
                self.b2c(src_enc[1][bc_rows])[0], teacher[2][bc_rows]))
            imp_losses.append(F.mse_loss(
                self.c2b(src_enc[2][bc_rows])[0], teacher[1][bc_rows]))

        if len(imp_losses) > 0:
            return torch.stack(imp_losses).mean()
        return torch.tensor(0.0, device=data_list[0].device)


    def _get_predictor(self, src_view, target_view):
        """Get the appropriate cross-view predictor module."""
        return self._predictor_map[(src_view, target_view)]

    def _compute_router_loss(self, data_list, mask, support_latent,
                             support_labels, impute_indices,
                             cross_sample_k, knn_base_temperature, exclude_self,
                             prototype_bank, tau=0.5,
                             original_data_list=None, support_norm=None,
                             support_confidence=None,
                             present_latent_cache=None, position_map=None,
                             confidence_cache=None,
                             use_cross_sample_impute=True,
                             pattern_ints=None, pattern_meta=None,
                             donor_masks=None,
                             support_mask=None,
                             knn_sim=None):
        """Compute router training loss with counterfactual regret supervision.

        For each imputed sample, temporarily mask each present view in turn to
        create a synthetic missing scenario. Compute each expert's imputation
        error against ground truth, then use KL divergence with soft targets
        (softmax(-E/tau)) to teach the router to weight better experts higher.

        Four experts:
        - CV (cross-view predictor): predicts missing view from remaining views
        - KNN (source-space key-value KNN): finds neighbors in source view space
        - Prototype (class prototype): uses class prototypes weighted by probs
        - Skip (no imputation): zero vector for missing view (safe rejection)

        Skip expert: when imputation quality is poor, not imputing at all can be
        better than bad imputation (Deep Safe IMVC, ICML 2022).

        Args:
            tau: temperature for soft target distribution (lower = sharper)
            original_data_list: [FIX-Bug1] unmasked original data for router feature
                encoding. Ensures train-test consistency (at test time, present views
                have full data, not zeroed-out masked data).
        """
        # Router supervision only needs latents for views that are observed in
        # each sample. Reuse infer_on_missing's differentiable encode cache when
        # available; this removes three duplicate full-data encoder passes.
        use_cache = (present_latent_cache is not None and position_map is not None
                     and confidence_cache is not None)
        if use_cache:
            def gather_latent(view, indices):
                return present_latent_cache[view][position_map[view, indices]]

            def gather_confidence(view, indices):
                return confidence_cache[view][position_map[view, indices]]
        else:
            if original_data_list is None:
                original_data_list = data_list
            full_latent, full_confidence = {}, {}
            for view in range(self.views):
                full_latent[view], _, confidence = self.encode(
                    original_data_list[view], self.att[view], self.emb[view],
                    self.aux_conf[view])
                full_confidence[view] = confidence.squeeze(-1)

            def gather_latent(view, indices):
                return full_latent[view][indices]

            def gather_confidence(view, indices):
                return full_confidence[view][indices]

        if use_cross_sample_impute and support_norm is None:
            support_norm = {
                view: safe_l2_normalize(
                    support_latent[view], dim=1, min_norm=1e-3)
                for view in range(self.views)
            }

        if pattern_ints is None:
            present_bits = (mask[impute_indices] == 1).long()
            pattern_ints = (present_bits[:, 0] + 2 * present_bits[:, 1]
                            + 4 * present_bits[:, 2])
        if pattern_meta is None:
            pattern_meta = {}
            # Valid incomplete patterns are fixed integers 1..6. This avoids
            # unique().tolist(), which synchronizes CUDA with the CPU.
            for pattern in range(1, 1 << self.views):
                if torch.where(pattern_ints == pattern)[0].numel() == 0:
                    continue
                pattern_meta[pattern] = (
                    [v for v in range(self.views) if not (pattern & (1 << v))],
                    [v for v in range(self.views) if pattern & (1 << v)],
                )
        router_losses = []
        hidden = self.hidden_dim[0]
        # Keep this feature on a scale comparable with the other router inputs.
        n_support = math.log1p(float(support_latent[0].shape[0]))

        # [FIX-Critical] Do NOT pre-compute a unified _cv_cache here — at this
        # point gather_latent still uses present_latent_cache/position_map which
        # has -1 for unobserved positions, causing IndexError on empty caches or
        # silently wrong data (last-row fallback). Instead:
        # - Existing-pattern loop: call gather_latent() directly per group (safe
        #   because group_indices always correspond to observed positions).
        # - Synthesis loop: build a separate support_cv_cache using support_latent
        #   AFTER the gather override is applied (around line 940).

        # At most six batches: three one-missing patterns x two synthetic targets.
        for pattern, (_, present_views) in pattern_meta.items():
            if len(present_views) < 2:
                continue
            group_positions = torch.where(pattern_ints == pattern)[0]
            group_indices = impute_indices[group_positions]
            group_size = group_indices.shape[0]

            for synthetic_missing_v in present_views:
                remaining_present = [v for v in present_views
                                     if v != synthetic_missing_v]
                source_v = remaining_present[0]

                # [FIX-Issue5] Filter out samples where target view is masked
                # in support set — their gt_latent is zero-input encoder bias.
                _grp = group_indices
                if support_mask is not None:
                    _sm_dev = support_mask.to(group_indices.device)
                    _valid_target = _sm_dev[group_indices, synthetic_missing_v].bool()
                    _grp = group_indices[_valid_target]
                    if _grp.numel() == 0:
                        continue
                group_size = _grp.shape[0]

                # Teacher target is the same deterministic eval-mode encoding
                # already stored in support_latent; it remains detached.
                gt_latent = support_latent[synthetic_missing_v][_grp].detach()

                # [FIX-Critical] Call gather_latent directly — safe because
                # group_indices always correspond to observed batch positions.
                cv_parts = []
                for src_v in remaining_present:
                    predictor = self._get_predictor(src_v, synthetic_missing_v)
                    cv_parts.append(
                        predictor(gather_latent(src_v, _grp))[0])
                cv_pred = (torch.stack(cv_parts).mean(dim=0)
                           if len(cv_parts) > 1 else cv_parts[0])
                cv_error = (cv_pred - gt_latent).square().mean(dim=1)

                if use_cross_sample_impute:
                    query_src = gather_latent(source_v, _grp)
                    self_indices = _grp if exclude_self else None
                    _dm_router = (donor_masks.get((source_v, synthetic_missing_v))
                                  if donor_masks else None)
                    _pc_router = None
                    if (knn_sim and source_v in knn_sim
                            and position_map is not None):
                        _obs_pos = position_map[source_v, _grp]
                        _pc_router = self._slice_knn_similarity(
                            knn_sim, source_v, _obs_pos)
                    knn_pred, knn_stats = self._batched_knn_search(
                        query_src, support_norm[source_v], support_latent[source_v],
                        top_k=cross_sample_k, temperature=knn_base_temperature,
                        aggregate_targets=[support_latent[synthetic_missing_v]],
                        exclude_self_indices=self_indices,
                        donor_mask=_dm_router,
                        return_stats=True,
                        precomputed_sim=_pc_router)
                    knn_error = (knn_pred - gt_latent).square().mean(dim=1)
                else:
                    knn_error = cv_error
                    knn_stats = (None, None, None)

                if prototype_bank is not None:
                    cf_parts = []
                    for view in range(self.views):
                        if view in remaining_present:
                            cf_parts.append(gather_latent(view, _grp))
                        else:
                            cf_parts.append(gt_latent.new_zeros(group_size, hidden))
                    cf_probs = F.softmax(
                        self.MMClasifier(torch.cat(cf_parts, dim=1)), dim=1)
                    # [FIX-Issue2] Leave-one-out prototype: remove each sample's
                    # own latent from its class prototype to prevent data leakage.
                    sample_latent_target = support_latent[synthetic_missing_v][
                        _grp].detach()
                    proto_pred = self._loo_proto_pred(
                        cf_probs, prototype_bank[synthetic_missing_v],
                        support_labels, sample_latent_target, _grp,
                        class_counts=prototype_bank.get('_class_counts'),
                        target_view=synthetic_missing_v)
                    proto_error = (proto_pred - gt_latent).square().mean(dim=1)
                    proto_entropy = -(cf_probs.float() * (cf_probs.float() + 1e-10).log()).sum(dim=1)
                else:
                    proto_error = gt_latent.new_full((group_size,), float('inf'))
                    proto_entropy = gt_latent.new_zeros(group_size)

                skip_error = gt_latent.square().mean(dim=1)

                cf_missing_pattern = (mask[_grp] == 0).to(gt_latent.dtype)
                cf_missing_pattern = cf_missing_pattern.clone()
                cf_missing_pattern[:, synthetic_missing_v] = 1.0

                source_conf = gt_latent.new_zeros(group_size, self.views)
                for view in remaining_present:
                    source_conf[:, view] = gather_confidence(view, _grp)
                pred_conf = source_conf[:, remaining_present].mean(dim=1)

                if knn_stats[0] is None:
                    knn_density = gt_latent.new_zeros(group_size)
                    knn_eff_k = gt_latent.new_zeros(group_size)
                    knn_margin = gt_latent.new_zeros(group_size)
                else:
                    knn_density, knn_eff_k, knn_margin = knn_stats

                router_features = torch.cat([
                    cf_missing_pattern,
                    source_conf,
                    torch.stack([pred_conf, knn_density, knn_margin], dim=1),
                    torch.stack([
                        knn_eff_k,
                        proto_entropy,
                        gt_latent.new_full((group_size,), n_support),
                    ], dim=1),
                ], dim=1)
                router_logits = self.router(router_features)
                errors = torch.stack(
                    [cv_error, knn_error, proto_error, skip_error], dim=1)
                with torch.no_grad():
                    soft_target = F.softmax(-errors / tau, dim=1)
                per_sample_kl = F.kl_div(
                    F.log_softmax(router_logits, dim=1), soft_target,
                    reduction='none').sum(dim=1)
                router_losses.append(per_sample_kl)

        # [FIX-Issue2 + Issue5] Always synthesize supervision to ensure router
        # receives expert-error signals. When the batch has complete samples,
        # synthesize from them. When it doesn't (e.g. high missing rate), fall
        # back to the support set which contains the full unmasked data.
        complete_mask = (mask == 1).all(dim=1)
        complete_indices = torch.where(complete_mask)[0]
        # [FIX-Critical] Track whether we are synthesizing from support set.
        # When True, gather_latent/gather_confidence must use support_latent/
        # support_confidence directly, NOT present_latent_cache/position_map
        # (which are indexed by batch positions and return wrong data for
        # support indices that don't correspond to any batch sample).
        _synthetic_from_support = False
        if complete_indices.numel() == 0:
            _synthetic_from_support = True
            complete_indices = torch.arange(
                support_latent[0].shape[0], device=mask.device)
            # [FIX-Issue5] When using support indices, filter to samples
            # where ALL views are valid in support_mask. Masked samples
            # have zero-input encoder bias and are invalid as teacher GT.
            if support_mask is not None:
                _sm_dev = support_mask.to(mask.device)
                _all_valid = _sm_dev.bool().all(dim=1)
                complete_indices = complete_indices[_all_valid]
        if complete_indices.numel() > 0:
            # Subsample if too many complete samples (avoid excessive compute)
            max_synthetic = 128  # per pattern, to balance across 6 patterns
            if complete_indices.numel() > max_synthetic:
                perm = torch.randperm(complete_indices.numel(),
                                      device=complete_indices.device)
                complete_indices = complete_indices[perm[:max_synthetic]]

            # [FIX-Critical] When synthesizing from support set, override gather
            # functions to index directly into support_latent/support_confidence.
            if _synthetic_from_support:
                def gather_latent(view, indices):
                    return support_latent[view][indices]
                def gather_confidence(view, indices):
                    return support_confidence[view][indices]

            # [PERF-3] Build support CV cache ONLY for complete_indices (the only
            # indices actually consumed downstream).  Wrapped in torch.no_grad()
            # because this cache feeds router-loss teacher supervision only.
            support_cv_cache = {}
            with torch.no_grad():
                for src_v in range(self.views):
                    for tgt_v in range(self.views):
                        if src_v != tgt_v:
                            support_cv_cache[(src_v, tgt_v)] = self._get_predictor(
                                src_v, tgt_v)(
                                    support_latent[src_v][complete_indices])[0]

            # --- Single-missing patterns (3 patterns: mask A, mask B, mask C) ---
            for synthetic_missing_v in range(self.views):
                remaining_present = [v for v in range(self.views)
                                     if v != synthetic_missing_v]
                source_v = remaining_present[0]

                gt_latent = support_latent[synthetic_missing_v][
                    complete_indices].detach()

                # [PERF-3] Use support CV cache (already sliced to complete_indices).
                cv_parts = []
                for src_v in remaining_present:
                    cv_parts.append(
                        support_cv_cache[(src_v, synthetic_missing_v)])
                cv_pred = (torch.stack(cv_parts).mean(dim=0)
                           if len(cv_parts) > 1 else cv_parts[0])
                cv_error = (cv_pred - gt_latent).square().mean(dim=1)

                if use_cross_sample_impute:
                    query_src = gather_latent(source_v, complete_indices)
                    _dm_synth = (donor_masks.get((source_v, synthetic_missing_v))
                                 if donor_masks else None)
                    knn_pred, knn_stats = self._batched_knn_search(
                        query_src, support_norm[source_v],
                        support_latent[source_v],
                        top_k=cross_sample_k,
                        temperature=knn_base_temperature,
                        aggregate_targets=[
                            support_latent[synthetic_missing_v]],
                        exclude_self_indices=(
                            complete_indices if exclude_self else None),
                        donor_mask=_dm_synth,
                        return_stats=True)
                    knn_error = (knn_pred - gt_latent).square().mean(dim=1)
                else:
                    knn_error = cv_error
                    knn_stats = (None, None, None)

                if prototype_bank is not None:
                    cf_parts = []
                    for view in range(self.views):
                        if view in remaining_present:
                            cf_parts.append(
                                gather_latent(view, complete_indices))
                        else:
                            cf_parts.append(gt_latent.new_zeros(
                                complete_indices.shape[0], hidden))
                    cf_probs = F.softmax(
                        self.MMClasifier(torch.cat(cf_parts, dim=1)), dim=1)
                    # [FIX-Issue2] Leave-one-out prototype for single-missing synthesis
                    proto_pred = self._loo_proto_pred(
                        cf_probs, prototype_bank[synthetic_missing_v],
                        support_labels, gt_latent, complete_indices,
                        class_counts=prototype_bank.get('_class_counts'),
                        target_view=synthetic_missing_v)
                    proto_error = (proto_pred - gt_latent).square().mean(dim=1)
                    proto_entropy = -(cf_probs.float() * (cf_probs.float() + 1e-10).log()).sum(dim=1)
                else:
                    proto_error = gt_latent.new_full(
                        (complete_indices.shape[0],), float('inf'))
                    proto_entropy = gt_latent.new_zeros(complete_indices.shape[0])

                skip_error = gt_latent.square().mean(dim=1)

                # Build router features for synthetic single-missing samples
                cf_missing_pattern = gt_latent.new_zeros(3)
                cf_missing_pattern[synthetic_missing_v] = 1.0
                cf_missing_pattern = cf_missing_pattern.unsqueeze(0).expand(
                    complete_indices.shape[0], 3)

                source_conf = gt_latent.new_zeros(
                    complete_indices.shape[0], self.views)
                for view in remaining_present:
                    source_conf[:, view] = gather_confidence(
                        view, complete_indices)
                pred_conf = source_conf[:, remaining_present].mean(dim=1)

                if knn_stats[0] is None:
                    knn_density = gt_latent.new_zeros(complete_indices.shape[0])
                    knn_eff_k = gt_latent.new_zeros(complete_indices.shape[0])
                    knn_margin = gt_latent.new_zeros(complete_indices.shape[0])
                else:
                    knn_density, knn_eff_k, knn_margin = knn_stats

                router_features = torch.cat([
                    cf_missing_pattern,
                    source_conf,
                    torch.stack([pred_conf, knn_density, knn_margin], dim=1),
                    torch.stack([
                        knn_eff_k,
                        proto_entropy,
                        gt_latent.new_full(
                            (complete_indices.shape[0],), n_support),
                    ], dim=1),
                ], dim=1)
                router_logits = self.router(router_features)
                errors = torch.stack(
                    [cv_error, knn_error, proto_error, skip_error], dim=1)
                with torch.no_grad():
                    soft_target = F.softmax(-errors / tau, dim=1)
                per_sample_kl = F.kl_div(
                    F.log_softmax(router_logits, dim=1), soft_target,
                    reduction='none').sum(dim=1)
                router_losses.append(per_sample_kl)

            # --- Double-missing patterns (3 patterns: keep A, keep B, keep C) ---
            for present_v in range(self.views):
                missing_views = [v for v in range(self.views)
                                 if v != present_v]
                source_v = present_v  # only one present view as source

                # [FIX-Issue6] Single KNN search for both targets, matching
                # inference behavior. Inference does one search with
                # aggregate_targets=[support_latent[v] for v in _miss].
                query_src = gather_latent(source_v, complete_indices)
                if use_cross_sample_impute:
                    # Per-target donor masks for double-missing
                    _dm_list = None
                    if donor_masks:
                        _dm_list = [
                            donor_masks.get((source_v, _tv))
                            for _tv in missing_views
                        ]
                    knn_preds, knn_stats_2 = self._batched_knn_search(
                        query_src, support_norm[source_v],
                        support_latent[source_v],
                        top_k=cross_sample_k,
                        temperature=knn_base_temperature,
                        aggregate_targets=[support_latent[v]
                                           for v in missing_views],
                        exclude_self_indices=(
                            complete_indices if exclude_self else None),
                        donor_mask=_dm_list,
                        return_stats=True)
                    if len(missing_views) == 1:
                        knn_preds = [knn_preds]
                else:
                    knn_stats_2 = (None, None, None)

                # [PERF-1] Compute cf_probs ONCE per present_v — it only depends
                # on which view is present, not on which views are missing.
                # Previously computed twice (once per target_v), now once.
                cf_probs_dm = None
                if prototype_bank is not None:
                    cf_parts_dm = []
                    for view in range(self.views):
                        if view == present_v:
                            cf_parts_dm.append(
                                gather_latent(view, complete_indices))
                        else:
                            cf_parts_dm.append(
                                support_latent[view].new_zeros(
                                    complete_indices.shape[0], hidden))
                    cf_probs_dm = F.softmax(
                        self.MMClasifier(torch.cat(cf_parts_dm, dim=1)), dim=1)

                # For each missing view, compute expert errors
                cv_errors = []
                knn_errors = []
                proto_errors_list = []
                skip_errors = []

                for idx, target_v in enumerate(missing_views):
                    gt_latent = support_latent[target_v][
                        complete_indices].detach()

                    # CV: predict target from source [PERF-3] (already sliced)
                    cv_pred = support_cv_cache[(source_v, target_v)]
                    cv_errors.append((cv_pred - gt_latent).square().mean(dim=1))

                    # KNN: use pre-computed prediction from single search
                    if use_cross_sample_impute:
                        knn_errors.append(
                            (knn_preds[idx] - gt_latent).square().mean(dim=1))
                    else:
                        knn_errors.append(cv_errors[-1])

                    # Prototype [PERF-1] Reuse cf_probs; leave-one-out to prevent leakage.
                    if prototype_bank is not None:
                        proto_pred = self._loo_proto_pred(
                            cf_probs_dm, prototype_bank[target_v],
                            support_labels, gt_latent, complete_indices,
                            class_counts=prototype_bank.get('_class_counts'),
                            target_view=target_v)
                        proto_errors_list.append(
                            (proto_pred - gt_latent).square().mean(dim=1))
                        if idx == 0:
                            proto_entropy_2 = -(cf_probs_dm.float() * (cf_probs_dm.float() + 1e-10).log()).sum(dim=1)
                    else:
                        proto_errors_list.append(gt_latent.new_full(
                            (complete_indices.shape[0],), float('inf')))
                        if idx == 0:
                            proto_entropy_2 = gt_latent.new_zeros(complete_indices.shape[0])

                    # Skip
                    skip_errors.append(gt_latent.square().mean(dim=1))

                # Average errors across the two missing views for router supervision
                cv_error_avg = torch.stack(cv_errors).mean(dim=0)
                knn_error_avg = torch.stack(knn_errors).mean(dim=0)
                proto_error_avg = torch.stack(proto_errors_list).mean(dim=0)
                skip_error_avg = torch.stack(skip_errors).mean(dim=0)

                # Build router features for synthetic double-missing samples
                cf_missing_pattern = gt_latent.new_ones(3)
                cf_missing_pattern[present_v] = 0.0
                cf_missing_pattern = cf_missing_pattern.unsqueeze(0).expand(
                    complete_indices.shape[0], 3)

                source_conf = gt_latent.new_zeros(
                    complete_indices.shape[0], self.views)
                source_conf[:, present_v] = gather_confidence(
                    present_v, complete_indices)
                pred_conf = source_conf[:, present_v:present_v+1].squeeze(1)

                if knn_stats_2 is None or (isinstance(knn_stats_2, tuple)
                                            and knn_stats_2[0] is None):
                    knn_density = gt_latent.new_zeros(complete_indices.shape[0])
                    knn_eff_k = gt_latent.new_zeros(complete_indices.shape[0])
                    knn_margin = gt_latent.new_zeros(complete_indices.shape[0])
                elif isinstance(knn_stats_2, list):
                    # Per-target stats: average across targets
                    _densities, _eff_ks, _margins = [], [], []
                    for _st in knn_stats_2:
                        if _st[0] is None:
                            _densities.append(gt_latent.new_zeros(
                                complete_indices.shape[0]))
                            _eff_ks.append(gt_latent.new_zeros(
                                complete_indices.shape[0]))
                            _margins.append(gt_latent.new_zeros(
                                complete_indices.shape[0]))
                        else:
                            _densities.append(_st[0])
                            _eff_ks.append(_st[1])
                            _margins.append(_st[2])
                    knn_density = torch.stack(_densities).mean(dim=0)
                    knn_eff_k = torch.stack(_eff_ks).mean(dim=0)
                    knn_margin = torch.stack(_margins).mean(dim=0)
                else:
                    knn_density, knn_eff_k, knn_margin = knn_stats_2

                router_features = torch.cat([
                    cf_missing_pattern,
                    source_conf,
                    torch.stack([pred_conf, knn_density, knn_margin], dim=1),
                    torch.stack([
                        knn_eff_k,
                        proto_entropy_2,
                        gt_latent.new_full(
                            (complete_indices.shape[0],), n_support),
                    ], dim=1),
                ], dim=1)
                router_logits = self.router(router_features)
                errors = torch.stack(
                    [cv_error_avg, knn_error_avg, proto_error_avg, skip_error_avg], dim=1)
                with torch.no_grad():
                    soft_target = F.softmax(-errors / tau, dim=1)
                per_sample_kl = F.kl_div(
                    F.log_softmax(router_logits, dim=1), soft_target,
                    reduction='none').sum(dim=1)
                router_losses.append(per_sample_kl)

        if router_losses:
            return torch.cat(router_losses).mean()
        return mask.new_zeros(())

    def compute_action_router_loss(self, mask, labels, tau=0.5):
        """Train the shared Router action head with classification utility.

        Uses router-fused candidates (CV+KNN+Proto+Skip) for GT gain computation,
        matching the exact same candidate construction as infer_on_missing at inference.

        Handles BOTH 1-missing and 2-missing samples (at most impute one):
        - 1-missing: 2 actions {skip (baseline), impute the missing view}
        - 2-missing: 3 actions {skip both, impute v1 only, impute v2 only}

        The counterfactual teacher is

            q(a|i) = softmax(-CE(f(Z_i^a), y_i) / tau).

        Skip is an ordinary competing action.  Every non-skip action inserts
        exactly one view, so no hand-tuned imputation-cost coefficient is
        necessary.

        Args:
            mask: [N, 3] binary mask (1=present, 0=missing)
            labels: [N] ground truth labels
            tau: temperature of the counterfactual action teacher

        Returns:
            KL divergence between teacher utilities and action logits
        """
        if tau <= 0:
            raise ValueError("action-router temperature must be positive")
        if (not hasattr(self, '_last_support_latent')
                or self._last_support_latent is None):
            return mask.new_zeros(())

        impute_indices = self._last_impute_indices     # [num_impute], detached
        support_latent = self._last_support_latent     # dict, detached

        candidate_cache = getattr(self, '_last_candidate_cache_t', None)
        candidate_valid = getattr(self, '_last_candidate_cache_valid', None)
        cached_action_logits = getattr(self, '_last_action_logits', None)
        if (candidate_cache is None or candidate_valid is None
                or cached_action_logits is None):
            return mask.new_zeros(())

        mask_impute = mask[impute_indices]
        missing = mask_impute == 0
        missing_count = missing.sum(dim=1)
        valid_candidates = ((~missing) | candidate_valid).all(dim=1)
        valid_samples = valid_candidates & ((missing_count == 1) | (missing_count == 2))
        valid_pos = torch.where(valid_samples)[0]
        if valid_pos.numel() == 0:
            return mask.new_zeros(())

        sample_indices = impute_indices[valid_pos]
        sample_labels = labels[sample_indices]
        sample_mask = mask_impute[valid_pos]
        sample_missing = missing[valid_pos]
        sample_candidates = candidate_cache[valid_pos]
        hd = self.hidden_dim[0]

        # [M, 3, H], with missing views explicitly zeroed.
        present_parts = torch.stack(
            [support_latent[v][sample_indices] for v in range(self.views)], dim=1)
        present_parts = present_parts * sample_mask.to(present_parts.dtype).unsqueeze(-1)
        flat_baseline = present_parts.reshape(sample_indices.shape[0], self.views * hd)

        # Three actions: skip, impute first missing, impute second missing.
        # Invalid action 2 for one-missing patients remains +inf and therefore
        # receives zero teacher probability.
        action_ce = flat_baseline.new_full(
            (sample_indices.shape[0], 3), float('inf'),
            dtype=torch.float32)

        # All utility targets are non-differentiable supervision.  Disable
        # classifier Dropout so repeated counterfactual evaluations of the same
        # sample produce a stable teacher distribution.
        classifier_was_training = self.MMClasifier.training
        if classifier_was_training:
            self.MMClasifier.eval()
        try:
            with torch.no_grad():
                baseline_logits = self.MMClasifier(flat_baseline)
                action_ce[:, 0] = F.cross_entropy(
                    baseline_logits, sample_labels, reduction='none')

                one_pos = torch.where(sample_missing.sum(dim=1) == 1)[0]
                if one_pos.numel() > 0:
                    one_parts = (
                        present_parts[one_pos] + sample_candidates[one_pos])
                    one_logits = self.MMClasifier(
                        one_parts.reshape(
                            one_pos.shape[0], self.views * hd))
                    action_ce[one_pos, 1] = F.cross_entropy(
                        one_logits, sample_labels[one_pos],
                        reduction='none')

                two_pos = torch.where(
                    sample_missing.sum(dim=1) == 2)[0]
                if two_pos.numel() > 0:
                    missing_two = sample_missing[two_pos]
                    candidate_two = sample_candidates[two_pos]
                    rank = missing_two.long().cumsum(dim=1)
                    first_mask = (rank == 1) & missing_two
                    second_mask = (rank == 2) & missing_two
                    base_two = present_parts[two_pos]
                    action_one = (
                        base_two
                        + candidate_two * first_mask.unsqueeze(-1))
                    action_two = (
                        base_two
                        + candidate_two * second_mask.unsqueeze(-1))
                    action_parts = torch.stack(
                        [action_one, action_two], dim=1)
                    counterfactual_logits = self.MMClasifier(
                        action_parts.reshape(-1, self.views * hd))
                    two_action_ce = F.cross_entropy(
                        counterfactual_logits,
                        sample_labels[two_pos].repeat_interleave(2),
                        reduction='none').view(two_pos.shape[0], 2)
                    action_ce[two_pos, 1:] = two_action_ce

                soft_target = F.softmax(-action_ce / tau, dim=1)
        finally:
            if classifier_was_training:
                self.MMClasifier.train()

        predicted_logits = cached_action_logits[valid_pos].float()
        one_missing = sample_missing.sum(dim=1) == 1
        predicted_logits = predicted_logits.clone()
        predicted_logits[one_missing, 2] = self._NEG_INF_CLAMP
        return F.kl_div(
            F.log_softmax(predicted_logits, dim=1),
            soft_target,
            reduction='batchmean')

    def _batched_knn_search(self, query_batch, support_norm, support_latent,
                            top_k=5, temperature=0.2,
                            aggregate_targets=None,
                            exclude_self_indices=None,
                            donor_mask=None,
                            return_stats=False,
                            precomputed_sim=None):
        """Batched KNN with pre-computed support_norm.

        Key optimization for 2-missing patterns: when aggregate_targets has
        multiple views, the same neighbor search is reused — only the
        aggregation target differs.

        Args:
            query_batch: [Q, dim] query embeddings (all from same source view).
            support_norm: [N, dim] pre-normalized support embeddings.
            support_latent: [N, dim] raw support embeddings (fallback when
                aggregate_targets is None).
            top_k: number of nearest neighbors.
            temperature: base temperature for adaptive softmax.
            aggregate_targets: list of [N, dim] tensors to aggregate from.
                If None, uses [support_latent]. If multiple, returns list of
                refined tensors sharing the same top_idx and weights.
            exclude_self_indices: [Q] long tensor of self-indices to mask.
            donor_mask: [N] bool tensor OR list of [N] bool tensors.
                Single mask: shared across all targets (single-target case).
                List of masks: one per target — each target gets independent
                topk with its own mask (double-missing per-target case).
                When None, no donor filtering is applied.
            return_stats: if True, also return (neighbor_var, eff_k, margin).
            precomputed_sim: optional [Q, N] cosine-similarity matrix. When
                supplied, normalization and query-support matrix multiplication
                are skipped; donor and self masks are still applied here.

        Returns:
            If aggregate_targets is None or len 1: [Q, dim] refined (or tuple with stats).
            If len > 1: list of [Q, dim] refined tensors (or tuple with stats if requested).
        """
        _nc = self._NEG_INF_CLAMP
        Q = query_batch.shape[0]
        N = support_norm.shape[0]
        targets = (aggregate_targets
                   if aggregate_targets is not None
                   else [support_latent])

        # [FIX] Q=0: nothing to do. top_k<=0: return zeros with correct (Q, dim) shape.
        if Q == 0:
            dummy = targets[0].new_zeros(0, targets[0].shape[1])
            if len(targets) > 1:
                dummy = [dummy] * len(targets)
            return (dummy, (None, None, None)) if return_stats else dummy
        if top_k <= 0:
            dummy = targets[0].new_zeros(Q, targets[0].shape[1])
            if len(targets) > 1:
                dummy = [dummy] * len(targets)
            return (dummy, (None, None, None)) if return_stats else dummy

        # [FIX] N<=1: check if single donor is valid for each query.
        # Returning query_batch as KNN "prediction" is semantically wrong —
        # KNN aggregates TARGET latents, not the source embedding itself.
        if N <= 1:
            if N == 0:
                dummy = targets[0].new_zeros(Q, targets[0].shape[1])
                if len(targets) > 1:
                    dummy = [dummy] * len(targets)
                return (dummy, (None, None, None)) if return_stats else dummy

            # N == 1: check if single donor is valid (self-exclusion + donor_mask)
            _donor_ok = torch.ones(Q, dtype=torch.bool, device=query_batch.device)
            # Check donor_mask — the single donor may be filtered out
            if isinstance(donor_mask, list):
                _dm_list_n1 = donor_mask
            elif donor_mask is not None:
                _dm_list_n1 = [donor_mask] * len(targets)
            else:
                _dm_list_n1 = [None] * len(targets)
            if exclude_self_indices is not None:
                _self_dev = exclude_self_indices.to(query_batch.device)
                _bounded = (_self_dev >= 0) & (_self_dev < N)
                _is_self = _bounded & (_self_dev == 0)
                _donor_ok = _donor_ok & ~_is_self

            # Per-target validity: each target may have its own donor_mask
            _tgt_ok_list = []
            for _dm_n1 in _dm_list_n1:
                _ok = _donor_ok.clone()
                if _dm_n1 is not None:
                    _dm_dev = _dm_n1.to(query_batch.device)
                    _ok = _ok & _dm_dev[0]  # single donor at index 0
                _tgt_ok_list.append(_ok)

            # Single donor: use its target latent where valid, zeros otherwise
            results = []
            for agg_latent, _ok in zip(targets, _tgt_ok_list):
                refined = torch.where(
                    _ok.unsqueeze(-1),
                    agg_latent.expand(Q, -1),
                    agg_latent.new_zeros(Q, agg_latent.shape[1]))
                results.append(refined)

            if len(targets) == 1:
                refined_out = results[0]
                stats = None
                if return_stats:
                    stats = (torch.zeros(Q, device=query_batch.device),
                             _tgt_ok_list[0].float(),
                             torch.zeros(Q, device=query_batch.device))
                return (refined_out, stats) if return_stats else refined_out
            if return_stats:
                return results, [
                    (torch.zeros(Q, device=query_batch.device), ok.float(),
                     torch.zeros(Q, device=query_batch.device))
                    for ok in _tgt_ok_list]
            return results

        # [FIX] Force FP32 for normalize + similarity to prevent gradient
        # Inf/NaN from FP16 backward pass.
        with torch.amp.autocast('cuda', enabled=False):
            if precomputed_sim is not None:
                if (precomputed_sim.ndim != 2
                        or tuple(precomputed_sim.shape) != (Q, N)):
                    raise ValueError(
                        "precomputed_sim must have shape "
                        f"({Q}, {N}), got {tuple(precomputed_sim.shape)}")
                if precomputed_sim.device != query_batch.device:
                    raise ValueError(
                        "precomputed_sim must be on the same device as "
                        "query_batch")
                sim = precomputed_sim.float()
            else:
                query_norm = safe_l2_normalize(
                    query_batch, dim=1, min_norm=1e-3)
                # The cached support may have been produced inside CUDA autocast.
                # Explicit FP32 here prevents Float/Half matmul mismatches.
                support_norm32 = support_norm.float()
                sim = torch.mm(query_norm, support_norm32.t())  # [Q, N]

        # --- Self-exclusion with bounds check (FIX: >= 0 prevents -1 hiding last donor) ---
        self_idx = (exclude_self_indices.to(query_batch.device)
                    if exclude_self_indices is not None else None)
        if exclude_self_indices is not None:
            # Cached rows may be reused by other patterns; do not mask them
            # in place.
            if precomputed_sim is not None:
                sim = sim.clone()
            row_idx = torch.arange(Q, device=query_batch.device)
            valid = (self_idx >= 0) & (self_idx < N)
            sim[row_idx[valid], self_idx[valid]] = float("-inf")

        targets = (aggregate_targets
                   if aggregate_targets is not None
                   else [support_latent])

        # Normalize donor_mask to list (one per target)
        if isinstance(donor_mask, list):
            dm_list = donor_mask
        elif donor_mask is not None:
            dm_list = [donor_mask] * len(targets)
        else:
            dm_list = [None] * len(targets)

        # One uniform vectorized path for full, empty and mixed donor rows.
        # The previous three-way Python branching called .all()/.any()/.item(),
        # forcing multiple CUDA synchronizations for every KNN search. Invalid
        # donors receive -inf; after top-k their clamped scores underflow to zero
        # softmax weight whenever a valid donor exists. Rows with no donors are
        # explicitly zeroed by torch.where below.
        k = min(top_k, N)
        results = []
        all_stats = []

        for agg_latent, dm in zip(targets, dm_list):
            sim_t = sim.clone() if (dm is not None or len(targets) > 1) else sim
            base_valid = torch.full(
                (Q,), N, dtype=torch.long, device=query_batch.device)
            dm_dev = None
            if dm is not None:
                dm_dev = dm.to(query_batch.device, dtype=torch.bool)
                sim_t[:, ~dm_dev] = float("-inf")
                base_valid = dm_dev.sum().expand(Q).clone()

            if self_idx is not None:
                bounded = (self_idx >= 0) & (self_idx < N)
                self_is_donor = bounded.clone()
                if dm_dev is not None:
                    safe_idx = self_idx.clamp(min=0, max=N - 1)
                    self_is_donor = bounded & dm_dev[safe_idx]
                base_valid = base_valid - self_is_donor.to(base_valid.dtype)

            top_sim, top_idx = torch.topk(sim_t, k=k, dim=1)
            top_sim = top_sim.clamp(min=_nc)
            # Compute adaptive-temperature statistics from valid neighbors only;
            # padded -inf slots must not distort the temperature when a target
            # view has fewer than k donors.
            valid_k = base_valid.clamp(min=0, max=k)
            slot_valid = (torch.arange(k, device=query_batch.device)
                          .unsqueeze(0) < valid_k.unsqueeze(1))
            has_donor = base_valid > 0
            safe_count = valid_k.clamp(min=1).unsqueeze(1)
            sim_max = torch.where(
                has_donor.unsqueeze(1), top_sim[:, :1],
                top_sim.new_zeros(Q, 1))
            last_valid = (safe_count - 1).long()
            sim_min = top_sim.gather(1, last_valid)
            sim_min = torch.where(
                has_donor.unsqueeze(1), sim_min,
                top_sim.new_zeros(Q, 1))
            sim_range = (sim_max - sim_min).clamp(min=1e-6)
            valid_mean = (torch.where(
                slot_valid, top_sim, top_sim.new_zeros(())).sum(
                    dim=1, keepdim=True) / safe_count)
            sim_spread = ((sim_max - valid_mean)
                          / sim_range)
            # sim_spread is in [0, 1], so the adaptive temperature must stay
            # between 50% and 100% of the user-supplied base temperature.  A
            # fixed 0.05 floor would make base temperatures below 0.05 behave
            # incorrectly and invalidate their sensitivity analysis.
            adaptive_temp = (temperature * (
                1.0 - 0.5 * sim_spread)).clamp(
                    min=0.5 * temperature,
                    max=temperature)
            weights = F.softmax(top_sim / adaptive_temp, dim=1)

            neighbors = agg_latent[top_idx].float()
            refined_all = torch.sum(
                neighbors * weights.unsqueeze(-1), dim=1)
            refined = torch.where(
                has_donor.unsqueeze(-1),
                refined_all.to(agg_latent.dtype),
                agg_latent.new_zeros(Q, agg_latent.shape[1]))
            results.append(refined)

            if return_stats:
                neighbor_var, eff_k, margin = self._compute_knn_stats(
                    refined, agg_latent, weights, top_sim, top_idx,
                    Q, query_batch.device)
                zeros = neighbor_var.new_zeros(Q)
                neighbor_var = torch.where(has_donor, neighbor_var, zeros)
                margin = torch.where(base_valid >= 2, margin, zeros)
                eff_k = torch.where(
                    base_valid == 0, zeros,
                    torch.where(base_valid == 1, eff_k.new_ones(Q), eff_k))
                all_stats.append((neighbor_var, eff_k, margin))

        if len(results) == 1:
            return (results[0], all_stats[0]) if return_stats else results[0]
        return (results, all_stats) if return_stats else results

    def _precompute_knn_similarities(self, latent_cache, support_norm,
                                     query_positions):
        """Cache cosine rows used by incomplete-query KNN searches.

        At low missing rates most patients are complete and never act as KNN
        queries during ordinary fusion.  Computing [N_observed, N_support] for
        every view therefore wastes both time and memory.  This cache stores
        only the requested observed-row positions and a dense row lookup.
        """
        similarities = {}
        with torch.amp.autocast('cuda', enabled=False):
            for view in range(self.views):
                query_latent = latent_cache[view]
                positions = query_positions.get(view)
                if (query_latent.shape[0] == 0 or positions is None
                        or positions.numel() == 0):
                    continue
                query_norm = safe_l2_normalize(
                    query_latent[positions], dim=1, min_norm=1e-3)
                row_map = torch.full(
                    (query_latent.shape[0],), -1, dtype=torch.long,
                    device=query_latent.device)
                row_map[positions] = torch.arange(
                    positions.shape[0], device=query_latent.device)
                similarities[view] = (
                    torch.mm(query_norm, support_norm[view].float().t()),
                    row_map,
                )
        return similarities

    @staticmethod
    def _slice_knn_similarity(similarity_cache, view, positions):
        """Select cached cosine rows for observed latent-cache positions."""
        cached = similarity_cache.get(view)
        if cached is None:
            return None
        similarity, row_map = cached
        return similarity[row_map[positions]]

    @staticmethod
    def _compute_knn_stats(refined, agg_latent, weights, top_sim, top_idx,
                           Q, device):
        """Compute KNN stats: (neighbor_var, eff_k, margin).

        Returns zero-filled stats when weights/top_sim are None (no donors).
        """
        if weights is None or top_sim is None or top_idx is None:
            zeros = torch.zeros(Q, dtype=torch.float32, device=device)
            return (zeros, zeros.clone(), zeros.clone())
        diff = agg_latent[top_idx].float() - refined.float().unsqueeze(1)
        weights32 = weights.float()
        neighbor_var = (weights32.unsqueeze(-1) * diff ** 2).sum(
            dim=1).mean(dim=1)
        # [FIX] FP16-safe entropy: cast to FP32 to prevent 1e-10 → 0
        _w32 = weights.float()
        eff_k = torch.exp(-(_w32 * (_w32 + 1e-10).log()).sum(dim=1))
        k = top_sim.shape[1]
        # [FIX] margin=0 when k<2 (clamped values make margin unreliable)
        margin = (top_sim[:, 0] - top_sim[:, 1]
                  if k >= 2
                  else torch.zeros(Q, dtype=torch.float32, device=device))
        return (neighbor_var, eff_k, margin)

    def _build_class_prototype_bank(self, support_latent, support_labels,
                                    support_mask=None):
        """Build class-conditional prototype bank from support latents.

        [Design note] Each prototype is the mean of ALL support latents for that
        class, including the query sample itself when it belongs to the same class.
        This self-inclusion bias is O(1/N_support) and negligible for typical
        support set sizes (N_support >> 1). Leave-one-out would require per-sample
        prototype reconstruction, which is prohibitively expensive.

        Args:
            support_latent: dict of [N, hidden] per-view latents.
            support_labels: [N] class labels.
            support_mask: optional [N, V] bool/float tensor. When provided,
                each view's prototype only includes samples where that view
                is present (mask[:, view] == 1). This prevents zero-input
                encoder bias from masked samples polluting prototypes.
                _class_counts becomes a per-view dict.
        """
        if support_latent is None or support_labels is None:
            return None
        if support_labels.numel() == 0:
            return None

        support_labels = support_labels.view(-1).long()
        num_class = self.num_class

        prototype_bank = {}

        if support_mask is not None:
            # [FIX-Issue3] Per-view prototype construction.
            # A single mask cannot correctly build three views' prototypes
            # because different views have different valid samples.
            _mask = support_mask.to(support_labels.device)
            _class_counts = {}
            for view in range(self.views):
                valid = _mask[:, view].bool()
                labels_v = support_labels[valid]
                latent_v = support_latent[view][valid]
                raw_counts_v = torch.bincount(
                    labels_v, minlength=num_class)
                counts_v = raw_counts_v.clamp_min(1)
                sums = latent_v.new_zeros(num_class, latent_v.shape[1])
                sums.index_add_(0, labels_v, latent_v)
                prototype_bank[view] = sums / counts_v.to(
                    latent_v.dtype).unsqueeze(1)
                _class_counts[view] = raw_counts_v.float()
            prototype_bank['_class_counts'] = _class_counts
        else:
            # Backward-compatible: shared counts across views
            raw_counts = torch.bincount(
                support_labels, minlength=num_class)
            counts = raw_counts.clamp_min(1)
            for view in range(self.views):
                latent = support_latent[view]
                sums = latent.new_zeros(num_class, latent.shape[1])
                sums.index_add_(0, support_labels, latent)
                prototype_bank[view] = sums / counts.to(
                    latent.dtype).unsqueeze(1)
            prototype_bank['_class_counts'] = raw_counts.float()

        return prototype_bank

    def _loo_proto_pred(self, class_probs, prototype_bank_view,
                        support_labels, sample_latent, sample_indices,
                        class_counts=None, target_view=None):
        """Leave-one-out prototype prediction — O(M×H) space, no [M,C,H] tensor.

        [FIX-Issue2] Computes the same result as
            loo_proto = _leave_one_out_prototype(...)
            pred = bmm(class_probs.unsqueeze(1), loo_proto).squeeze(1)
        but avoids materializing the [M, num_class, hidden] tensor.

        For singleton classes (count ≤ 1), instead of falling back to the
        self-containing prototype, subtracts the sample's own contribution so
        the Prototype expert carries no self-leakage signal.

        Args:
            class_counts: optional pre-computed counts from
                _build_class_prototype_bank['_class_counts'].
                Can be a [num_class] tensor (complete mode) or a dict
                {view: [num_class] tensor} (masked mode). When dict,
                target_view selects the per-view counts.
            target_view: int, required when class_counts is a dict.

        Returns:
            [M, hidden] leave-one-out prototype prediction.
        """
        if prototype_bank_view is None:
            return None

        # Ensure labels are on the correct device (support_labels may live on
        # a different device than sample_indices under multi-GPU or CPU eval).
        support_labels = support_labels.view(-1).to(
            device=sample_indices.device, dtype=torch.long)

        # Base prediction: standard weighted sum of prototypes [M, H]
        base = torch.mm(class_probs, prototype_bank_view)

        sample_labels = support_labels[sample_indices]
        num_class = prototype_bank_view.shape[0]

        # Reuse cached counts if available, otherwise compute
        if class_counts is None:
            class_counts = torch.bincount(
                support_labels, minlength=num_class).float()
        elif isinstance(class_counts, dict):
            # Per-view counts from masked-mode prototype bank
            if target_view is not None and target_view in class_counts:
                class_counts = class_counts[target_view].to(
                    device=sample_indices.device)
            else:
                class_counts = torch.bincount(
                    support_labels, minlength=num_class).float()
        else:
            class_counts = class_counts.to(device=sample_indices.device)

        proto_c = prototype_bank_view[sample_labels]          # [M, H]
        count_c = class_counts[sample_labels].unsqueeze(-1)   # [M, 1]

        # Leave-one-out adjusted prototype for each sample's own class
        adjusted = (proto_c * count_c - sample_latent) / (count_c - 1).clamp_min(1)

        # Singleton fix: when count ≤ 1, subtract self entirely (delta = -proto_c)
        # so that base + p_own * delta removes the self-contribution cleanly,
        # instead of falling back to the self-containing prototype.
        mask = (count_c <= 1).float()
        delta = (adjusted - proto_c) * (1 - mask) + (-proto_c) * mask

        # Correction: only the sample's own class differs from the base.
        # [FIX-AMP] Compute correction in FP32 for numerical stability, then
        # cast back to base.dtype (FP16/BF16) to match the output buffer.
        p_own = class_probs.gather(1, sample_labels.unsqueeze(1))  # [M, 1]
        correction = (p_own.float() * delta.float()).to(base.dtype)
        return base + correction

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

    def _get_mask_structure(self, mask):
        """Cache indices derived solely from a fixed missingness mask.

        Train, validation and test masks are reused for many epochs. Computing
        nonzero/where indices repeatedly is especially costly on CUDA because
        dynamic output sizes synchronize the device. A tiny four-entry cache is
        sufficient for the three dataset splits while bounding memory use.
        """
        version = tensor_version_or_none(mask)
        cache = getattr(self, '_mask_structure_cache', [])
        if version is not None:
            for position, entry in enumerate(cache):
                if entry['mask'] is mask and entry['version'] == version:
                    # Index tensors created inside inference_mode cannot later
                    # participate in an autograd-tracked index operation. This
                    # occurs when an external caller evaluates and then trains
                    # again with the exact same mask object. Rebuild the derived
                    # indices as ordinary tensors in that case.
                    if (entry.get('created_in_inference_mode', False)
                            and torch.is_grad_enabled()):
                        continue
                    if position:
                        cache.insert(0, cache.pop(position))
                    return entry

        present = mask == 1
        obs_idx = tuple(torch.where(present[:, v])[0]
                        for v in range(self.views))
        pos_map = torch.full(
            (self.views, mask.shape[0]), -1,
            dtype=torch.long, device=mask.device)
        for v, rows in enumerate(obs_idx):
            pos_map[v, rows] = torch.arange(rows.shape[0], device=mask.device)

        missing = ~present
        single_missing = tuple(torch.where(
            missing[:, target] &
            torch.stack([present[:, v] for v in range(self.views)
                         if v != target], dim=1).all(dim=1))[0]
            for target in range(self.views))
        impute_indices = torch.where(missing.any(dim=1))[0]
        bits = present[impute_indices].long()
        pattern_ints = sum(bits[:, v] * (1 << v)
                           for v in range(self.views))
        pattern_positions = {
            pattern: torch.where(pattern_ints == pattern)[0]
            for pattern in range(1, 1 << self.views)
        }
        entry = {
            'mask': mask, 'version': version, 'present': present,
            'obs_idx': obs_idx, 'pos_map': pos_map,
            'single_missing': single_missing,
            'impute_indices': impute_indices,
            'pattern_ints': pattern_ints,
            'pattern_positions': pattern_positions,
            'created_in_inference_mode':
                torch.is_inference_mode_enabled(),
        }
        if version is not None:
            cache = [old for old in cache if old['mask'] is not mask]
            cache.insert(0, entry)
            self._mask_structure_cache = cache[:4]
        return entry

    def infer_on_missing(self, data_list, mask,
                         support_data_list=None,
                         support_labels=None,
                         support_mask=None,
                         use_cross_sample_impute=True,
                         use_prototype_bank=True,
                         cross_sample_k=5,
                         knn_base_temperature=0.2,
                         exclude_self=False,
                         return_latents=False,
                         compute_router_loss=False,
                         router_temperature=0.5,
                         use_action_router=True,
                         build_all_candidates=False,
                         cache_teacher_latent=False):
        effective_action_router = use_action_router
        # make sure eval all modules before
        x1_train, x2_train, x3_train = data_list

        # [FIX-Issue5/6] exclude_self=True requires query/support index alignment.
        # When True, _group_idx from impute_indices is used to index BOTH
        # data_list (query) and support_latent (derived from support_data_list).
        # If the two sets have different sizes or ordering, LOO and KNN
        # exclude_self will operate on WRONG indices.
        #
        # Length checks catch size mismatches but NOT ordering differences.
        # The caller MUST guarantee query and support are the same set in the
        # same order. The Trainer achieves this by passing the same data slice
        # as both query and support.
        #
        # [Debug check] When label is provided, verify label consistency as a
        # proxy for ordering. This catches the most common misalignment case.
        if exclude_self:
            _q_n = x1_train.shape[0]
            if support_data_list is not None:
                for _vi, _sv in enumerate(support_data_list):
                    _s_n = _sv.shape[0]
                    if _q_n != _s_n:
                        raise ValueError(
                            f"exclude_self=True requires query and support to "
                            f"have the same sample count and index order. "
                            f"Query has {_q_n} samples but support view {_vi} "
                            f"has {_s_n}. Set exclude_self=False if they are "
                            f"different sets.")
            if support_labels is not None:
                _sl_n = support_labels.shape[0]
                if _q_n != _sl_n:
                    raise ValueError(
                        f"exclude_self=True requires support_labels length "
                        f"({_sl_n}) to match query sample count ({_q_n}).")

        # [FIX-Issue6] Additional input validations
        if support_mask is not None:
            if support_mask.ndim != 2:
                raise ValueError(
                    f"support_mask must be 2D [N, views], got "
                    f"{support_mask.ndim}D with shape {tuple(support_mask.shape)}.")
            _n_sup = (support_data_list[0].shape[0]
                      if support_data_list is not None else 0)
            if support_mask.shape[0] != _n_sup:
                raise ValueError(
                    f"support_mask first dim ({support_mask.shape[0]}) must "
                    f"match support_data length ({_n_sup}).")
            if support_mask.shape[1] != self.views:
                raise ValueError(
                    f"support_mask second dim ({support_mask.shape[1]}) must "
                    f"match number of views ({self.views}).")
            # Keep a strong reference and the tensor version. This detects any
            # in-place edit without a per-forward sum().item() GPU sync, and the
            # strong reference prevents allocator pointer reuse from bypassing
            # validation for a different tensor.
            _sm_version = tensor_version_or_none(support_mask)
            _sm_same = (_sm_version is not None and
                getattr(self, '_sm_validated_tensor', None) is support_mask and
                getattr(self, '_sm_validated_version', None) == _sm_version)
            if not _sm_same:
                # Validate values: must be 0 or 1 only
                _sm_min = support_mask.min().item()
                _sm_max = support_mask.max().item()
                if _sm_min < 0 or _sm_max > 1 or (
                        _sm_min != 0 and _sm_min != 1) or (
                        _sm_max != 0 and _sm_max != 1):
                    raise ValueError(
                        f"support_mask must contain only 0/1 values, "
                        f"got range [{_sm_min}, {_sm_max}].")
                if not torch.isin(support_mask,
                        support_mask.new_tensor([0, 1])).all():
                    raise ValueError(
                        "support_mask must contain only 0 and 1 values.")
                # Reject all-zero rows
                _row_sums = support_mask.sum(dim=1)
                if (_row_sums == 0).any():
                    raise ValueError(
                        "support_mask contains all-zero rows (no valid views). "
                        "These samples cannot serve as support donors.")
                self._sm_validated_tensor = support_mask
                self._sm_validated_version = _sm_version

        # Validate the query mask once per tensor version. Unlike a sum-based
        # fingerprint, this catches same-sum edits and adds no recurring sync.
        if mask.ndim != 2:
            raise ValueError(
                f"mask must be 2D [N, views], got {mask.ndim}D "
                f"with shape {tuple(mask.shape)}.")
        if mask.shape[0] != x1_train.shape[0]:
            raise ValueError(
                f"mask first dim ({mask.shape[0]}) must match query sample "
                f"count ({x1_train.shape[0]}).")
        _m_version = tensor_version_or_none(mask)
        _m_same = (_m_version is not None and
            getattr(self, '_mask_validated_tensor', None) is mask and
            getattr(self, '_mask_validated_version', None) == _m_version)
        if not _m_same:
            if mask.shape[1] != self.views:
                raise ValueError(
                    f"mask second dim ({mask.shape[1]}) must match "
                    f"number of views ({self.views}).")
            # Value checks
            _mask_min = mask.min().item()
            _mask_max = mask.max().item()
            if _mask_min < 0 or _mask_max > 1:
                raise ValueError(
                    f"mask values out of [0,1] range, "
                    f"got [{_mask_min}, {_mask_max}].")
            if not torch.isin(mask, mask.new_tensor([0, 1])).all():
                raise ValueError(
                    "mask must contain only 0 and 1 values, "
                    "found non-binary entries.")
            # [FIX] Reject all-zero rows: a sample with all views missing
            # cannot be classified (no target view for any action).
            _mask_row_sums = mask.sum(dim=1)
            if (_mask_row_sums == 0).any():
                raise ValueError(
                    "mask contains all-zero rows (no valid views for some "
                    "samples). Every sample must have at least one view present.")
            self._mask_validated_tensor = mask
            self._mask_validated_version = _m_version
        # the previous epoch/batch.
        self._last_candidate_cache_t = None
        self._last_candidate_cache_valid = None
        self._last_action_logits = None
        self._last_router_loss = None
        self._last_impute_indices = None
        self._last_support_latent = None
        mask_structure = self._get_mask_structure(mask)
        _present_mask = mask_structure['present']
        a_idx_eval, b_idx_eval, c_idx_eval = _present_mask.unbind(dim=1)

        # predict on each omics without missing
        # [FIX-Bug2 + P5] encode 返回 (weighted, raw, conf)
        a_latent_eval, _, a_conf_eval = self.encode(x1_train[a_idx_eval], self.att[0], self.emb[0], self.aux_conf[0])
        b_latent_eval, _, b_conf_eval = self.encode(x2_train[b_idx_eval], self.att[1], self.emb[1], self.aux_conf[1])
        c_latent_eval, _, c_conf_eval = self.encode(x3_train[c_idx_eval], self.att[2], self.emb[2], self.aux_conf[2])

        # [P6] Use new_zeros to create latent buffers matching AMP dtype automatically
        latent_code_a_eval = a_latent_eval.new_zeros(x1_train.shape[0], self.hidden_dim[0])
        latent_code_b_eval = b_latent_eval.new_zeros(x2_train.shape[0], self.hidden_dim[0])
        latent_code_c_eval = c_latent_eval.new_zeros(x3_train.shape[0], self.hidden_dim[0])
        # Dense cache for the initial two-source CV prediction of each possible
        # single-missing target. Router fusion reuses it instead of invoking the
        # same two predictors a second time.
        _initial_cv_cache = [
            latent_code_a_eval.new_zeros(latent_code_a_eval.shape)
            for _ in range(self.views)
        ]

        # ENCODE CACHE: each view encoded once, all subsequent access via indexing.
        # _lat_cache[v] = weighted embeddings for all samples where view v is present
        # _obs_idx[v] = global indices of samples where view v is present
        _lat_cache = {0: a_latent_eval, 1: b_latent_eval, 2: c_latent_eval}
        _obs_idx = {v: mask_structure['obs_idx'][v]
                    for v in range(self.views)}
        # Dense device-side map: _pos_map[v, global_idx] -> row in _lat_cache[v].
        # This removes per-patient .item()/.tolist() synchronization and Python dict lookup.
        _pos_map = mask_structure['pos_map']

        # [P5] source_conf from encode() confidence output (avoids redundant aux_conf calls)
        _src_conf_all = {
            0: a_conf_eval.squeeze(-1),  # [n_present_0]
            1: b_conf_eval.squeeze(-1),
            2: c_conf_eval.squeeze(-1),
        }

        # Write observed latents to latent_code_*_eval immediately (out-of-place via index_put).
        # Ensures downstream proto_fusion / proto_probs use actual encoded values
        # for observed views instead of zeros (fixes the ordering bug).
        # Using index_put (non-inplace) to avoid autograd version conflicts.
        latent_code_a_eval = latent_code_a_eval.index_put((_obs_idx[0],), a_latent_eval)
        latent_code_b_eval = latent_code_b_eval.index_put((_obs_idx[1],), b_latent_eval)
        latent_code_c_eval = latent_code_c_eval.index_put((_obs_idx[2],), c_latent_eval)
        # Preserve the observed-only dense buffers before CV predictions replace
        # missing slots. Masked support can reuse these directly without three
        # additional large scatter operations.
        _observed_latent_cache = (
            latent_code_a_eval, latent_code_b_eval, latent_code_c_eval)

        # Single-missing A: B and C are both observed.
        ano_bcbothhas_idx = mask_structure['single_missing'][0]
        if ano_bcbothhas_idx.numel() != 0:
            _b_pos = _pos_map[1, ano_bcbothhas_idx]
            _c_pos = _pos_map[2, ano_bcbothhas_idx]
            ano_bcbothhas_1 = _lat_cache[1][_b_pos]
            ano_bcbothhas_2 = _lat_cache[2][_c_pos]
            ano_bcbothhas_all = (
                self.b2a(ano_bcbothhas_1)[0]
                + self.c2a(ano_bcbothhas_2)[0]) / 2.0
            _initial_cv_cache[0] = _initial_cv_cache[0].index_put(
                (ano_bcbothhas_idx,), ano_bcbothhas_all)
            # [FIX-Issue4] Always write initial CV; Selector decides later
            latent_code_a_eval = latent_code_a_eval.index_put(
                (ano_bcbothhas_idx,), ano_bcbothhas_all)

        # Single-missing B: A and C are both observed.
        bno_acbothhas_idx = mask_structure['single_missing'][1]
        if bno_acbothhas_idx.numel() != 0:
            _a_pos = _pos_map[0, bno_acbothhas_idx]
            _c_pos = _pos_map[2, bno_acbothhas_idx]
            bno_acbothhas_1 = _lat_cache[0][_a_pos]
            bno_acbothhas_2 = _lat_cache[2][_c_pos]
            bno_acbothhas_all = (
                self.a2b(bno_acbothhas_1)[0]
                + self.c2b(bno_acbothhas_2)[0]) / 2.0
            _initial_cv_cache[1] = _initial_cv_cache[1].index_put(
                (bno_acbothhas_idx,), bno_acbothhas_all)
            latent_code_b_eval = latent_code_b_eval.index_put(
                (bno_acbothhas_idx,), bno_acbothhas_all)

        # Single-missing C: A and B are both observed.
        cno_abbothhas_idx = mask_structure['single_missing'][2]
        if cno_abbothhas_idx.numel() != 0:
            _a_pos = _pos_map[0, cno_abbothhas_idx]
            _b_pos = _pos_map[1, cno_abbothhas_idx]
            cno_abbothhas_1 = _lat_cache[0][_a_pos]
            cno_abbothhas_2 = _lat_cache[1][_b_pos]
            cno_abbothhas_all = (
                self.a2c(cno_abbothhas_1)[0]
                + self.b2c(cno_abbothhas_2)[0]) / 2.0
            _initial_cv_cache[2] = _initial_cv_cache[2].index_put(
                (cno_abbothhas_idx,), cno_abbothhas_all)
            latent_code_c_eval = latent_code_c_eval.index_put(
                (cno_abbothhas_idx,), cno_abbothhas_all)

        # Action logits are produced later from the same reliability features
        # as the expert logits.  Sentinel action 3 means "impute all" and is
        # used only when the selective action head is disabled.
        impute_indices = mask_structure['impute_indices']
        _hd = self.hidden_dim[0]
        _selector_actions = torch.full(
            (impute_indices.shape[0],), 3, dtype=torch.long,
            device=mask.device)

        # [FIX-Bug3] Always compute candidates — avoids GPU→CPU sync from
        # bool((_selector_actions != 0).any()).  During training build_all_candidates
        # is already True; during inference the rare edge case where every sample
        # selects action 0 saves negligible compute vs. the sync cost.
        _need_selected_candidates = True
        need_support_latent = (
            (use_cross_sample_impute or use_prototype_bank
             or effective_action_router or compute_router_loss)
            and _need_selected_candidates)
        support_latent = None
        if need_support_latent:
            if support_data_list is None:
                import warnings
                warnings.warn(
                    "support_data_list is None, falling back to (possibly masked) data_list. "
                    "KNN donors' missing-target views will be zero; Prototype bank will "
                    "include encoder bias from zero-input views; Router teacher may see "
                    "erroneous support latent. Pass original (unmasked) data as support_data_list.",
                    stacklevel=2,
                )
                support_data_list = data_list

            # Default masked-support training uses the exact same tensors for
            # query and support. Reuse the already encoded observed rows instead
            # of running all three encoders a second time. Missing rows are never
            # valid donors because support_mask filters them per view.
            reuse_masked_support = (
                support_mask is mask and
                len(support_data_list) == self.views and
                all(support_data_list[v] is data_list[v]
                    for v in range(self.views)))
            support_latent = {}
            support_confidence = {}
            if reuse_masked_support:
                for view in range(self.views):
                    cached_conf = _src_conf_all[view].detach()
                    support_latent[view] = _observed_latent_cache[view].detach()
                    support_confidence[view] = cached_conf.new_zeros(
                        mask.shape[0]).index_put(
                            (_obs_idx[view],), cached_conf)
            else:
                with torch.no_grad():
                    for view in range(self.views):
                        _sl, _, _sc = self.encode(
                            support_data_list[view], self.att[view],
                            self.emb[view], self.aux_conf[view])
                        support_latent[view] = _sl
                        support_confidence[view] = _sc.squeeze(-1)

            # Pre-compute normalized support vectors once per view for all
            # pattern-batched KNN searches.
            support_norm = ({
                v: safe_l2_normalize(
                    support_latent[v], dim=1, min_norm=1e-3)
                for v in range(self.views)
            } if use_cross_sample_impute else None)

        # ==================== Router-based four-expert fusion ====================
        # Replaces fixed-weight three-way fusion with learned router.

        # Prototype probabilities are only consumed by Router fusion.  Avoid a
        # [FIX-Issue3] Compute proto_probs with missing positions zeroed via mask,
        # so the classifier never sees initial CV predictions at missing views.
        # This is consistent whether the selective action head is enabled or not.
        proto_probs = None
        prototype_bank = None
        if support_latent is not None:
            proto_inputs = [
                latent_code_a_eval * mask[:, 0:1].to(latent_code_a_eval.dtype),
                latent_code_b_eval * mask[:, 1:2].to(latent_code_b_eval.dtype),
                latent_code_c_eval * mask[:, 2:3].to(latent_code_c_eval.dtype),
            ]
            proto_probs = F.softmax(
                self.MMClasifier(torch.cat(proto_inputs, dim=1)), dim=1)
            _sm = (support_mask.to(mask.device)
                   if support_mask is not None else None)
            if use_prototype_bank and support_labels is not None:
                prototype_bank = self._build_class_prototype_bank(
                    support_latent, support_labels.to(mask.device),
                    support_mask=_sm)

            # [FIX-Issue1] Pre-compute KNN donor masks from support_mask.
            # donor_masks[(src, tgt)] = support samples with both views valid.
            donor_masks = {}
            if use_cross_sample_impute and _sm is not None:
                for _src in range(self.views):
                    for _tgt in range(self.views):
                        if _src != _tgt:
                            donor_masks[(_src, _tgt)] = (
                                _sm[:, _src].bool() & _sm[:, _tgt].bool())

        if impute_indices.numel() > 0 and support_latent is not None:
            # [P1+P4] PATTERN-BASED BATCHED KNN: group by missing pattern,
            # batch KNN within each group. Reduces KNN searches from N to ~6.
            _n_support_val = math.log1p(
                float(support_latent[0].shape[0]))

            _pattern_ints = mask_structure['pattern_ints']

            # Pre-allocate feature tensor (fills per-pattern, then reorders)
            _n_impute = impute_indices.shape[0]
            _feat_dim = 12
            _router_feats = torch.zeros(_n_impute, _feat_dim, device=mask.device)

            # Cache pattern metadata for fusion loop (avoids re-computing .tolist())
            _pattern_meta = {}  # pattern_int → (missing_views, present_views, source_v, target_v_first)
            _router_knn_cache = {}  # (pattern, target) -> prediction aligned to group order
            # Router-supervision epochs query the same incomplete rows multiple
            # times. Cache only those rows; complete samples used for synthetic
            # teachers are intentionally excluded to avoid low-missing-rate
            # O(N^2) over-allocation.
            _knn_sim = {}
            if (compute_router_loss and use_cross_sample_impute
                    and support_norm is not None):
                _knn_query_positions = {}
                for _view in range(self.views):
                    _query_global = impute_indices[
                        _present_mask[impute_indices, _view]]
                    _knn_query_positions[_view] = _pos_map[
                        _view, _query_global]
                _knn_sim = self._precompute_knn_similarities(
                    _lat_cache, support_norm, _knn_query_positions)

            for _pat_val in range(1, 1 << self.views):
                _group_pos = mask_structure['pattern_positions'][_pat_val]
                if _group_pos.numel() == 0:
                    continue
                _group_idx = impute_indices[_group_pos]  # global sample indices

                # Decode pattern → missing/present views
                _miss = [v for v in range(3) if not (_pat_val & (1 << v))]
                _pres = [v for v in range(3) if (_pat_val & (1 << v))]
                _pattern_meta[_pat_val] = (_miss, _pres)

                G = _group_pos.shape[0]

                # Source conf: vectorized from cache
                source_conf_g = torch.zeros(G, 3, device=mask.device)
                for v in _pres:
                    _positions = _pos_map[v, _group_idx]
                    source_conf_g[:, v] = _src_conf_all[v][_positions]

                pred_conf_g = (source_conf_g[:, _pres].mean(dim=1)
                               if _pres else torch.zeros(G, device=mask.device))

                # Batched KNN for router stats (on first missing view only)
                knn_density_g = torch.zeros(G, device=mask.device)
                knn_margin_g = torch.zeros(G, device=mask.device)
                knn_eff_k_g = torch.zeros(G, device=mask.device)
                if use_cross_sample_impute and _pres and _miss:
                    _src_v = _pres[0]
                    if (build_all_candidates or compute_router_loss
                            or not effective_action_router):
                        _knn_local = torch.arange(G, device=mask.device)
                    else:
                        _knn_local = torch.where(
                            _selector_actions[_group_pos] != 0)[0]
                    _knn_group_idx = _group_idx[_knn_local]
                    _positions = _pos_map[_src_v, _knn_group_idx]
                    query_batch = _lat_cache[_src_v][_positions]
                    if query_batch.shape[0] > 0:
                        # [FIX-Issue1] Per-target donor masks for router cache KNN
                        _dm_list = None
                        if donor_masks:
                            _dm_list = [
                                donor_masks.get((_src_v, _tv))
                                for _tv in _miss
                            ]
                        _knn_outputs, knn_stats = self._batched_knn_search(
                            query_batch, support_norm[_src_v], support_latent[_src_v],
                            top_k=cross_sample_k, temperature=knn_base_temperature,
                            aggregate_targets=[support_latent[v] for v in _miss],
                            exclude_self_indices=(
                                _knn_group_idx if exclude_self else None),
                            donor_mask=_dm_list,
                            return_stats=True,
                            precomputed_sim=self._slice_knn_similarity(
                                _knn_sim, _src_v, _positions),
                        )
                        if len(_miss) == 1:
                            _knn_outputs = [_knn_outputs]
                        for target_v, prediction in zip(_miss, _knn_outputs):
                            dense_prediction = support_latent[target_v].new_zeros(
                                G, support_latent[target_v].shape[1])
                            _router_knn_cache[(_pat_val, target_v)] = (
                                dense_prediction.index_put((_knn_local,), prediction))
                        if knn_stats is not None:
                            # knn_stats may be a list of tuples (multi-target)
                            # or a single tuple (single-target)
                            if isinstance(knn_stats, list):
                                _avg_nv = torch.stack(
                                    [s[0] for s in knn_stats if s[0] is not None]
                                ).mean(dim=0) if any(s[0] is not None for s in knn_stats) else None
                                _avg_ek = torch.stack(
                                    [s[1] for s in knn_stats if s[1] is not None]
                                ).mean(dim=0) if any(s[1] is not None for s in knn_stats) else None
                                _avg_mg = torch.stack(
                                    [s[2] for s in knn_stats if s[2] is not None]
                                ).mean(dim=0) if any(s[2] is not None for s in knn_stats) else None
                            else:
                                _avg_nv, _avg_ek, _avg_mg = knn_stats
                            if _avg_nv is not None:
                                knn_density_g[_knn_local] = _avg_nv.to(knn_density_g.dtype)
                                knn_eff_k_g[_knn_local] = _avg_ek.to(knn_eff_k_g.dtype)
                                knn_margin_g[_knn_local] = _avg_mg.to(knn_margin_g.dtype)

                # Proto entropy: vectorized
                proto_probs_g = proto_probs[_group_idx]
                proto_entropy_g = -(proto_probs_g.float() * (proto_probs_g.float() + 1e-10).log()).sum(dim=1)

                # Missing pattern vector (same for all in group)
                missing_pattern_g = torch.zeros(3, device=mask.device)
                for v in _miss:
                    missing_pattern_g[v] = 1.0

                n_support_g = torch.full((G,), _n_support_val, device=mask.device)

                # Build features: [G, 12]
                feats_g = torch.cat([
                    missing_pattern_g.unsqueeze(0).expand(G, 3),
                    source_conf_g,
                    torch.stack([pred_conf_g, knn_density_g, knn_margin_g], dim=1),
                    torch.stack([knn_eff_k_g, proto_entropy_g, n_support_g], dim=1),
                ], dim=1)
                _router_feats[_group_pos] = feats_g

            router_logits, action_logits = self.router(
                _router_feats, return_action=True)
            router_weights = F.softmax(router_logits, dim=1)  # softmax at fusion point
            if effective_action_router:
                # Action 2 has no meaning for a one-missing patient.  Mask it
                # for the discrete inference decision, while retaining the
                # original logits for the differentiable KL supervision.
                decision_logits = action_logits.detach().clone()
                one_missing = (
                    (mask[impute_indices] == 0).sum(dim=1) == 1)
                decision_logits[one_missing, 2] = self._NEG_INF_CLAMP
                _selector_actions = decision_logits.argmax(dim=1)
            else:
                _selector_actions = torch.full(
                    (_n_impute,), 3, dtype=torch.long, device=mask.device)
            if build_all_candidates:
                self._last_action_logits = action_logits
            _hd = self.hidden_dim[0]

            # Tensor-only candidate cache and output buffers.
            _candidate_cache_t = latent_code_a_eval.new_zeros(_n_impute, 3, _hd)
            _candidate_cache_valid = torch.zeros(
                _n_impute, 3, dtype=torch.bool, device=mask.device)
            latent_list = [latent_code_a_eval, latent_code_b_eval, latent_code_c_eval]

            # An action is defined relative to the zero-imputation
            # baseline. Explicitly clear missing slots first; this fixes the old
            # one-missing action-0 path that accidentally retained initial CV.
            if effective_action_router:
                for view in range(self.views):
                    missing_global = impute_indices[mask[impute_indices, view] == 0]
                    latent_list[view] = latent_list[view].index_put(
                        (missing_global,),
                        latent_list[view].new_zeros(missing_global.shape[0], _hd))

            # Process one pattern at a time.  CV, KNN, prototype and residual
            # fusion are all batched; no per-patient Python loop remains.
            for _pat_val, (_miss, _pres) in _pattern_meta.items():
                _all_group_pos = mask_structure['pattern_positions'][_pat_val]
                if build_all_candidates:
                    _compute_local = torch.arange(
                        _all_group_pos.shape[0], device=mask.device)
                else:
                    _compute_local = torch.where(
                        _selector_actions[_all_group_pos] != 0)[0]
                _compute_pos = _all_group_pos[_compute_local]
                if _compute_pos.numel() == 0:
                    continue

                _group_idx = impute_indices[_compute_pos]
                _actions_g = _selector_actions[_compute_pos]
                # Search source neighbors once and aggregate every missing target
                # with the shared top-k topology.
                _knn_by_target = {}
                if use_cross_sample_impute and _pres and _miss:
                    if all((_pat_val, v) in _router_knn_cache for v in _miss):
                        _knn_by_target = {
                            v: _router_knn_cache[(_pat_val, v)][_compute_local]
                            for v in _miss
                        }
                    else:
                        source_v = _pres[0]
                        query_batch = _lat_cache[source_v][
                            _pos_map[source_v, _group_idx]]
                        # [FIX-Issue1] Per-target donor masks for fusion KNN
                        # Each target uses its own mask independently.
                        _dm_list = None
                        if donor_masks:
                            _dm_list = [
                                donor_masks.get((source_v, _tv))
                                for _tv in _miss
                            ]
                        _knn_group = self._batched_knn_search(
                            query_batch, support_norm[source_v], support_latent[source_v],
                            top_k=cross_sample_k, temperature=knn_base_temperature,
                            aggregate_targets=[support_latent[v] for v in _miss],
                            exclude_self_indices=_group_idx if exclude_self else None,
                            donor_mask=_dm_list,
                            precomputed_sim=self._slice_knn_similarity(
                                _knn_sim, source_v,
                                _pos_map[source_v, _group_idx]))
                        if len(_miss) == 1:
                            _knn_group = [_knn_group]
                        _knn_by_target = dict(zip(_miss, _knn_group))

                weights_g = router_weights[_compute_pos]
                for missing_order, target_v in enumerate(_miss):
                    remaining = [v for v in _pres if v != target_v]
                    if len(_miss) == 1:
                        cv_pred = _initial_cv_cache[target_v][_group_idx]
                    elif remaining:
                        cv_parts = []
                        for src_v in remaining:
                            source_latent = _lat_cache[src_v][
                                _pos_map[src_v, _group_idx]]
                            cv_parts.append(
                                self._get_predictor(src_v, target_v)(source_latent)[0])
                        cv_pred = (torch.stack(cv_parts).mean(dim=0)
                                   if len(cv_parts) > 1 else cv_parts[0])
                    else:
                        cv_pred = latent_list[target_v][_group_idx]

                    knn_pred = _knn_by_target.get(target_v, cv_pred)
                    if prototype_bank is not None:
                        if exclude_self:
                            # Training: sample IS in support set → LOO to prevent leakage.
                            # [FIX-Issue4] In masked mode, only apply LOO to samples
                            # whose target view is valid in support_mask. Masked samples
                            # did NOT contribute to the per-view prototype, so LOO
                            # subtraction would be incorrect.
                            if support_mask is not None:
                                _sm_dev = support_mask.to(_group_idx.device)
                                _view_valid = _sm_dev[_group_idx, target_v].bool()
                                _proto_pred = torch.mm(
                                    proto_probs[_group_idx],
                                    prototype_bank[target_v])
                                _loo_idx = _group_idx[_view_valid]
                                _loo_pos = torch.where(_view_valid)[0]
                                _sample_lat = support_latent[target_v][
                                    _loo_idx].detach()
                                # Empty LOO batches are valid, so no .any()
                                # synchronization is needed here.
                                _loo_pred = self._loo_proto_pred(
                                    proto_probs[_loo_idx],
                                    prototype_bank[target_v],
                                    support_labels, _sample_lat, _loo_idx,
                                    class_counts=prototype_bank.get(
                                        '_class_counts'),
                                    target_view=target_v)
                                _proto_pred = _proto_pred.index_put(
                                    (_loo_pos,), _loo_pred)
                                proto_pred = _proto_pred
                            else:
                                sample_lat = support_latent[target_v][
                                    _group_idx].detach()
                                proto_pred = self._loo_proto_pred(
                                    proto_probs[_group_idx],
                                    prototype_bank[target_v],
                                    support_labels, sample_lat, _group_idx,
                                    class_counts=prototype_bank.get(
                                        '_class_counts'),
                                    target_view=target_v)
                        else:
                            # Testing: sample NOT in support set, different index
                            # space → plain mm, no LOO (would cause IndexError).
                            proto_pred = torch.mm(
                                proto_probs[_group_idx],
                                prototype_bank[target_v])
                    else:
                        proto_pred = cv_pred

                    fused = (cv_pred
                             + weights_g[:, 1:2] * (knn_pred - cv_pred)
                             + weights_g[:, 2:3] * (proto_pred - cv_pred)
                             + weights_g[:, 3:4] * (-cv_pred))

                    target_column = torch.full_like(_compute_pos, target_v)
                    _candidate_cache_t = _candidate_cache_t.index_put(
                        (_compute_pos, target_column), fused)
                    _candidate_cache_valid[_compute_pos, target_v] = True

                    # [FIX-Issue1] 3-action scheme: 0=skip, 1=impute first missing, 2=impute second missing
                    if not effective_action_router or len(_miss) == 1:
                        write_mask = (_actions_g != 0)
                    elif missing_order == 0:
                        write_mask = (_actions_g == 1)
                    else:
                        write_mask = (_actions_g == 2)
                    latent_list[target_v] = latent_list[target_v].index_put(
                        (_group_idx[write_mask],), fused[write_mask])

            latent_code_a_eval, latent_code_b_eval, latent_code_c_eval = latent_list

            # Router training loss (factual supervision) — controlled by compute_router_loss flag
            # so it works even when model is in eval() mode during training
            if compute_router_loss:
                router_loss = self._compute_router_loss(
                    [x1_train, x2_train, x3_train], mask, support_latent,
                    support_labels, impute_indices,
                    cross_sample_k, knn_base_temperature, exclude_self,
                    prototype_bank, tau=router_temperature,
                    # [FIX-Bug1] 传入原始未掩码数据，确保 router 特征编码与推理一致
                    original_data_list=support_data_list,
                    support_norm=support_norm,
                    support_confidence=support_confidence,
                    present_latent_cache=_lat_cache,
                    position_map=_pos_map,
                    confidence_cache=_src_conf_all,
                    use_cross_sample_impute=use_cross_sample_impute,
                    pattern_ints=_pattern_ints,
                    pattern_meta=_pattern_meta,
                    donor_masks=donor_masks,
                    support_mask=support_mask,
                    knn_sim=_knn_sim,
                )
                # Store router loss as attribute for train_missing_cg to pick up
                self._last_router_loss = router_loss

            # Cache only the state consumed after this call. Validation/test
            # inference does not need to retain full support/candidate tensors.
            if cache_teacher_latent or build_all_candidates:
                self._last_support_latent = {
                    v: support_latent[v].detach() for v in support_latent}
            if build_all_candidates:
                self._last_impute_indices = impute_indices.detach()
                self._last_candidate_cache_t = _candidate_cache_t.detach()
                self._last_candidate_cache_valid = _candidate_cache_valid.detach()

        # [FIX-Issue5] Fallback path when KNN/prototype support is disabled:
        # perform CV-only imputation (no router, no KNN, no prototype).
        # CV imputation is independent of KNN/Prototype switches.
        if support_latent is None and impute_indices.numel() > 0:
            latent_list = [latent_code_a_eval, latent_code_b_eval, latent_code_c_eval]
            _hd = self.hidden_dim[0]
            _n_impute = impute_indices.shape[0]

            # Build a CV-only candidate cache for the non-router fallback.
            _candidate_cache_t = latent_code_a_eval.new_zeros(_n_impute, 3, _hd)
            _candidate_cache_valid = torch.zeros(
                _n_impute, 3, dtype=torch.bool, device=mask.device)

            for pattern in range(1, 1 << self.views):
                group_pos = mask_structure['pattern_positions'][pattern]
                if group_pos.numel() == 0:
                    continue
                group_idx = impute_indices[group_pos]
                actions = _selector_actions[group_pos]
                missing_views = [v for v in range(self.views)
                                 if not (pattern & (1 << v))]
                present_views = [v for v in range(self.views)
                                 if pattern & (1 << v)]

                for missing_order, target_v in enumerate(missing_views):
                    remaining = [v for v in present_views if v != target_v]
                    # Compute CV prediction from remaining present views
                    if len(missing_views) == 1:
                        cv_pred = _initial_cv_cache[target_v][group_idx]
                    elif remaining:
                        cv_parts = []
                        for src_v in remaining:
                            source_latent = _lat_cache[src_v][
                                _pos_map[src_v, group_idx]]
                            cv_parts.append(
                                self._get_predictor(src_v, target_v)(source_latent)[0])
                        cv_pred = (torch.stack(cv_parts).mean(dim=0)
                                   if len(cv_parts) > 1 else cv_parts[0])
                    else:
                        cv_pred = latent_list[target_v][group_idx]

                    # Store CV prediction in candidate cache for selector GT
                    target_column = torch.full_like(group_pos, target_v)
                    _candidate_cache_t = _candidate_cache_t.index_put(
                        (group_pos, target_column), cv_pred.detach())
                    _candidate_cache_valid[group_pos, target_v] = True

                    # [FIX-Issue1] 3-action scheme: 0=skip, 1=impute first, 2=impute second
                    if not effective_action_router or len(missing_views) == 1:
                        keep = actions != 0
                    elif missing_order == 0:
                        keep = (actions == 1)
                    else:
                        keep = (actions == 2)

                    write_idx = group_idx[keep]
                    if write_idx.numel() > 0:
                        latent_list[target_v] = latent_list[target_v].index_put(
                            (write_idx,), cv_pred[keep])
                    # Clear missing slots for samples that selected action 0
                    clear_idx = group_idx[~keep]
                    if clear_idx.numel() > 0:
                        latent_list[target_v] = latent_list[target_v].index_put(
                            (clear_idx,), latent_list[target_v].new_zeros(
                                clear_idx.shape[0], _hd))
            latent_code_a_eval, latent_code_b_eval, latent_code_c_eval = latent_list

            # [FIX-Issue3] Use build_all_candidates instead of self.training.
            # train_missing_cg calls self.eval() before infer_on_missing(),
            # so self.training is always False here. build_all_candidates is
            # True when selector GT computation is needed.
            if (cache_teacher_latent
                    or (effective_action_router and build_all_candidates)):
                if support_data_list is None:
                    support_data_list = data_list
                reuse_cv_support = (
                    support_mask is mask and
                    all(support_data_list[v] is data_list[v]
                        for v in range(self.views)))
                _cv_support_latent = {}
                if reuse_cv_support:
                    for v in range(self.views):
                        _cv_support_latent[v] = (
                            _observed_latent_cache[v].detach())
                else:
                    with torch.no_grad():
                        for v in range(self.views):
                            _cv_support_latent[v] = self.encode(
                                support_data_list[v], self.att[v],
                                self.emb[v], self.aux_conf[v])[0]
                self._last_support_latent = {v: _cv_support_latent[v].detach()
                                             for v in range(self.views)}
                if build_all_candidates:
                    self._last_impute_indices = impute_indices.detach()
                    self._last_candidate_cache_t = _candidate_cache_t.detach()
                    self._last_candidate_cache_valid = _candidate_cache_valid.detach()

        # Observed latents already written at function entry (after encoding).
        # No need to re-write here.

        latent_fusion_train = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval], dim=1)
        MMlogit = self.MMClasifier(latent_fusion_train)
        if return_latents:
            return MMlogit, {
                0: latent_code_a_eval,
                1: latent_code_b_eval,
                2: latent_code_c_eval,
            }
        return MMlogit

'''
CLCL_trainer
'''

"""Reusable neural-network blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import xavier_init


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self, prediction_dim, activation='relu', batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.LayerNorm(self._prediction_dim[i + 1]))

            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.LayerNorm(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        # 不加 Softmax：cross-view MSE loss 需要无界输出以匹配 feat_emb 的值域
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

class Router(nn.Module):
    """Unified sample-level router for expert fusion and selective actions.

    A shared trunk consumes the 12 reliability features.  Two lightweight heads
    then predict:

      1. expert logits over CV, KNN, class prototype and soft skip;
      2. action logits over skip, impute-first-missing and
         impute-second-missing.

    The action head replaces the former independent ActionSelector.  It keeps
    the downstream-utility idea while avoiding a second feature extractor and
    a separate selector loss weight.

    Input features (12-dim):
      [0:3]   missingness pattern (one-hot for 3 views)
      [3:6]   source view confidence (aux_conf for present views)
      [6:9]   mean source confidence, KNN local dispersion, KNN margin
      [9:12]  KNN effective_k, prototype entropy H(p_i), log(1+n_support)

    The expert head is supervised by counterfactual latent reconstruction
    reliability.  The action head is supervised by counterfactual
    classification utility.  Both losses belong to the single unified
    imputation objective.
    """

    def __init__(self, input_dim=12, hidden_dim=64, num_experts=4,
                 num_actions=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.expert_head = nn.Linear(hidden_dim, num_experts)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, return_action=False):
        """Return expert logits, and optionally action logits."""
        hidden = self.trunk(x)
        expert_logits = self.expert_head(hidden)
        if return_action:
            return expert_logits, self.action_head(hidden)
        return expert_logits

'''
losses
'''

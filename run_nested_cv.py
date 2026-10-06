"""Command-line entry point for CLUECL3 nested cross-validation."""

import argparse

from trainer import CLCLSA_Trainer
from utils import set_random_seed, str2bool


if __name__ == '__main__':
    # python main_clcl.py --data_folder=ROSMAP --hidden_dim=300 --num_epoch=2500
    # python main_clcl.py --data_folder=BRCA --hidden_dim=200 --num_epoch=2500
    parser = argparse.ArgumentParser()

    # dataset settings
    parser.add_argument('--data_folder', type=str, default="BRCA")
    parser.add_argument('--missing_rate', type=float, default=0,
                        help='Requested missing rate; interpretation is set by missing_rate_mode.')
    parser.add_argument('--missing_rate_mode', type=str, default='sample',
                        choices=['sample', 'entry'],
                        help='"sample": fraction of patients with >=1 missing view; '
                             '"entry": fraction of all sample-view entries missing.')
    parser.add_argument('--two_missing_ratio', type=float, default=0.5,
                        help='In sample mode, fraction of incomplete patients that miss two views.')
    parser.add_argument('--exp', type=str, default="./exp")
    parser.add_argument('--seed', type=int, default=28)#BRCA LGG,GBM,KIPAN28   ROSMAP42

    # model params
    parser.add_argument('--hidden_dim', type=str, default="128")#GBM350,其他128
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help='Number of warmup epochs for cosine annealing scheduler')
    parser.add_argument('--dropout', type=float, default=0.1)#KIPAN,BRCA0.5 GBM,LGG0.1 ROSMAP 0.2
    parser.add_argument('--prediction', type=str, default="64,32")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--use_amp', type=str2bool, default=True,
                        help='Enable AMP mixed precision training (reduces memory, speeds up on Tensor Cores)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (prevents gradient explosion at high missing rates)')


    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for InfoNCE contrastive loss (typical range: 0.05~0.5). Set to 0 to disable.')
    parser.add_argument('--lambda_cil', type=float, default=1.0,
                        help='Weight for the cross-view InfoNCE loss; set to 0 to disable.')
    parser.add_argument('--contrastive_update_interval', type=int, default=5,
                        help=('Compute InfoNCE every N epochs. The faster default '
                              'is 5; use 2 to reproduce the previous schedule or '
                              '1 to compute it every epoch.'))
#al0.9，  seed58
    parser.add_argument('--lambda_al', type=float, default=1)#KIPAN,BRCA0.9-1.1   ,GBM,LGG,ROSMAP2.1
    parser.add_argument('--use_cross_sample_impute', type=str2bool, default=True)
    parser.add_argument('--cross_sample_k', type=int, default=5)
    parser.add_argument('--use_prototype_bank', type=str2bool, default=True)
    parser.add_argument('--support_mode', type=str, default='masked',  # [FIX] safer default
                        choices=['complete', 'masked'],
                        help='Support data source for KNN/Prototype. '
                             '"complete" = pre-masking data (reference panel, '
                             'may leak other samples\' true features). '
                             '"masked" = post-masking data (no leakage, but '
                             'support also has zeros at missing views).')
    parser.add_argument('--knn_base_temperature', type=float, default=0.2,
                        help='Base (upper bound) temperature for adaptive KNN softmax weighting')
    parser.add_argument(
        '--lambda_imputation', type=float, default=0.1,
        help=('Single weight for the unified imputation objective: '
              'pairwise reconstruction + expert-router KL + '
              'classification-utility action-router KL.'))
    parser.add_argument('--router_temperature', type=float, default=0.5,
                        help='Temperature shared by Router soft teachers.')
    parser.add_argument('--router_update_interval', type=int, default=10,
                        help=('Compute both expert and action counterfactual '
                              'Router teachers every N epochs. Use 5 to '
                              'reproduce the previous schedule.'))
    parser.add_argument(
        '--use_action_router', type=str2bool, default=True,
        help=('Enable classification-utility selective routing: skip, '
              'impute the first missing view, or impute the second missing '
              'view. At most one view is imputed per incomplete patient.'))

    # training params
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument(
        '--outer_folds', type=int, default=10,
        help=('Number of outer stratified folds. Each merged patient is used '
              'as a held-out outer-test sample exactly once.'))
    parser.add_argument(
        '--inner_folds', type=int, default=5,
        help=('Number of inner stratified folds used only inside each outer '
              'training set for epoch/model selection.'))
    parser.add_argument('--test_interval', type=int, default=25,
                        help=('Common inner-validation epoch interval. All '
                              'inner folds run to num_epoch so their ACC can '
                              'be averaged at identical epochs.'))

    args = parser.parse_args()
    run_params = vars(args)
    set_random_seed(run_params['seed'])
    run_params['hidden_dim'] = [
        int(x) for x in run_params['hidden_dim'].split(",")]
    run_params['prediction'] = {
        i: [int(x) for x in run_params['prediction'].split(",")]
        for i in range(3)}
    cl_trainer = CLCLSA_Trainer(run_params)
    cl_trainer.train()

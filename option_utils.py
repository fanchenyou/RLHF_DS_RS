import os
import argparse

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of lr')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight_decay on l2 reg')
    parser.add_argument('--lr_sh_rate', type=int, default=100,
                        help='number of steps to drop the lr')
    parser.add_argument('--milestones', type=int, default=[50, 100],
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=True,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='',
                        help='personal tag for the model ')
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-p', '--model_dir', type=str, default='',
                        help='path of test models')
    parser.add_argument('-d', '--dataset', default='eth', required=False,
                        help='eth,hotel,univ,zara1,zara2')
    parser.add_argument('--eps', type=float, default=2e-3,
                        help='rank operation eps')
    parser.add_argument('--seed', type=int, default=1357,
                        help='random seed')
    parser.add_argument('--social_dist_sigma', type=float, default=1,
                        help='social distance, set to 2m')
    parser.add_argument('-uc', '--use_clip', type=int, default=1, choices=[0, 1],
                        help='use clip contrastive loss')
    parser.add_argument('-up', '--use_pairwise_rel', type=int, default=0, choices=[0, 1],
                        help='use future pairwise distance regression')
    parser.add_argument('--a1', type=float, default=None, help='alpha_1')
    parser.add_argument('--a2', type=float, default=None, help='alpha_2')
    # rl parameter
    parser.add_argument('--w_clip', type=float, default=None, help='alpha_1')
    parser.add_argument('--w_rl', type=float, default=None, help='alpha_2')
    # parser.add_argument('--md', type=str, default='train',
    #                     help='pre-training/training/training_rl/testing diffusion model')
    return parser



def parse_args(parser):
    args = parser.parse_args()

    print("Training initiating....")
    print(args)

    if len(args.tag) == 0:
        args.tag = args.dataset

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # region Parameters
    if args.dataset == 'nba':
        args.obs_len = 10
        args.pred_len = 20
    else:
        args.obs_len = 8
        args.pred_len = 12


    return args


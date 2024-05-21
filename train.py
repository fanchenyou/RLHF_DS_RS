import os
import argparse
import pickle
import argparse
import datetime
from torch.utils.data.dataloader import DataLoader
from tools.utils import *
from diffusion_models.trainer import Trainer
from option_utils import get_options, parse_args


def get_data(dataset_name, obs_seq_len, pred_seq_len):
    data_set = './datasets/' + dataset_name + '/'
    processed_path = './dataset_processed/'
    os.makedirs(processed_path, exist_ok=True)
    processed_data_set = './dataset_processed/' + dataset_name + '/'
    os.makedirs(processed_data_set, exist_ok=True)

    # region Dataset selection
    if dataset_name == 'robo':
        from loader import TrajectoryRobotDataset as TrajectoryDataset
    else:
        from loader import TrajectoryDataset

    dset_train = TrajectoryDataset(
        data_set + 'train/',
        processed_data_set + 'train.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,
        shuffle=True,
        num_workers=6)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        processed_data_set + 'val.pth',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=4)

    # endregion
    return loader_train, loader_val


def get_save_dir(dset, reject_sampling, use_rs_all, use_ip_rl):
    result_dir = './result_diffusion/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    run_id = datetime.datetime.now().strftime('%m-%d-%H-%M')
    if args.md == 'pre':
        checkpoint_dir = os.path.join(result_dir, "pre_%s_%s" % (dset, run_id))
    else:
        suffix = 'rl'
        if reject_sampling:
            suffix = 'rl_rs'
        elif use_rs_all:
            suffix= 'rl_rs_all'
        elif use_ip_rl:
            suffix='rl_ip'
        checkpoint_dir = os.path.join(result_dir, "diff_%s_%s_%s" % (dset, run_id, suffix))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print('Checkpoint dir %s' % (checkpoint_dir,))
    return checkpoint_dir


def get_test_data(dataset_name, obs_seq_len, pred_seq_len, permute=True):
    data_set = './datasets/' + args.dataset + '/'
    processed_data_set = './dataset_processed/' + args.dataset + '/'
    assert os.path.isdir(processed_data_set)

    if dataset_name == 'robo':
        from loader import TrajectoryRobotDataset as TrajectoryDataset
    else:
        from loader import TrajectoryDataset

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        processed_data_set + 'test.pkl',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4)
    return loader_test


if __name__ == '__main__':

    parser = get_options()
    parser.add_argument('--md', type=str, default='train',
                        help='pre-training/training/training_rl/testing diffusion model')
    parser.add_argument('-rs', '--reject_sampling', action="store_true", default=True,
                        help='Use reject sampling')
    parser.add_argument('-rs_all', '--use_rs_all', action="store_true", default=False,
                        help='Use reject sampling')
    parser.add_argument('-ds', '--diff_score', action="store_true", default=False,
                        help='Use diff score when test')
    parser.add_argument('-t', '--test', action="store_true", default=False,
                        help='testing model')
    parser.add_argument('--use_ip_rl', action="store_true", default=False,
                        help='Use ipscore as RL reward')
    parser.add_argument('--suffix', type=str, default='', help='path of test models')
    args = parse_args(parser)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)

    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len

    if args.test:
        args.md = 'test'

    if args.md == 'test':
        loader_test = get_test_data(args.dataset, obs_seq_len, pred_seq_len)
        checkpoint_dir = args.model_dir

        assert len(args.model_dir) > 0
        model_path = os.path.join(args.model_dir, 'val_best.pth')
        t = Trainer(args.dataset, obs_seq_len, pred_seq_len,
                    device, None, loader_test, args.lr, checkpoint_dir, args.md, model_path,
                    args)
        if args.md=='test':
            t.test_single_model()
        elif args.md=='test_vis_rs_reward':
            t.test_single_model_vis_rs_reward()
        elif args.md=='test_vis_trajectory':
            assert len(args.suffix)>0
            # t.test_single_model_vis_trajectory(args.suffix)
            t.test_single_model_vis_adf_trajectory(args.suffix)
        else:
            assert 1==2

    else:

        loader_train, loader_val = get_data(args.dataset, obs_seq_len, pred_seq_len)
        checkpoint_dir = get_save_dir(args.dataset, args.reject_sampling, args.use_rs_all, args.use_ip_rl)
        model_path = None
        if len(args.model_dir) > 0:
            model_path = os.path.join(args.model_dir, 'val_best.pth')
            assert os.path.isfile(model_path)
        t = Trainer(args.dataset, obs_seq_len, pred_seq_len, device, loader_train, loader_val, args.lr, checkpoint_dir,
                    args.md, model_path, args)

        if args.md == 'train':
            t.fit()
        elif args.md == 'train_rl':
            t.fit_rl()
        elif args.md == 'pre':
            t.pre()
        elif args.md == 'pre_test':
            t.pre_test()

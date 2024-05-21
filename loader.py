import os
import math
import pickle as pkl
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from tools.utils import *
from tools.data_utils import *
import random


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, data_pkl_file, obs_len=8, pred_len=8, skip=1, threshold=0.002,
                 min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        if not os.path.isfile(data_pkl_file):

            all_files = os.listdir(self.data_dir)
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            loss_mask_list = []
            non_linear_ped = []

            for path in all_files:
                data = read_file(path, delim)

                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                             self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                               self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                     ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        # ipdb.set_trace()
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        # rel_curr_ped_seq[:, 1:] = \
                        #     curr_ped_seq[:, 1:] - np.reshape(curr_ped_seq[:, 0], (2,1))
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)

            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert to Graphs
            self.v_obs = []
            self.v_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                # greatly reduce graph construction
                v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
                self.v_obs.append(v_.clone())
                # print(v_.size())
                v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
                # print(v_.size())
                self.v_pred.append(v_.clone())

            pbar.close()

            torch.save([self.v_obs, self.v_pred, self.obs_traj, self.pred_traj,
                        self.obs_traj_rel, self.pred_traj_rel, self.loss_mask, self.non_linear_ped,
                        self.seq_start_end], data_pkl_file)

            print('Save to %s' % (data_pkl_file,))

            data = torch.load(data_pkl_file)

        else:
            data = torch.load(data_pkl_file)
            self.v_obs, self.v_pred, self.obs_traj, self.pred_traj, self.obs_traj_rel, \
                self.pred_traj_rel, self.loss_mask, self.non_linear_ped, self.seq_start_end = data
            self.num_seq = len(self.seq_start_end)
            print('Load from %s with %d data' % (data_pkl_file, self.num_seq))
            # print(self.v_obs[0], self.A_obs[-1])

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.v_pred[index]
        ]
        return out


class TrajectoryRobotDataset(Dataset):
    """Dataloder for the Robotics Trajectory datasets"""

    def __init__(self, data_dir=None, data_pkl_file=None, obs_len=8, pred_len=8, skip=1, threshold=0.002,
                 min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryRobotDataset, self).__init__()
        data_folder = './datasets/robo'
        agent_names = ['agent_1', 'agent_2', 'AV']

        # a=list(range(1,101))
        # random.shuffle(a)
        # print(a)
        # exit()
        shuffle_list = [22, 59, 26, 88, 19, 76, 41, 49, 69, 33, 60, 74, 29, 83, 39, 50, 45, 57, 40, 78, 52, 56, 100, 66,
                        89, 34, 20, 3, 94, 92, 93, 58, 70, 80, 10, 35, 84, 63, 44, 48, 16, 47, 5, 71, 28, 91, 82, 53,
                        97, 79, 46, 77, 67, 72, 90, 85, 37, 87, 51, 8, 1, 18, 12, 73, 27, 36, 7, 9, 81, 75, 11, 64, 13,
                        43, 61, 98, 31, 96, 54, 25, 38, 99, 42, 2, 65, 86, 68, 55, 17, 30, 62, 21, 32, 15, 14, 23, 4,
                        95, 6, 24]

        data_split = {'train': shuffle_list[:60],
                      'val': shuffle_list[60:80],
                      'test': shuffle_list[80:], }

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.num_peds = 3

        if not os.path.isfile(data_pkl_file):
            if 'train' in data_dir:
                split = 'train'
            elif 'val' in data_dir:
                split = 'val'
            else:
                split = 'test'
            split_nums = data_split[split]

            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            loss_mask_list = []
            non_linear_ped = []

            for sn in split_nums:
                # print(sn)
                agent_data = []
                agent_length = []
                for agent_name in agent_names:
                    file_name = os.path.join(data_folder, agent_name, '%s_pose%d.json' % (agent_name, sn))
                    assert os.path.isfile(file_name)
                    data = json.load(open(file_name))
                    agent_data.append(data)
                    agent_length.append(len(data))
                    # print(len(data),'--1')
                # print(sn,'===')
                agent_min_length = min(agent_length)
                num_sequences = int(math.ceil((agent_min_length - self.seq_len) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):

                    # curr_seq_data = np.zeros(1,self.num_peds, self.seq_len, 3)
                    # curr_seq_data = [data[idx:self.seq_len] for data in agent_data]
                    # curr_seq_data = np.concatenate(
                    #     frame_data[idx:idx + self.seq_len], axis=0)
                    # peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = self.num_peds  # max(self.max_peds_in_frame, len(peds_in_curr_seq))
                    curr_seq_rel = np.zeros((self.num_peds, 2, self.seq_len))
                    curr_seq = np.zeros((self.num_peds, 2, self.seq_len))
                    curr_loss_mask = np.zeros((self.num_peds, self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    # print(idx,'--',agent_min_length)
                    for ped_id in range(self.num_peds):
                        curr_seq_data_json = agent_data[ped_id][idx:idx + self.seq_len]
                        # print(len(curr_seq_data_json))
                        curr_seq_data = [[tmp_data['x'], tmp_data['y']] for tmp_data in curr_seq_data_json]
                        # print(curr_seq_data)
                        curr_seq_data = np.stack(curr_seq_data)
                        curr_ped_seq = np.around(curr_seq_data, decimals=4)
                        # print(curr_seq_data.shape)
                        # curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = np.transpose(curr_ped_seq)
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        # ipdb.set_trace()
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        # rel_curr_ped_seq[:, 1:] = \
                        #     curr_ped_seq[:, 1:] - np.reshape(curr_ped_seq[:, 0], (2,1))
                        _idx = num_peds_considered
                        # print(curr_seq.shape, curr_ped_seq.shape)
                        curr_seq[_idx, :, :] = curr_ped_seq
                        curr_seq_rel[_idx, :, :] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, :] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)

            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert to Graphs
            self.v_obs = []
            self.v_pred = []
            print("Processing Data .....")
            # exit()

            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                # greatly reduce graph construction
                v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
                self.v_obs.append(v_.clone())
                # print(v_.size())
                v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
                # print(v_.size())
                self.v_pred.append(v_.clone())

            pbar.close()

            torch.save([self.v_obs, self.v_pred, self.obs_traj, self.pred_traj,
                        self.obs_traj_rel, self.pred_traj_rel, self.loss_mask, self.non_linear_ped,
                        self.seq_start_end], data_pkl_file)

            print('Save to %s' % (data_pkl_file,))

            data = torch.load(data_pkl_file)

        else:
            data = torch.load(data_pkl_file)
            self.v_obs, self.v_pred, self.obs_traj, self.pred_traj, self.obs_traj_rel, \
                self.pred_traj_rel, self.loss_mask, self.non_linear_ped, self.seq_start_end = data
            self.num_seq = len(self.seq_start_end)
            print('Load from %s with %d data' % (data_pkl_file, self.num_seq))
            print(self.obs_traj.shape, self.pred_traj_rel.shape)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.v_pred[index]
        ]
        return out


def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D, pre_motion_mask, fut_motion_mask) = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D, dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D, dim=0)
    fut_motion_mask = torch.stack(fut_motion_mask, dim=0)
    pre_motion_mask = torch.stack(pre_motion_mask, dim=0)

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'nba',
    }
    # out = [
    #     batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum
    # ]

    return data


class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, obs_len=5, pred_len=10, training=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        print(self.obs_len, self.pred_len)

        if training:
            data_root = './dataset_processed/nba/nba_train.npy'
        else:
            data_root = './dataset_processed/nba/nba_test.npy'

        self.trajs = np.load(data_root)  # (N,15,11,2)
        self.trajs /= (94 / 28)
        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:32500:50]
            # self.trajs = self.trajs[12500:25000]

        self.batch_len = len(self.trajs)
        print(self.batch_len)

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs - self.trajs[:, self.obs_len - 1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0, 2, 3, 1)
        self.traj_norm = self.traj_norm.permute(0, 2, 3, 1)

        self.actor_num = self.traj_abs.shape[1]
        assert self.actor_num == 11

        # torch.Size([12500, 11, 30, 2]), torch.Size([12500, 11, 30, 2]), 11
        print(self.traj_abs.shape, self.traj_norm.shape, self.actor_num)

        num_sequences = self.traj_abs.size(0)
        print(num_sequences)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :, :self.obs_len]
        fut_motion_3D = self.traj_abs[index, :, :, self.obs_len:]
        pre_motion_rel = self.traj_norm[index, :, :, :self.obs_len]
        fut_motion_rel = self.traj_norm[index, :, :, self.obs_len:]

        # print(pre_motion_3D.size(),fut_motion_3D.size(),pre_motion_rel.size(),fut_motion_3D.size())
        v_obs = seq_to_graph(pre_motion_3D, pre_motion_rel, True)
        v_pred = seq_to_graph(fut_motion_3D, fut_motion_rel, False)

        pre_motion_mask = torch.ones(11, self.obs_len)
        fut_motion_mask = torch.ones(11, self.pred_len)
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_rel, fut_motion_rel,
            pre_motion_mask, fut_motion_mask,
            v_obs, v_pred
        ]
        return out

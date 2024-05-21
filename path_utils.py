import torch
import numpy as np

def get_dist(x1, y1, x2, y2):
    dist = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def get_social_dist_exp(dt, two_sigma_square):
    val = torch.exp(-dt ** 2 / two_sigma_square)
    return val


# accumulate the one-step relative (x,y) distance
# add up to the initial position
# provides the final absolute position
def nodes_xy_to_nodes_abs_tensor(seq_x_rel, seq_y_rel, x_init_abs, y_init_abs):
    # seq_x/y_rel is (12, N_obj)
    # x/y_init_abs is (1, N_obj)
    seq_x_abs = torch.cumsum(seq_x_rel, dim=0) + x_init_abs[0:1, :]
    seq_y_abs = torch.cumsum(seq_y_rel, dim=0) + y_init_abs[0:1, :]
    return seq_x_abs, seq_y_abs

def nodes_rel_to_nodes_abs_tensor(seq_rel, init_abs):
    # seq_x/y_rel is (12, N_obj, 2)
    # x/y_init_abs is (1, N_obj, 2)
    seq_abs = torch.cumsum(seq_rel, dim=0) + init_abs[0:1, :, :]
    return seq_abs


# calculate the cost of a path
# see "Understanding Human Avoidance Behavior"
# cost function II
# seq_x_rel size is (fut_len=12, N)
# TODO: use history length as a basis, divide pred_length with hist_length, as energy associated with the length of segment
# To add spacing-varying (step-dependent) feature, we pass in final_state
# check "Generation of human walking paths", Papadopoulos et al., 2016

def calc_rotate(seq_x, seq_y):
    # segment length -- lambda_k
    seq_len = torch.sqrt(seq_x ** 2 + seq_y ** 2)
    # print(seq_x_rel.size(), seq_len.size())
    # seq_len[seq_len.eq(0)] = 1e-5

    # segment degree -- phi_k
    # TODO: check NaN, should be [-pi,pi]
    seq_phi_atan2 = torch.atan2(seq_y, seq_x)
    # seq_phi_sine[seq_len.eq(0)] = 0

    # 30 deg -> 360 - 30 deg
    # t_ind = seq_x<0
    # seq_phi_sine[t_ind]

    # normalize arccosine from [0, pi] to [0,1]
    return seq_len, seq_phi_atan2


def calc_unary_path_energy(seq_x_rel, seq_y_rel, seq_x_hist_rel, seq_y_hist_rel, to_sum=True):
    assert seq_x_rel.ndim == 2 and seq_x_hist_rel.ndim == 2
    assert seq_x_rel.size(0) == seq_y_rel.size(0)

    # from dataloader
    # seq_x_rel <- x(k+1) - x(k)
    # seq_y_rel y(k+1) - y(k)

    seq_len = torch.sqrt(seq_x_rel ** 2 + seq_y_rel ** 2)

    seq_len_no_zero, seq_phi_atan2 = calc_rotate(seq_x_rel, seq_y_rel)
    seq_rotation_norm = seq_phi_atan2 / torch.pi

    # history: last one segment is used to compute the rotation
    _, seq_0_phi_atan2 = calc_rotate(seq_x_hist_rel[-1:, :], seq_y_hist_rel[-1:, :])
    seq_0_rotation_norm = seq_0_phi_atan2 / torch.pi

    # print(seq_0_rotation_norm.size())
    ## check the degree of atan2
    if 2==3:
        for j in range(seq_0_rotation_norm.size(1)):
            # 90 deg
            if np.abs(seq_0_rotation_norm[0,j].item()-0.5)<1e-3:
                print(seq_x_hist_rel[-1:, j], seq_y_hist_rel[-1:, j],seq_0_rotation_norm[0,j].item())
                print()
            # -90 deg
            if np.abs(seq_0_rotation_norm[0,j].item()+0.5)<1e-3:
                print(seq_x_hist_rel[-1:, j], seq_y_hist_rel[-1:, j],seq_0_rotation_norm[0,j].item())
                print()
            # 0 deg -- only x-axis changes
            if np.abs(seq_0_rotation_norm[0,j].item())<1e-3:
                print(seq_x_hist_rel[-1:, j], seq_y_hist_rel[-1:, j],seq_0_rotation_norm[0,j].item())
                print()
            if np.abs(seq_0_rotation_norm[0,j].item()-0.25)<1e-3:
                print(seq_x_hist_rel[-1:, j], seq_y_hist_rel[-1:, j],seq_0_rotation_norm[0,j].item())
                print()
            if np.abs(seq_0_rotation_norm[0,j].item()+0.25)<1e-3:
                print(seq_x_hist_rel[-1:, j], seq_y_hist_rel[-1:, j],seq_0_rotation_norm[0,j].item())
                print()

    # then compute the curvature as the division of segment rotation delta and segment length
    seq_rotation_norm_delta = torch.zeros_like(seq_rotation_norm).to(seq_rotation_norm.device)
    seq_rotation_norm_delta[0:1, :] += seq_rotation_norm[0:1, :] - seq_0_rotation_norm[-1:, :]
    seq_rotation_norm_delta[1:, :] += seq_rotation_norm[1:, :] - seq_rotation_norm[:-1, :]
    # print(seq_rotation_norm)
    # print(seq_rotation_norm_delta)
    curvature = seq_rotation_norm_delta / seq_len_no_zero

    val = seq_len * curvature ** 2

    if to_sum:
        energy = torch.sum(val, dim=0)
        return energy

    return val



'''
See paper Eq(9)
The cost function rates if a trajectory 
leads towards its goal and maintains a desired speed.
Additionally, it rewards trajectories that steer an agent
away from an expected point of closest approach to another
agent.
'''

def calc_binary_path_energy(social_dist_two_sigma_square, seq_abs, to_sum=True):


    assert seq_abs.ndim == 3 and seq_abs.ndim == 3
    # Input size is [12, N_obj]

    # [12, N_obj, N_obj]
    seq_abs_diff = seq_abs.unsqueeze(2) - seq_abs.unsqueeze(1)

    # energy calculation
    seq_xy_abs_dist_sq = torch.sum(seq_abs_diff ** 2, dim=-1) #seq_abs_diff ** 2 + seq_abs_diff ** 2
    dist_energy_all = torch.exp(-seq_xy_abs_dist_sq / social_dist_two_sigma_square)
    if to_sum:
        total_energy_matrix = torch.mean(dist_energy_all,dim=0)
        return total_energy_matrix
    return dist_energy_all, torch.sqrt(seq_xy_abs_dist_sq)


def calc_binary_path_energy_xy(social_dist_two_sigma_square, seq_x_abs, seq_y_abs, to_sum=True):


    # assert seq_x_abs.ndim == 2 and seq_y_abs.ndim == 2
    # Input size is [12, N_obj]

    # [12, N_obj, N_obj]
    seq_x_abs_diff = seq_x_abs.unsqueeze(-1) - seq_x_abs.unsqueeze(1)
    seq_y_abs_diff = seq_y_abs.unsqueeze(-1) - seq_y_abs.unsqueeze(1)

    # energy calculation
    seq_xy_abs_dist_sq = seq_x_abs_diff ** 2 + seq_y_abs_diff ** 2
    dist_energy_all = torch.exp(-seq_xy_abs_dist_sq / social_dist_two_sigma_square)
    if to_sum:
        total_energy_matrix = torch.mean(dist_energy_all,dim=0)
        return total_energy_matrix
    return dist_energy_all
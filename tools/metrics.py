import math
import torch
import numpy as np
import torch.distributions.multivariate_normal as torchdist


def ade(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_, max_nodes=88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def nodes_sample_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[1]):
        for ped in range(nodes.shape[2]):
            # [N_sample, 2] += [1, 2]
            nodes_[:, s, ped, :] = np.sum(nodes[:, :s + 1, ped, :], axis=1) + init_node[ped:ped + 1, :]

    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def pairwise_loss(pred_rank, true_rank):
    # N = len(pred_rank)
    pred_rank_diff = pred_rank.unsqueeze(-1) - pred_rank.unsqueeze(0)
    true_rank_diff = true_rank.unsqueeze(-1) - true_rank.unsqueeze(0)

    pred_true_prod = - pred_rank_diff * true_rank_diff
    pred_true_loss = torch.clamp(pred_true_prod, min=0)
    loss_rank = torch.mean(pred_true_loss)

    return loss_rank


def get_multivariate_normal_dist(V_pred):
    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr
    #
    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(V_pred.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy
    mean = V_pred[:, :, 0:2]
    mvnormal = torchdist.MultivariateNormal(mean, cov)
    return mean, cov, mvnormal


# https://stackoverflow.com/questions/43031731/negative-values-in-log-likelihood-of-a-bivariate-gaussian
# see Social-LSTM Eq(3)-(4)
def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result


def intimacy_politeness_score(pred_pair_xy, true_pair_xy, social_dist, hinge_ratio=0.25):
    # assert pred_pair_xy.ndim == true_pair_xy.ndim == 4
    # assert pred_pair_xy.size(-1) == true_pair_xy.size(-1) == 2
    # print(pred_pair_xy.size(), true_pair_xy.size())
    pred_pair_dist = torch.sqrt(torch.sum(pred_pair_xy ** 2, dim=-1))
    true_pair_dist = torch.sqrt(torch.sum(true_pair_xy ** 2, dim=-1))

    # assume pred_pair is [12, N_obj, N_obj]
    # if avg dist is less than 1SD
    # we think of them as of one same group
    # otherwise think of them as of different groups
    # print(social_dist)

    true_pair_avg_dist = torch.mean(true_pair_dist, dim=0, keepdim=True)

    # hinge loss of intimacy
    # find out the moments these grouped agents are less than 1SD
    intimacy_score_2 = 0
    same_group_mask = (torch.logical_and(true_pair_avg_dist > 0, true_pair_avg_dist <= 1.0 * social_dist)). \
        repeat(pred_pair_dist.size(0), 1, 1)
    same_group_intimate_mask = true_pair_dist[same_group_mask] <= 1.0 * social_dist
    true_pair_dist_masked = true_pair_dist[same_group_mask][same_group_intimate_mask]
    pred_pair_dist_masked = pred_pair_dist[same_group_mask][same_group_intimate_mask]
    # score is 1 from 0~true_d, then goes down to 0 at 1.25*true_d
    if true_pair_dist_masked.size(0) > 0:
        if 1 == 2:
            for pred_val, true_val in zip(pred_pair_dist_masked, true_pair_dist_masked):
                sc = 0
                if pred_val <= true_val:
                    sc = 1.0  # reward is 1
                else:
                    sc = ((1 + hinge_ratio) * true_val - pred_val) / (hinge_ratio * true_val)
                    sc = sc.item()
                    if sc < 0:
                        sc = 0
                    assert 0 <= sc <= 1.0
                intimacy_score += sc
            intimacy_score /= true_pair_dist_masked.size(0)

        intimacy_score_all = ((1 + hinge_ratio) * true_pair_dist_masked - pred_pair_dist_masked) / (
                hinge_ratio * true_pair_dist_masked)
        intimacy_score_all = torch.clamp(intimacy_score_all, min=0, max=1)
        intimacy_score_2 = torch.mean(intimacy_score_all).item()
        # print(intimacy_score_2, intimacy_score)

    # hinge loss of intimacy
    # find out the moments these grouped agents are less than 1SD
    politeness_score_2 = 0
    diff_group_mask = (true_pair_avg_dist > 1.0 * social_dist).repeat(pred_pair_dist.size(0), 1, 1)
    diff_group_politeness_mask = true_pair_dist[diff_group_mask] > 1.0 * social_dist
    true_pair_dist_masked = true_pair_dist[diff_group_mask][diff_group_politeness_mask]
    pred_pair_dist_masked = pred_pair_dist[diff_group_mask][diff_group_politeness_mask]
    # score is 1 from 0~true_d, then goes down to 0 at 1.25*true_d
    if true_pair_dist_masked.size(0) > 0:
        if 1 == 2:
            for pred_val, true_val in zip(pred_pair_dist_masked, true_pair_dist_masked):
                sc = 0
                if pred_val >= true_val:
                    sc = 1.0  # cost is 1
                else:
                    sc = (pred_val - (1 - hinge_ratio) * true_val) / (hinge_ratio * true_val)
                    sc = sc.item()
                    if sc < 0:
                        sc = 0
                    assert 0 <= sc <= 1.0
                politeness_score += sc
            politeness_score /= true_pair_dist_masked.size(0)

        politeness_score_all = (pred_pair_dist_masked - (1 - hinge_ratio) * true_pair_dist_masked) / (
                hinge_ratio * true_pair_dist_masked)
        politeness_score_all = torch.clamp(politeness_score_all, min=0, max=1)
        politeness_score_2 = torch.mean(politeness_score_all).item()
        # print(politeness_score_2, politeness_score)

    return intimacy_score_2, politeness_score_2


def intimacy_politeness_score_per_step_OLD(pred_pair_xy, true_pair_xy, social_dist, hinge_ratio=0.25):
    # assert pred_pair_xy.ndim == true_pair_xy.ndim == 4
    # assert pred_pair_xy.size(-1) == true_pair_xy.size(-1) == 2
    # print(pred_pair_xy.size(), true_pair_xy.size())
    pred_pair_dist = torch.sqrt(torch.sum(pred_pair_xy ** 2, dim=-1))
    true_pair_dist = torch.sqrt(torch.sum(true_pair_xy ** 2, dim=-1))

    # assume pred_pair is [12, N_obj, N_obj]
    # if avg dist is less than 1SD
    # we think of them as of one same group
    # otherwise think of them as of different groups

    T, n_obj = true_pair_dist.size(0), true_pair_dist.size(1)
    true_pair_avg_dist = torch.mean(true_pair_dist, dim=0, keepdim=False)

    scores = torch.zeros(pred_pair_dist.size(0), pred_pair_dist.size(1),
                         device=pred_pair_dist.device)

    intimacy_score_all = ((1 + hinge_ratio) * true_pair_dist - pred_pair_dist) / (
            hinge_ratio * true_pair_dist)
    intimacy_score_all = torch.clamp(intimacy_score_all, min=0, max=1)

    politeness_score_all = (pred_pair_dist - (1 - hinge_ratio) * true_pair_dist) / (
            hinge_ratio * true_pair_dist)
    politeness_score_all = torch.clamp(politeness_score_all, min=0, max=1)

    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            if true_pair_avg_dist[i, j] <= 1.0 * social_dist:
                # same group
                for t in range(T):
                    if true_pair_dist[t, i, j] <= 1.0 * social_dist:
                        # assert intimacy_score_all[t, i, j]==intimacy_score_all[t, j, i]
                        scores[t, i] += intimacy_score_all[t, i, j]
                        scores[t, j] += intimacy_score_all[t, j, i]
            else:
                for t in range(T):
                    if true_pair_dist[t, i, j] > 1.0 * social_dist:
                        # assert politeness_score_all[t, i, j]==politeness_score_all[t, j, i]
                        scores[t, i] += politeness_score_all[t, i, j]
                        scores[t, j] += politeness_score_all[t, j, i]

    return scores


def intimacy_politeness_score_per_step(pred_pair_xy, true_pair_xy, social_dist, hinge_ratio=0.25):
    # assert pred_pair_xy.ndim == true_pair_xy.ndim == 4
    # assert pred_pair_xy.size(-1) == true_pair_xy.size(-1) == 2
    # print(pred_pair_xy.size(), true_pair_xy.size())
    # hinge_ratio = 0.0
    pred_pair_dist = torch.sqrt(torch.sum(pred_pair_xy ** 2, dim=-1))
    true_pair_dist = torch.sqrt(torch.sum(true_pair_xy ** 2, dim=-1))
    # print(social_dist)

    # assume pred_pair is [12, N_obj, N_obj]
    # if avg dist is less than 1SD
    # we think of them as of one same group
    # otherwise think of them as of different groups

    T, n_obj = true_pair_dist.size(0), true_pair_dist.size(1)
    true_pair_avg_dist = torch.mean(true_pair_dist, dim=0, keepdim=False)

    scores = torch.zeros(pred_pair_dist.size(0), pred_pair_dist.size(1),
                         device=pred_pair_dist.device)

    intimacy_score_all = ((1 + hinge_ratio) * true_pair_dist - pred_pair_dist) / (hinge_ratio * true_pair_dist + 1e-5)
    intimacy_score_all = torch.clamp(intimacy_score_all, min=0, max=1)

    politeness_score_all = (pred_pair_dist - (1 - hinge_ratio) * true_pair_dist) / (hinge_ratio * true_pair_dist + 1e-5)
    politeness_score_all = torch.clamp(politeness_score_all, min=0, max=1)

    counter = torch.zeros_like(scores)

    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            if true_pair_avg_dist[i, j] <= 1.0 * social_dist:
                # same group
                for t in range(T):
                    if true_pair_dist[t, i, j] <= 1.0 * social_dist:
                        # assert intimacy_score_all[t, i, j]==intimacy_score_all[t, j, i]
                        # if pred_pair_dist[t, i, j] > 1.0 * social_dist:
                        #     assert intimacy_score_all[t, i, j] < 0
                        scores[t, i] += intimacy_score_all[t, i, j]
                        scores[t, j] += intimacy_score_all[t, j, i]
                        counter[t, i] += 1
                        counter[t, j] += 1

            else:
                for t in range(T):
                    if true_pair_dist[t, i, j] > 1.0 * social_dist:
                        # assert politeness_score_all[t, i, j]==politeness_score_all[t, j, i]
                        if pred_pair_dist[t, i, j] < 1.0 * social_dist:
                            # print(politeness_score_all[t, i, j],'00000=====')
                            # assert politeness_score_all[t, i, j] < 0
                            scores[t, i] += politeness_score_all[t, i, j]
                            scores[t, j] += politeness_score_all[t, j, i]
                        else:
                            scores[t, i] += min(2e-1, politeness_score_all[t, i, j])
                            scores[t, j] += min(2e-1, politeness_score_all[t, j, i])
                        counter[t, i] += 1
                        counter[t, j] += 1
    scores /= (counter+1e-5)
    # print(scores)

    return scores

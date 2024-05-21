import torch
import torch.nn.functional as F

class MinimumL2Loss:
    def __init__(self, margin=10.0):
        self.margin = margin

    def __call__(self, outputs, targets):
        diff = (outputs - targets)
        diff_c = torch.clamp(diff, min=-self.margin, max=self.margin)
        loss = (diff_c**2).mean()
        return loss

class hingeloss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, outputs, targets):
        a = self.margin - outputs * targets
        b = torch.clamp(a, min=0)
        loss = torch.mean(b)
        return loss

# https://blog.csdn.net/google19890102/article/details/79496256
class logloss:
    def __init__(self):
        pass

    def __call__(self, outputs, targets):
        a = outputs * targets
        b = torch.log(1 + torch.exp(-a))
        loss = torch.mean(b)
        return loss

def pairwise_loss(pred_rank, true_rank):
    # N = len(pred_rank)
    pred_rank_diff = pred_rank.unsqueeze(-1) - pred_rank.unsqueeze(0)
    true_rank_diff = true_rank.unsqueeze(-1) - true_rank.unsqueeze(0)

    pred_true_prod = - pred_rank_diff * true_rank_diff
    pred_true_loss = torch.clamp(pred_true_prod, min=0)
    loss_rank = torch.sqrt(torch.mean(pred_true_loss))

    return loss_rank


class l1loss:
    def __init__(self):
        pass

    def __call__(self, model):
        regularization_loss = 0

        for param in model.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        return regularization_loss


class l2loss:
    def __init__(self):
        pass

    def __call__(self, model):
        regularization_loss = 0

        # for param in model.parameters():
        #     # print(torch.sum(param**2))
        #     regularization_loss += torch.sum(param**2)

        for name, param in model.named_parameters():
            if 'value_head' in name or 'output' in name:
                regularization_loss += torch.sum(param ** 2)

        return regularization_loss

class l2loss_new:
    def __init__(self):
        pass

    def __call__(self, model):
        regularization_loss = 0

        for name, param in model.named_parameters():
            if ('scale_encoder' in name or 'var_decoder' in name or 'mean_decoder' in name
                    or 'scale_decoder' in name or 'value_head' in name):
                #print(name)
                regularization_loss += torch.sum(param ** 2)

        return regularization_loss


class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None, gamma=2., reduction='mean'):
        torch.nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()
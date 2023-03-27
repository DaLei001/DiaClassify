import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="imiTOqzHp0lZLqDOHdoPcm678",
    project_name="general",
    workspace="dalei001",
)
# def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
def hierarchical_contrastive_loss(z1, z2, z3, z4, label_1, label_2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    ## supervised learning part
    # label_1 = torch.LongTensor(label_1)
    # label_2 = torch.LongTensor(label_2)
    z3 = np.squeeze(z3)
    z4 = np.squeeze(z4)
    # z3 = nn.Sigmoid()(z3)
    # z4 = nn.Sigmoid()(z4)
    loss_1_supervised_learning = nn.BCEWithLogitsLoss()(z3, label_1.float())
    loss_2_supervised_learning = nn.BCEWithLogitsLoss()(z4, label_2.float())

    loss_supervised_learning = (loss_1_supervised_learning + loss_2_supervised_learning)/2
    # loss_supervised_learning = loss_1_supervised_learning
    # loss_supervised_learning = 0.8 * loss_1_supervised_learning + 0.2 * loss_2_supervised_learning  #'acc': 0.86364, 'auprc': 0.84862
    # loss_supervised_learning = 1 * loss_1_supervised_learning + 0 * loss_2_supervised_learning  #'acc': 0.92045, 'auprc': 0.88664
    # loss_supervised_learning = 0 * loss_1_supervised_learning + 1 * loss_2_supervised_learning  #'acc': 0., 'auprc': 0.
    # loss_supervised_learning = 0.2 * loss_1_supervised_learning + 0.8 * loss_2_supervised_learning  #'acc': 0., 'auprc': 0.

    ## self-supervised learning part
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1

    # return loss / d
    # return (loss / d) * 0.5 + loss_supervised_learning * 0.5
    metrics = {"loss1":loss_1_supervised_learning,
               "loss2":loss_2_supervised_learning,
               "loss_SSL":loss / d,
               "total_loss":(loss / d) * 0.2 + loss_supervised_learning * 0.8}
    experiment.log_metrics(metrics)
    return (loss / d) * 0.2 + loss_supervised_learning * 0.8

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

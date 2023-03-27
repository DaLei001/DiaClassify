import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 设置随机数种子
# setup_seed(20)

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
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

    return loss / d


def hierarchical_contrastive_loss_new(z1, z2, z3, z4, label_1, label_2, alpha=0.5, temporal_unit=0, uncertaintyloss=None):
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
    # return (loss / d) * 0.8 + loss_supervised_learning * 0.2
    return loss / d, loss_supervised_learning

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

# 不确定损失函数
class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss

        # for i in range(self.v_num):
        #     # loss += input[i] / (2 * self.sigma[i] ** 2)
        #     loss += input[i] / (self.sigma[i] ** 2)
        #     loss += torch.log(self.sigma[i]) # UW {'acc': 0.8409090909090909, 'auprc': 0.9445863347351199}
        #     # loss += torch.log(1+torch.log(self.sigma[i].pow(2))) # RUW {'acc': 0.8409090909090909, 'auprc': 0.9252091525005544}
        # # a = self.sigma.pow(2)
        # # b = a.prod()
        # # c = torch.log(b)
        # # loss += c
        # # loss += torch.log(self.sigma.pow(2).prod())
        # return loss

# 动态加权平均损失函数
class DynamicWeightAverageLoss(nn.Module):
    def __init__(self, v_num, opt_weight='dwa',total_epoch = 300):
        super(DynamicWeightAverageLoss, self).__init__()
        self.opt_weight = opt_weight
        self.T = 2.0
        self.lambda_weight = np.ones([3, total_epoch])
        self.avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
        self.total_epoch = total_epoch
        # self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        # self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5]))
        self.logsigma = nn.Parameter(torch.randn(v_num))
        self.v_num = v_num


    def forward(self, n_epochs, *input):
        
        index = n_epochs
        train_loss = [input[i]  for i in range(self.v_num)]
        if self.opt_weight == 'dwa':
            if index == 0 or index == 1:
                self.lambda_weight[:, index] = 1.0
            else:
                w_1 = self.avg_cost[index - 1, 0] / (self.avg_cost[index - 2, 0] + 1e-10)
                w_2 = self.avg_cost[index - 1, 1] / (self.avg_cost[index - 2, 1] + 1e-10)
                self.lambda_weight[0, index] = 2 * np.exp(w_1 / self.T) / (np.exp(w_1 / self.T) + np.exp(w_2 / self.T))
                self.lambda_weight[1, index] = 2 * np.exp(w_2 / self.T) / (np.exp(w_1 / self.T) + np.exp(w_2 / self.T))

        if self.opt_weight == 'equal' or self.opt_weight == 'dwa':
            # loss = sum([self.lambda_weight[i, index] * train_loss[i] for i in range(self.v_num)])
            loss = sum([(1 / (2 * torch.exp(self.logsigma[i]))) * self.lambda_weight[i, index] * train_loss[i] for i in range(self.v_num)])
        else:
            loss = sum(1 / (2 * torch.exp(self.logsigma[i])) * train_loss[i] + self.logsigma[i] / 2 for i in range(2))
        return loss

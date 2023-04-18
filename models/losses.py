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

def hierarchical_contrastive_loss_neighbor(z1, z2, neighbor, alpha=0.5, temporal_unit=0):
    # z1,z2: [B x T x Co]
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
        loss += supervised_contrastive_loss(z1,z2,neighbor)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1

    return loss / d

def hierarchical_contrastive_loss_return2(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    loss1 = torch.tensor(0., device=z1.device)
    loss2 = torch.tensor(0., device=z1.device)
    d = 0
    ## self-supervised learning part
    while z1.size(1) > 1:
        if alpha != 0:
            # loss += alpha * instance_contrastive_loss(z1, z2)
            loss1 += instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                # loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
                loss2 += temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            # loss += alpha * instance_contrastive_loss(z1, z2)
            loss1 += instance_contrastive_loss(z1, z2)
        d += 1

    return loss1 / d, loss2 / d

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

def supervised_contrastive_loss(z1, z2, labels, temperature=0.07, alpha=0.5):
    # z1:B*T*C, z2:B*T*C
    B, T, C0= z1.size(0), z1.size(1), z1.size(2)
    # N, C = B, T*C0

    z1 = torch.mean(z1, dim=-1)
    z2 = torch.mean(z2, dim=-1)
    N, C = z1.size(0), z2.size(1)

    z1 = torch.reshape(z1,(B,-1)) # [B, T*C0]
    z2 = torch.reshape(z2,(B,-1)) # [B, T*C0]

    # Compute true positives from view
    local_pos = torch.matmul(torch.reshape(z1,(N,1,C)), torch.reshape(z2,(N,C,1))) # [N,1,1]
    local_pos = torch.reshape(local_pos, (N,1)) / temperature # [N,1]
    joint_expectation = local_pos.sum(dim=1) # [N,], 

    # Compute true normalization  计算相似性矩阵
    logits = torch.div(torch.matmul(z1, z2.transpose(0, 1)), temperature) # similarity_matrix [N,N]
    expectations_marginal_per_sample = torch.logsumexp(logits, dim=1)  # [N,]
    
    # Neighborhood terms
    # Neighborhood positives for aggregation
    neighbors_mask = torch.eq(labels.unsqueeze(1),labels.unsqueeze(0)).float()  # [N,N]
    number_neigh = neighbors_mask.sum(dim=1) # [N,]
    neighbors_expectation = (logits * neighbors_mask).sum(dim=1) / number_neigh
    aggregation_loss = (expectations_marginal_per_sample - neighbors_expectation).mean()
    
    # Neighborhood negatives for discrimination
    expectations_neighborhood_per_sample = torch.log(\
        torch.sum(torch.exp(logits) * neighbors_mask, dim=1))
    n_X_ent_per_sample = expectations_neighborhood_per_sample - joint_expectation
    disc_loss = n_X_ent_per_sample.mean()

    loss = alpha * aggregation_loss + (1.0 - alpha) * disc_loss

    # Computing the contrastive accuracy
    # preds = logits.argmax(dim=1)
    # local_labels = torch.arange(N)
    # correct_preds = (preds == local_labels)
    # accuracy = correct_preds.float().mean()

    # return loss, aggregation_loss, disc_loss, accuracy
    return loss, aggregation_loss, disc_loss
    # return loss


# 不确定损失函数
class UncertaintyLoss(nn.Module):

    def __init__(self, v_num, flag="RevisedUncertaintyLoss"): # flag:UncertaintyLoss,RevisedUncertaintyLoss,RestrainedRevisedUncertaintyLoss
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num
        self.flag = flag
        self.fi = 5

    def forward(self, *input):
        loss = 0
        if self.flag == "UncertaintyLoss":
            for i in range(self.v_num):
                loss += input[i] / (2 * self.sigma[i] ** 2)
            loss += torch.log(self.sigma.pow(2).prod())
            return loss
        elif self.flag == "RevisedUncertaintyLoss":
            for i in range(self.v_num):
                loss += input[i] / (2 * self.sigma[i] ** 2)
                loss += torch.log(torch.abs(1+torch.log(self.sigma[i].pow(2))))
            return loss
        elif self.flag == "RestrainedRevisedUncertaintyLoss":
            for i in range(self.v_num):
                loss += input[i] / (2 * self.sigma[i] ** 2)
                loss += torch.log(torch.abs(1+torch.log(self.sigma[i].pow(2))))
            self.fi = self.sigma[0] + self.sigma[1]
            loss += torch.abs(self.fi - (torch.abs(torch.log(torch.abs(self.sigma[0]))) + torch.abs(torch.log(torch.abs(self.sigma[1])))))
            return loss
        else:
            raise RuntimeError("Please input right flag!")

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

if __name__=="__main__":
    labels = torch.randint(0,2,size=(10,))
    z1 = torch.randint(0,2,size=(10,5,4))
    z2 = torch.randint(0,2,size=(10,5,4))
    supervised_contrastive_loss(z1,z2,labels)
    pass
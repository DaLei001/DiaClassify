from copy import deepcopy
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from learner import Learner, MLP

class MAML(nn.Module):
    """
    Meta Learner -- MAML
    """
    def __init__(self, args, config):
        super(MAML, self).__init__()
        self.update_lr = args.update_lr # task-level inner update learning rate，default=0.01
        self.meta_lr = args.meta_lr     # meta-level outer learning rate，default=1e-3
        self.n_label = args.n_label
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step           # task-level inner update steps
        self.update_step_test = args.update_step_test # update steps for finetunning
        self.n_attr = args.n_attr

        # 定义【学习器】和【优化器】
        self.net = Learner(config)
        # self.net = MLP(input_size=self.n_attr, hidden_size=100, output_size=self.n_label)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [batch, k_shot, attr]
        :param y_spt:   [batch, k_shot]
        :param x_qry:   [batch, k_query, attr]
        :param y_qry:   [batch, k_query]
        :return:
        """
        task_num, n_instance, n_feature = x_spt.size()
        querysz = x_qry.size(1)

        # update_step是更新步数，默认是5，但是loss和correct保存6个值，包括了网络参数赋初始值时，还没更新时的loss和correct
        losses_q = [0 for _ in range(self.update_step + 1)] # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        
        for i in range(task_num):

            # 1. 运行第 i 个任务，并计算 k=0 的损失
            logits = self.net(x_spt[i], vars=None, bn_training=True) # Todo
            loss = F.binary_cross_entropy_with_logits(logits, y_spt[i]) # 计算损失函数
            grad = torch.autograd.grad(loss, self.net.parameters()) # 计算梯度
            # fast_weights 其实就是用上次的模型参数θ0-lr*梯度，以求得当前的模型参数
            # 将grad作为p[0],self.net.parameters()作为p[1]，传入p[1] - self.update_lr * p[0]函数中，结果以list形式返回
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) # 内部快适应参数，

            # 此处是第一次网络参数更新之前的loss和accuracy，使用的是网络模型初始值参数
            with torch.no_grad():   # 在 torch.no_grad()模块下产生的tensor的requires_grad为False，反向传播时就不会自动求导了
                # [k_shot, n_label]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True) # Todo
                # logits_q = self.net(x_qry[i]) # Todo
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[0] += loss_q

                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = torch.sigmoid(logits_q)
                pred_q = torch.round(pred_q)
                correct = torch.eq(pred_q, y_qry[i]).sum().item() # torch.eq(a,b):对tensor a 和 b 逐元素比较是否相同，返回的tensor由True或False组成
                corrects[0] = corrects[0] + correct/self.n_label

            # 此处是第一次网络参数更新之后的loss和accuracy，使用的是网络模型更新一次之后的参数
            with torch.no_grad():
                # [k_shot, n_label]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True) # Todo
                # logits_q = self.net(x_qry[i]) # Todo
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [k_shot]
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = torch.sigmoid(logits_q)
                pred_q = torch.round(pred_q)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct/self.n_label

            for k in range(1, self.update_step):
                # 1. 运行第 i 个任务，并计算k=1~K-1的损失函数
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                # logits = self.net(x_spt[i])
                loss = F.binary_cross_entropy_with_logits(logits, y_spt[i])
                # 2. 计算 theta_pi 的梯度
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # logits_q = self.net(x_qry[i])
                # loss_q 将被复写，仅仅保留最新更新步的loss_q
                loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q


                with torch.no_grad():
                    # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    pred_q = torch.sigmoid(logits_q)
                    pred_q = torch.round(pred_q)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item() # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct/self.n_label


        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        # return accs
        return np.mean(accs)
    
    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [k_shot(setsz), attr]
        :param y_spt:   [k_shot(setsz)]
        :param x_qry:   [k_query(querysz), attr]
        :param y_qry:   [k_query(querysz)]
        :return:
        """
        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        logits = torch.sum(logits, dim=1).unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            pred_q = torch.sigmoid(logits_q)
            pred_q = torch.round(pred_q)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct/self.n_label

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            pred_q = torch.sigmoid(logits_q)
            pred_q = torch.round(pred_q)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct/self.n_label

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            logits = torch.sum(logits, dim=1).unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            logits_q = torch.sum(logits_q, dim=1).unsqueeze(1)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.binary_cross_entropy_with_logits(logits_q, y_qry)

            with torch.no_grad():
                # pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                pred_q = torch.sigmoid(logits_q)
                pred_q = torch.round(pred_q)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct/self.n_label


        del net

        accs = np.array(corrects) / querysz

        # return accs
        return np.mean(accs)


def main():
    pass

if __name__ == '__main__':
    main()
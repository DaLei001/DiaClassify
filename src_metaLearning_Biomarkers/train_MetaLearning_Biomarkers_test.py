#encoding=utf-8
# 利用元学习去训练biomarkers数据

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def samplePoints(k):
    #随机生成50个样本点
    x = np.random.rand(k,50)
    #各个元素的采样概率均为0.5
    y = np.random.choice([0,1],size=k,p=[.5,.5]).reshape([-1,1])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x = x.float()
    y = y.float()
    return x,y

def dataloader(split='train'):
    dataset = "Dia182_normProb_0.7_1"
    if dataset == "Dia182_normProb_0.7_1": # sub_60_sample_182
        df = pd.read_csv("../datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy2.csv",header=None).values
        # df = pd.read_csv("../datasets/Diabetes_v3/All_DM_sub_60_sample_182_classify_copy.csv",header=None).values
    # elif dataset == "Dia175_normProb_0.7_1":  # sub_110_sample_175
    #     df = pd.read_csv("../datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.7_classify_copy.csv",header=None).values
    elif dataset == "Dia182_normProb_0.3_1": # sub_60_sample_182
        df = pd.read_csv("../datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.3_classify_copy2.csv",header=None).values
    # elif dataset == "Dia175_normProb_0.3_1":  # sub_110_sample_175
    #     df = pd.read_csv("../datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.3_classify_copy.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,1:-22]

    y_df = df[:,0:1]
    more_y_df_total = df[:,-22:]
    more_y_df_raw = more_y_df_total[:,::2]    
    more_y_df_binary = more_y_df_total[:,1::2]
    more_y_df = more_y_df_total
    X_df_biomarkers = more_y_df_raw
    all_y_df = np.append(y_df,more_y_df,axis=1)

class MamlModel(nn.Module):
    # 此处使用的是最简单的回归任务，使用了一个单层的神经网络进行实现，没有特别复杂，便于理解原理
    def __init__(self, input_dim, out_dim, n_sample) -> None:
        super(MamlModel,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.W = nn.Parameter(torch.zeros(size=[input_dim, out_dim]))

    def forward(self):
        #生成用于训练的样本
        X_train,Y_train = samplePoints(self.n_sample)
        #单层网络的训练
        Y_predict =torch.matmul(X_train,self.W)
        Y_predict = Y_predict.reshape(-1,1)
        return Y_train,Y_predict

maml = MamlModel(50,1,10)
optimer = optim.Adam(maml.parameters(), lr=0.01, weight_decay=1e-5)
loss_function = nn.MSELoss()

'''
下面：定义一些超参数，包括迭代次数，任务数量，对于原始参数更新的学习率(对于每一个任务更新的学习率，我直接定义在
了优化器里面)，原始的向量
'''
epoches = 1000
tasks = 10
beta = 0.0001
theta_matrix = torch.zeros(size=[10,50,1])
theta_matrix = theta_matrix.float()
ori_theta = torch.randn(size=[50,1])
ori_theta = ori_theta.float()
meta_gradient = torch.zeros_like(ori_theta)

#下面定义训练过程
def train(epoch):
    # 对每一个任务进行迭代(训练),保留每一个任务梯度下降之后的参数
    global ori_theta, meta_gradient
    loss_sum = 0.0
    for i in range(tasks):
        maml.W.data = ori_theta.data
        optimer.zero_grad()
        Y_train, Y_predict = maml()
        loss_value = loss_function(Y_train, Y_predict)
        loss_sum = loss_sum + loss_value.data.item()
        loss_value.backward()
        optimer.step()
        # print(maml.W.shape)
        theta_matrix[i, :] = maml.W
    # 对每一个任务进行迭代（测试），利用保留的梯度下降之后的参数作为训练参数，计算梯度和
    for i in range(tasks):
        maml.W.data = theta_matrix[i]
        optimer.zero_grad()
        Y_test, Y_predict_test = maml()
        loss_value = loss_function(Y_test, Y_predict_test)
        loss_value.backward()
        optimer.step()
        meta_gradient = meta_gradient + maml.W
    # 更新初始的ori_theta
    ori_theta = ori_theta - beta * meta_gradient / tasks
    print("the Epoch is {:04d}".format(epoch),"the Loss is {:.4f}".format(loss_sum/tasks))


if __name__ == "__main__":
    # for epoch in range(epoches):
    #     train(epoch)
    dataloader()
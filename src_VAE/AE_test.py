import os
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
# import numpy as np
import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
from models.AE import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# from data import *
# from hotelling import stats,plots,helpers

import datautils
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载数据
isBioNormalized = True
dataset = "Dia182"
# train_data, train_bio_data, train_labels, test_data, test_bio_data, \
#     test_labels = datautils.load_Diabete_classification_v5(dataset, isBioNormalized)
train_data, train_bio_data, train_labels, val_data, val_bio_data, val_labels, test_data, test_bio_data, \
    test_labels = datautils.load_Diabete_classification_v5_with_val(dataset, isBioNormalized)

train_dataset = TensorDataset(torch.from_numpy(train_bio_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

val_dataset = TensorDataset(torch.from_numpy(val_bio_data).to(torch.float), torch.from_numpy(val_labels).to(torch.float))
valLoader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True)

test_dataset = TensorDataset(torch.from_numpy(test_bio_data).to(torch.float), torch.from_numpy(test_labels).to(torch.float))
testLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        

# 创建模型
# vae = VAE(784, 512, 256, 2)
# vae = VAE2(5, 4, 3, 2)
vae = VAE1(11, 2)
# vae = VAE1(5, 3, 1)

# 优化
# optimizer = torch.optim.SGD(vae.parameters(), lr=1e-4, weight_decay=5e-4)
# optimizer = torch.optim.SGD(vae.parameters(), lr=3e-3, weight_decay=5e-4)

# 损失函数
def lossFunction(recon_x, x, mu, log_var, y_1, label):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 11), reduction='mean')   # AE 的交叉熵损失函数
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())          # VAE 中的KL散度
    bce_y = F.binary_cross_entropy(y_1, label, reduction='mean')             # 潜在变量与质量变量的交叉熵函数
    # return 0.3 * bce + 0.2 * kld + 0.5 * bce_y, bce, kld, bce_y
    return 0.5 * bce + 0.5 * kld, bce, kld, bce_y

# 测试部分的训练损失函数，仅对分类标签进行finetune
def lossFunction_finetune(recon_x, x, mu, log_var, y_1, label):
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 11), reduction='mean')   # AE 的交叉熵损失函数
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())          # VAE 中的KL散度
    bce_y = F.binary_cross_entropy(y_1, label, reduction='mean')             # 潜在变量与质量变量的交叉熵函数
    # return 0.3 * bce + 0.2 * kld + 0.5 * bce_y, bce, kld, bce_y
    return bce_y, bce, kld, bce_y

# 训练函数
def train(epoch, optimizer):
    vae.train()
    for batch_idx, data_obtain in enumerate(trainLoader, 0):
        data, label = data_obtain
        recon_batch, mu, log_var, y_1 = vae(data)
        loss, bce, kld, bce_y = lossFunction(recon_batch, data, mu, log_var, y_1, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 128 == 0:
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(trainLoader.dataset),
                                                                          100. * batch_idx / len(trainLoader),
                                                                          loss.item() / len(data)))

# 测试函数
def test():
    vae.eval()
    testLoss = 0
    testbce = 0
    testkld = 0
    testbce_y = 0
    with torch.no_grad():
        for data_obtain in testLoader:
            data, label = data_obtain
            recon, mu, log_var, y_1  = vae(data)
            tloss, bce, kld, bce_y= lossFunction(recon, data, mu, log_var, y_1, label)
            testLoss += tloss.item()
            testbce += bce.item()
            testkld += kld.item()
            testbce_y += bce_y.item()
    testLoss /= len(testLoader.dataset)
    testbce /= len(testLoader.dataset)
    testkld /= len(testLoader.dataset)
    testbce_y /= len(testLoader.dataset)

    print('=====> Test set loss: {:.4f}  bce loss: {:.4f}  kld loss: {:.4f}  bce_y loss: {:.4f}'.format(testLoss,testbce,testkld,testbce_y))
    return testLoss, testbce, testkld, testbce_y

def fine_tune(model):
    model.train()
    print()
    print("Finetune Start!!")
    print()
    print()
    print()

    for batch_idx, data_obtain in enumerate(valLoader, 0):
        data, label = data_obtain
        recon_batch, mu, log_var, y_1 = model(data)
        loss, bce, kld, bce_y = lossFunction_finetune(recon_batch, data, mu, log_var, y_1, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(trainLoader.dataset),
                                                                          100. * batch_idx / len(trainLoader),
                                                                          loss.item() / len(data)))
    model.eval()
    acc = 0
    with torch.no_grad():
        for data_obtain in valLoader:
            data, label = data_obtain
            recon, mu, log_var, y_1  = model(data)
            tloss, bce, kld, bce_y= lossFunction_finetune(recon, data, mu, log_var, y_1, label)

            testLoss += tloss.item()
            testbce += bce.item()
            testkld += kld.item()
            testbce_y += bce_y.item()
    testLoss /= len(testLoader.dataset)
    testbce /= len(testLoader.dataset)
    testkld /= len(testLoader.dataset)
    testbce_y /= len(testLoader.dataset)

    print('=====> Test set loss: {:.4f}  bce loss: {:.4f}  kld loss: {:.4f}  bce_y loss: {:.4f}'.format(testLoss,testbce,testkld,testbce_y))
    return testLoss, testbce, testkld, testbce_y

# 训练
min_loss = 10000
num_epoch = 3000
num_epoch_finetune = 1000
testLosses = []
bces = []
klds = []
bce_ys = []

# 训练阶段
for epoch in range(num_epoch):
    if epoch < num_epoch * 0.25:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-4, weight_decay=5e-4)
    elif epoch < num_epoch * 0.5:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-5, weight_decay=5e-4)
    elif epoch < num_epoch * 0.75:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-6, weight_decay=5e-4)
    train(epoch, optimizer)
    testLoss, bce, kld, bce_y = test()
    testLosses.append(testLoss)
    bces.append(bce)
    klds.append(kld)
    bce_ys.append(bce_y)
    # 保存训练模型
    if testLoss < min_loss:
        min_loss = testLoss
        print("save model!!!!!!!")
        torch.save(vae.state_dict(), 'training/exp05-Dia182-Bio_VAE/model_2.pth')

# finetune 阶段
for epoch in range(num_epoch_finetune):
    if epoch < num_epoch * 0.25:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-4, weight_decay=5e-4)
    elif epoch < num_epoch * 0.5:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-5, weight_decay=5e-4)
    elif epoch < num_epoch * 0.75:
        optimizer = torch.optim.SGD(vae.parameters(), lr=1e-6, weight_decay=5e-4)

    model = VAE1(11,2)
    model.load_state_dict(torch.load('training/exp05-Dia182-Bio_VAE/model_2.pth'))
    model.eval()

    fine_tune(model)

    train(epoch, optimizer)
    testLoss, bce, kld, bce_y = test()
    testLosses.append(testLoss)
    bces.append(bce)
    klds.append(kld)
    bce_ys.append(bce_y)
    # 保存训练模型
    if testLoss < min_loss:
        min_loss = testLoss
        print("save model!!!!!!!")
        torch.save(vae.state_dict(), 'training/exp05-Dia182-Bio_VAE/model_2.pth')



# 测试过程损失函数图
fig = plt.figure(1)
ax1 = plt.subplot(2,2,1)
plt.plot(range(1,num_epoch+1),testLosses)
plt.xlabel('epoch')
plt.ylabel('testLosses')
ax2 = plt.subplot(2,2,2)
plt.plot(range(1,num_epoch+1),bces)
plt.xlabel('epoch')
plt.ylabel('bces')
ax3 = plt.subplot(2,2,3)
plt.plot(range(1,num_epoch+1),klds)
plt.xlabel('epoch')
plt.ylabel('klds')
ax4 = plt.subplot(2,2,4)
plt.plot(range(1,num_epoch+1),bce_ys)
plt.xlabel('epoch')
plt.ylabel('bce_ys')
plt.show()

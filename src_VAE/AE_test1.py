import math
import os
import aiofiles
import numpy as np 
import torch
from models.AE import *
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import datautils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# model = VAE1(11,6)
# model = VAE(11,8,8,5)
# model = VAE0(11,9,7,5)

# 加载数据
def data_loader():
    isBioNormalized = True
    # dataset = "Dia437"
    dataset = "Dia182"
    # train_data, train_bio_data, train_labels, test_data, test_bio_data, \
    #     test_labels = datautils.load_Diabete_classification_v5(dataset, isBioNormalized)
    train_data, train_bio_data, train_labels, val_data, val_bio_data, val_labels, test_data, test_bio_data, \
        test_labels = datautils.load_Diabete_classification_v5_with_val(dataset, isBioNormalized)
    # train_data, train_bio_data, train_labels, val_data, val_bio_data, val_labels, test_data, test_bio_data, \
    #     test_labels = datautils.load_Diabete_classification_v6(dataset, isBioNormalized)

    train_dataset = TensorDataset(torch.from_numpy(train_bio_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
    trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

    val_dataset = TensorDataset(torch.from_numpy(val_bio_data).to(torch.float), torch.from_numpy(val_labels).to(torch.float))
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(torch.from_numpy(test_bio_data).to(torch.float), torch.from_numpy(test_labels).to(torch.float))
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    return trainLoader, valLoader, testLoader

# 加载数据,使用的是paper2中的数据及分组
def data_loader_origin():
    isBioNormalized = True
    dataset = "Dia182"
    # train_data, train_bio_data, train_labels, test_data, test_bio_data, \
    #     test_labels = datautils.load_Diabete_classification_v5(dataset, isBioNormalized)
    train_data, train_bio_data, train_labels, val_data, val_bio_data, val_labels, test_data, test_bio_data, \
        test_labels = datautils.load_Diabete_classification_v5_with_val_origin(dataset, isBioNormalized)

    train_dataset = TensorDataset(torch.from_numpy(train_bio_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
    trainLoader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

    val_dataset = TensorDataset(torch.from_numpy(val_bio_data).to(torch.float), torch.from_numpy(val_labels).to(torch.float))
    valLoader = DataLoader(val_dataset, batch_size=2, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(torch.from_numpy(test_bio_data).to(torch.float), torch.from_numpy(test_labels).to(torch.float))
    testLoader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    return trainLoader, valLoader, testLoader

# 定义损失函数
def lossFunction(recon_x, x, mu, log_var, y_1, label, mode='train'):
    bce = F.binary_cross_entropy(recon_x, x, reduction='mean')   # AE 的交叉熵损失函数
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())          # VAE 中的KL散度
    bce_y = F.binary_cross_entropy(y_1, label, reduction='mean')             # 潜在变量与质量变量的交叉熵函数
    
    if mode == 'train':
        # 训练过程不用类别分类的标签
        # return kld + bce_y, bce, kld, bce_y
        # return 0.3 * bce + 0.2 * kld + 0.5 * bce_y, bce, kld, bce_y
        return bce + kld, bce, kld, bce_y 
        # return bce + kld + bce_y, bce, kld, bce_y
    elif mode == 'finetune':
        # 微调finetune阶段
        return bce + kld + bce_y, bce, kld, bce_y
    else:
        raise RuntimeError("The mode for loss function is wrong")
    
# 定义指标
def cal_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    # print("acc:",acc,"******* recall:",recall,"------ F1:",f1)
    return acc, recall, f1

def plotShow(num_epoch, testLosses, bces, klds, bce_ys, accs, recalls, f1s):
    # 测试过程损失函数图
    fig = plt.figure(1)

    ax1 = plt.subplot(3,3,1)
    plt.plot(range(1,num_epoch+1),testLosses)
    plt.xlabel('epoch')
    plt.ylabel('testLosses')

    ax2 = plt.subplot(3,3,2)
    plt.plot(range(1,num_epoch+1),bces)
    plt.xlabel('epoch')
    plt.ylabel('bces')

    ax3 = plt.subplot(3,3,3)
    plt.plot(range(1,num_epoch+1),klds)
    plt.xlabel('epoch')
    plt.ylabel('klds')

    ax4 = plt.subplot(3,3,4)
    plt.plot(range(1,num_epoch+1),bce_ys)
    plt.xlabel('epoch')
    plt.ylabel('bce_ys')

    ax5 = plt.subplot(3,3,5)
    plt.plot(range(1,num_epoch+1),accs)
    plt.xlabel('epoch')
    plt.ylabel('accs')

    ax6 = plt.subplot(3,3,6)
    plt.plot(range(1,num_epoch+1),recalls)
    plt.xlabel('epoch')
    plt.ylabel('recalls')

    ax7 = plt.subplot(3,3,7)
    plt.plot(range(1,num_epoch+1),f1s)
    plt.xlabel('epoch')
    plt.ylabel('f1s')


    plt.show()

def plotShow1(num_epoch, train_losses, valid_losses):
    # 测试过程损失函数图
    fig, ax = plt.subplots()
    ax.plot(range(1,num_epoch+1), train_losses, label='train')
    ax.plot(range(1,num_epoch+1), valid_losses, label='validation')
    ax.legend()
    plt.show()

def train(trainLoader, model, epoch, optimizer, flag='train'):
    model.train()
    for batch_idx, data_obtain in enumerate(trainLoader, 0):
        data, label = data_obtain
        recon_batch, mu, log_var, y_1 = model(data)
        loss, bce, kld, bce_y = lossFunction(recon_batch, data, mu, log_var, y_1, label, mode=flag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 128 == 0:
        if batch_idx % 100 == 0 and flag=='train':
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(trainLoader.dataset),
                                                                          100. * batch_idx / len(trainLoader),
                                                                          loss.item() / len(data)))
        elif batch_idx % 10 == 0 and flag=='finetune':
            print('Finetune Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(trainLoader.dataset),
                                                                          100. * batch_idx / len(trainLoader),
                                                                          loss.item() / len(data)))

def test(testLoader):
    model.eval()
    testLoss = 0
    testbce = 0
    testkld = 0
    testbce_y = 0
    accs = 0
    recalls = 0
    f1s = 0
    y_1s = []
    labels = []
    with torch.no_grad():
        for data_obtain in testLoader:
            data, label = data_obtain
            recon, mu, log_var, y_1  = model(data)
            tloss, bce, kld, bce_y= lossFunction(recon, data, mu, log_var, y_1, label, mode='train')
            # acc, recall, f1 = cal_metrics(label, y_1.round())
            # accs += acc.item()
            # recalls += recall.item()
            # f1s += f1.item()
            y_1s.extend(y_1.round().numpy())
            labels.extend(label.numpy())
            testLoss += tloss.item()
            testbce += bce.item()
            testkld += kld.item()
            testbce_y += bce_y.item()
    testLoss /= len(testLoader.dataset)
    testbce /= len(testLoader.dataset)
    testkld /= len(testLoader.dataset)
    testbce_y /= len(testLoader.dataset)
    labels = np.squeeze(labels)
    y_1s = np.squeeze(y_1s)
    # accs, recalls, f1s = cal_metrics(labels, y_1s)
    [accs, pre, recalls, npv, spe, f1s, mcc, g_mean] = evaluate(labels,y_1s)
    # accs /= len(testLoader.dataset)
    # recalls /= len(testLoader.dataset)
    # f1s /= len(testLoader.dataset)

    print('=====> Test set loss: {:.4f}  bce loss: {:.4f}  kld loss: {:.4f}  bce_y loss: {:.4f}  acc: {:.4f}  recall: {:.4f}  f1: {:.4f}'.format(testLoss,testbce,testkld,testbce_y,accs, recalls, f1s))
    return testLoss, testbce, testkld, testbce_y, accs, recalls, f1s

def evaluate(true, pred):

    C2 = metrics.confusion_matrix(true, pred, labels=[0,1])
    TP = C2[0][0]
    FP = C2[1][0]
    TN = C2[1][1]
    FN = C2[0][1]

    acc = (TP+TN) / (TP+FP+TN+FN)
    pre = TP / (TP+FP)
    rec = TP / (TP+FN)
    npv = TN / (TN+FN)
    spe = TN / (TN+FP)

    f1 = (2*pre*rec) / (pre+rec)
    mcc = ((TP*TN)-(FP*FN)) / (math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    g_mean = math.sqrt(rec * spe)

    res = [acc, pre, rec, npv, spe, f1, mcc, g_mean]
    return res

def run():
    # 参数定义
    min_loss = 10000
    num_epoch_train = 300
    num_epoch_finetune = 300
    testLosses = []
    bces = []
    klds = []
    bce_ys = []
    accs = []
    recalls = []
    f1s = []

    # 数据导入
    trainLoader, valLoader, testLoader = data_loader()

    # 模型定义
    # global model
    # model = VAE1(11,2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # 训练阶段
    for epoch in range(num_epoch_train):
        if epoch < num_epoch_train * 0.25:
            # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif epoch < num_epoch_train * 0.5:
            # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=5e-4)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        elif epoch < num_epoch_train * 0.75:
            # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=5e-4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        
        # train(trainLoader, epoch, optimizer, flag='train')
        flag='train'
        model.train()
        for batch_idx, data_obtain in enumerate(trainLoader, 0):
            data, label = data_obtain
            recon_batch, mu, log_var, y_1 = model(data)
            loss, bce, kld, bce_y = lossFunction(recon_batch, data, mu, log_var, y_1, label, mode=flag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch_idx % 128 == 0:
            if batch_idx % 100 == 0 and flag=='train':
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(trainLoader.dataset),
                                                                            100. * batch_idx / len(trainLoader),
                                                                            loss.item() / len(data)))
            

        # 训练阶段的测试集测试
        testLoss, bce, kld, bce_y, acc, recall, f1 = test(testLoader)

        testLosses.append(testLoss)
        bces.append(bce)
        klds.append(kld)
        bce_ys.append(bce_y)
        accs.append(acc)
        recalls.append(recall)
        f1s.append(f1)
        # 保存训练模型
        if testLoss < min_loss:
            min_loss = testLoss
            print("save model!!!!!!!")
            torch.save(model.state_dict(), 'training/exp05-Dia182-Bio_VAE/model_2.pth')

    # 画训练阶段的测试集测试损失函数图
    plotShow(num_epoch_train, testLosses, bces, klds, bce_ys, accs, recalls, f1s)

    testLosses = []
    bces = []
    klds = []
    bce_ys = []
    accs = []
    recalls = []
    f1s = []

    # 微调阶段finetune
    for epoch in range(num_epoch_finetune):
        if epoch < num_epoch_finetune * 0.25:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif epoch < num_epoch_finetune * 0.5:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        elif epoch < num_epoch_finetune * 0.75:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        # model.load_state_dict(torch.load('training/exp05-Dia182-Bio_VAE/model_2.pth'))
        # train(valLoader, epoch, optimizer, flag='finetune')
        flag='finetune'
        model.train()
        for batch_idx, data_obtain in enumerate(trainLoader, 0):
            data, label = data_obtain
            recon_batch, mu, log_var, y_1 = model(data)
            loss, bce, kld, bce_y = lossFunction(recon_batch, data, mu, log_var, y_1, label, mode=flag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch_idx % 128 == 0:
            if batch_idx % 100 == 0 and flag=='train':
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(trainLoader.dataset),
                                                                            100. * batch_idx / len(trainLoader),
                                                                            loss.item() / len(data)))
            elif batch_idx % 10 == 0 and flag=='finetune':
                print('Finetune Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(trainLoader.dataset),
                                                                            100. * batch_idx / len(trainLoader),
                                                                            loss.item() / len(data)))
        


        # 训练阶段的测试集测试
        if epoch % 10:
            testLoss, bce, kld, bce_y, acc, recall, f1 = test(testLoader)
        # testLosses.append(testLoss)
        # bces.append(bce)
        # klds.append(kld)
        # bce_ys.append(bce_y)
        # accs.append(acc)
        # recalls.append(recall)
        # f1s.append(f1)

    # 画训练阶段的测试集测试损失函数图
    # plotShow(num_epoch_finetune, testLosses, bces, klds, bce_ys, accs, recalls, f1s)


    pass

def run1():
    epoch_train = 300

    trainLoader, valLoader, testLoader = data_loader()
    model = VAE(11,8,8,5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ## 训练阶段
    best_loss = 1e9
    train_losses = []
    valid_losses = []
    
    testLosses = []
    bces = []
    klds = []
    bce_ys = []
    accs = []
    recalls = []
    f1s = []
    
    for epoch in range(epoch_train):
        model.train()
        train_loss = 0.
        train_num = len(trainLoader.dataset)
        
        # 训练
        for batch_idx, data_obtain in enumerate(trainLoader, 0):
            data, label = data_obtain
            recon_batch, mu, log_var, y_1 = model(data)
            loss, bce, kld, bce_y = lossFunction(recon_batch, data, mu, log_var, y_1, label,mode='train')
            
            train_loss += loss.item()
            
            loss = loss / len(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出记录值
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(trainLoader.dataset),
                                                                            100. * batch_idx / len(trainLoader),
                                                                            loss.item() / len(data)))
        
        train_losses.append(train_loss / train_num) # 记录每个epoch的训练损失函数值
        
        # 验证
        model.eval()
        testLoss, testbce, testkld, testbce_y = 0., 0., 0., 0.
        accs, recalls, f1s = 0., 0., 0.
        valid_num = len(testLoader.dataset)
        with torch.no_grad():
            for data_obtain in testLoader:
                data, label = data_obtain
                recon, mu, log_var, y_1  = model(data)
                tloss, bce, kld, bce_y= lossFunction(recon, data, mu, log_var, y_1, label, mode='train')

                testLoss +=tloss.item()
                testbce += bce.item()
                testkld += kld.item()

            valid_losses.append(testLoss / valid_num)


            # print('=====> Test set loss: {:.4f}  bce loss: {:.4f}  kld loss: {:.4f}  bce_y loss: {:.4f}  acc: {:.4f}  recall: {:.4f}  f1: {:.4f}'.format(testLoss,testbce,testkld,testbce_y,accs, recalls, f1s))
            print('=====> Test set loss: {:.4f}  bce loss: {:.4f}  kld loss: {:.4f} '.format(testLoss,testbce,testkld))
    

            # 保存训练模型
            if testLoss < best_loss:
                best_loss = testLoss
                print("save model!!!!!!!")
                torch.save(model.state_dict(), 'training/exp05-Dia182-Bio_VAE/best_model.pth')

    # 画训练阶段的测试集测试损失函数图
    plotShow1(epoch_train, train_losses ,valid_losses)


            



    # 微调阶段




if __name__ == "__main__":

    # run()
    run1()
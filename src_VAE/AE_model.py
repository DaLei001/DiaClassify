from models.AE import *
import torch.nn.functional as F

class VAE_model:
    '''VAE model'''

    def __init__(self):

        super().__init__()
        self.vae = VAE1(11,2)
        pass

    def fit(self, trainLoader):
        # 训练
        min_loss = 10000
        num_epoch = 3000
        testLosses = []
        bces = []
        klds = []
        bce_ys = []
        for epoch in range(num_epoch):
            if epoch < num_epoch * 0.25:
                optimizer = torch.optim.SGD(self.vae.parameters(), lr=1e-4, weight_decay=5e-4)
            elif epoch < num_epoch * 0.5:
                optimizer = torch.optim.SGD(self.vae.parameters(), lr=1e-5, weight_decay=5e-4)
            elif epoch < num_epoch * 0.75:
                optimizer = torch.optim.SGD(self.vae.parameters(), lr=1e-6, weight_decay=5e-4)
            
            self.train(trainLoader, epoch, optimizer)
            
            testLoss, bce, kld, bce_y = test()
            testLosses.append(testLoss)
            bces.append(bce)
            klds.append(kld)
            bce_ys.append(bce_y)
            # 保存训练模型
            if testLoss < min_loss:
                min_loss = testLoss
                print("save model!!!!!!!")
                torch.save(self.vae.state_dict(), 'training/exp05-Dia182-Bio_VAE/model_2.pth')

        pass

    def encode(self):


        pass

    def train(self, trainLoader, epoch, optimizer):
        self.vae.train()
        for batch_idx, data_obtain in enumerate(trainLoader, 0):
            data, label = data_obtain
            recon_batch, mu, log_var, y_1 = self.vae(data)
            loss, bce, kld, bce_y = self.lossFunction(recon_batch, data, mu, log_var, y_1, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 128 == 0:
                print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(trainLoader.dataset),
                                                                            100. * batch_idx / len(trainLoader),
                                                                            loss.item() / len(data)))
        pass


    def lossFunction(recon_x, x, mu, log_var, y_1, label):
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 11), reduction='mean')   # AE 的交叉熵损失函数
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())          # VAE 中的KL散度
        bce_y = F.binary_cross_entropy(y_1, label, reduction='mean')             # 潜在变量与质量变量的交叉熵函数
        # return 0.3 * bce + 0.2 * kld + 0.5 * bce_y, bce, kld, bce_y
        return 0.5 * bce + 0.5 * kld, bce, kld, bce_y
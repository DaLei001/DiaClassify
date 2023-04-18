# 实现自动编码器
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(x_dim, h_dim1)
        # self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # 解码器
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        # self.fc_y = nn.Linear(x_dim, 1)
        self.fc_y = nn.Linear(z_dim, 1)
    # 编码器
    def encoder(self, x):
        h = nn.ReLU()(self.fc1(x))
        # h = nn.ReLU()(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    # 换算z的公式
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 生成std形状大小的随机标准正态分布的噪声
        return eps.mul(std).add_(mu) # 得到z
    # 解码器
    def decoder(self, z):
        h = nn.ReLU()(self.fc4(z))
        h = nn.ReLU()(self.fc5(h))
        return nn.Sigmoid()(self.fc6(h))
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        y_1 = nn.Sigmoid()(self.fc_y(z))
        # y_1 = nn.Sigmoid()(self.fc_y(mu))
        return self.decoder(z), mu, log_var, y_1

class VAE0(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE0, self).__init__()
        # 编码器
        self.conv1 = nn.Conv1d(x_dim, h_dim1, 1, 1, 0)
        self.BN1 = nn.BatchNorm1d(h_dim1)
        self.fc1 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # 解码器
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.conv2 = nn.Conv1d(h_dim1, x_dim, 1, 1, 0)
        self.BN2 = nn.BatchNorm1d(x_dim)
        # self.fc6 = nn.Linear(h_dim1, x_dim)

        # self.fc_y = nn.Linear(x_dim, 1)
        self.fc_y = nn.Linear(z_dim, 1)
    # 编码器
    def encoder(self, x):
        h = self.BN1(self.conv1(x))
        h = h.view(-1,9)
        h = nn.ReLU()(self.fc1(h))
        return self.fc31(h), self.fc32(h)
    # 换算z的公式
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 生成std形状大小的随机标准正态分布的噪声
        return eps.mul(std).add_(mu) # 得到z
    # 解码器
    def decoder(self, z):
        h = nn.ReLU()(self.fc4(z))
        h = nn.ReLU()(self.fc5(h))
        h = h.reshape((-1,9,1))
        h = self.BN2(self.conv2(h))
        # return nn.Sigmoid()(self.fc6(h))
        return nn.Sigmoid()(h)
        # return self.BN2(self.conv2(h))
    def forward(self, x):
        # x = torch.permute(x,(0,2,1))
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        y_1 = nn.Sigmoid()(self.fc_y(z))
        # y_1 = nn.Sigmoid()(self.fc_y(mu))
        return self.decoder(z), mu, log_var, y_1

class VAE1(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(VAE1, self).__init__()
        # 编码器
        self.fc11 = nn.Linear(x_dim, z_dim)
        self.fc12 = nn.Linear(x_dim, z_dim)

        # 解码器
        self.fc4 = nn.Linear(z_dim, x_dim)
        # self.fc6 = nn.Linear(h_dim1, x_dim)

        self.fc_y = nn.Linear(z_dim, 1)

    # 编码器
    def encoder(self, x):
        h1 = nn.SELU()(self.fc11(x))
        h2 = nn.SELU()(self.fc12(x))
        return h1, h2
    # 换算z的公式
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 生成std形状大小的随机标准正态分布的噪声
        return eps.mul(std).add_(mu) # 得到z

    # 解码器
    def decoder(self, z):
        return nn.Sigmoid()(self.fc4(z))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 11))
        z = self.sampling(mu, log_var)
        y_1 = nn.Sigmoid()(self.fc_y(z))
        return self.decoder(z), mu, log_var, y_1

class VAE2(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE2, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # 解码器
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        self.fc_y = nn.Linear(z_dim, 1)
    # 编码器
    def encoder(self, x):
        h = nn.ReLU()(self.fc1(x))
        h = nn.ReLU()(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    # 换算z的公式
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 生成std形状大小的随机标准正态分布的噪声
        return eps.mul(std).add_(mu) # 得到z
    # 解码器
    def decoder(self, z):
        h = nn.ReLU()(self.fc4(z))
        h = nn.ReLU()(self.fc5(h))
        return nn.Sigmoid()(self.fc6(h))
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 5))
        z = self.sampling(mu, log_var)
        y_1 = nn.Sigmoid()(self.fc_y(z))
        return self.decoder(z), mu, log_var, y_1

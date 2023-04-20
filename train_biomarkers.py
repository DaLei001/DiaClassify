import numpy as np
from sklearn import svm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import datautils
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def MI_based_feature_subset_selection(train_bio_data, train_labels,val_bio_data, val_labels,test_bio_data, test_labels):
    model1 = SelectKBest(mutual_info_classif,k=2)
    model1.fit(train_bio_data,train_labels)
    X_new = model1.transform(train_bio_data)
    # X_new = model1.transform(val_bio_data)
    X_new1 = model1.transform(test_bio_data)

    model = svm.SVC()
    model.fit(train_bio_data,train_labels)
    # model.fit(val_bio_data,val_labels)
    score1 = model.score(test_bio_data, test_labels)

    model.fit(X_new, train_labels)
    # model.fit(X_new, val_labels)
    score2 = model.score(X_new1, test_labels)
    
    return score1, score2

# 重新定义自编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 11, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 11)
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 64 * 11)
        self.conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose1d(32, input_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 11)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# 定义变分自编码器
class VAE1(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 5, 512),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(512, 20)
        self.fc2 = nn.Linear(512, 20)
        self.decoder = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 5),
            nn.ReLU(),
            nn.Unflatten(1, (256, 5)),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward1(self, x):
        x = x.view(-1, self.input_dim)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc1(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 定义损失函数
def loss_function(x_recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_recon, x.view(-1, input_dim, x.size(1)), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# 训练模型
def train(model, dataloader, optimizer, device):
    model.train()
    train_loss, train_BCE_loss, train_KLD_loss = 0, 0, 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        loss, BCE, KLD = loss_function(x_recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_BCE_loss += BCE.item()
        train_KLD_loss += KLD.item()
        optimizer.step()
    avg_loss = train_loss / len(dataloader.dataset)
    avg_BCE = train_BCE_loss / len(dataloader.dataset)
    avg_KLD = train_KLD_loss / len(dataloader.dataset)
    return avg_loss, avg_BCE, avg_KLD

# 测试模型
def test(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            inputs = data.view(data.size(0), -1).to(device)
            _, mu, _ = model(inputs)
            features.append(mu.cpu().numpy())

    return np.squeeze(features)

if __name__ == '__main__':
    task_type = 'classification'
    isBioNormalized = True
    train_data, train_bio_data, train_labels,\
            val_data, val_bio_data, val_labels,\
            test_data, test_bio_data, test_labels\
            = datautils.load_Diabete_classification_v6("Dia437", isBioNormalized)
    
    # # 针对Biomarker中11项指标，基于互信息值的特征子集选择（FSS）
    # # score1, score2分别是有无FSS的SVM结果，0.75-->0.96
    # score1, score2 = MI_based_feature_subset_selection(train_bio_data, train_labels,\
    #                                                    val_bio_data, val_labels,test_bio_data, test_labels)        

    # 设置超参数
    MI_K = 11
    input_dim = 1
    hidden_dim = 400
    latent_dim = 20
    lr = 1e-3
    epochs = 300
    batch_size = 8


    model1 = SelectKBest(mutual_info_classif,k=MI_K)
    model1.fit(train_bio_data,train_labels)
    train_bio_data = model1.transform(train_bio_data)
    val_bio_data = model1.transform(val_bio_data)
    test_bio_data = model1.transform(test_bio_data) 
    
    train_dataset = TensorDataset(torch.from_numpy(train_bio_data).to(torch.float),torch.from_numpy(train_labels).to(torch.float))
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=True)
    val_dataset = TensorDataset(torch.from_numpy(val_bio_data).to(torch.float),torch.from_numpy(val_labels).to(torch.float))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataset = TensorDataset(torch.from_numpy(test_bio_data).to(torch.float),torch.from_numpy(test_labels).to(torch.float))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    # model = VAE(input_dim, hidden_dim, latent_dim)
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        avg_loss, avg_BCE, avg_KLD = train(model, train_loader, optimizer, device)
        print('Epoch [{}/{}], Average Loss: {:.2f}, Average BCE Loss: {:.2f}, Average KLD Loss: {:.2f}'.format(epoch+1, epochs, avg_loss, avg_BCE, avg_KLD))

    # 测试模型
    features_val = test(model, val_loader, device)
    features_test = test(model, test_loader, device)

    svm_model = svm.SVC()
    svm_model.fit(val_bio_data, val_labels)
    score1 = svm_model.score(test_bio_data, test_labels)
    
    svm_model.fit(features_val, val_labels)
    score2 = svm_model.score(features_test, test_labels)
    pass

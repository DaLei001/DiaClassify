from collections import OrderedDict
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# import torchvision

class TransformerModel1(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, num_heads=8, hidden_size=512):
        super(TransformerModel1, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

class LinearEncoder(nn.Module):
    def __init__(self, input_dims, output_dims) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.l1 = nn.Linear(input_dims, 64)
        self.l2 = nn.Linear(64, output_dims)
        
    def forward(self, x, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        return x

class LstmEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=32, num_layers=1) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        # batch_first – 如果为True，那么输入和输出Tensor的形状为(batch, seq, feature)
        # bidirectional – 如果为True，将会变成一个双向RNN，默认为False
        self.lstm = nn.LSTM(input_dims, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(hidden_size, output_dims)

        
    def forward(self, x, mask=None):
        x, _ = self.lstm(x)
        x = self.linear1(x)
        return x

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class CnnLstmEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=64, num_layers=1, mask_mode='binomial') -> None:
        super().__init__()
        self.mask_mode = mask_mode
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.relu = nn.ReLU(inplace=True)
        # 要确保输入和输出的维度一致，设卷积核大小为F*F，滑动步长为S，填充为P
        # 输出的维度为W2 = [(W1-F+2*P)/S]+1
        self.conv1 = nn.Conv1d(in_channels=input_dims, out_channels=2*hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=2*hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=32, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dims, out_channels=hidden_size, kernel_size=1)
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=1, stride=1)
        )

        # batch_first – 如果为True，那么输入和输出Tensor的形状为(batch, seq, feature)
        # bidirectional – 如果为True，将会变成一个双向RNN，默认为False
        self.lstm = nn.LSTM(32, 32, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(32, output_dims)

    def mask_fun(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        return x

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.mask_fun(x)
        x = self.conv2(x)
        x = self.mask_fun(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)

        x = self.mask_fun(x)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        # x = x[:, -1, :]
        return x

class SLNet(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=8) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_size = hidden_size
        # self.net = torchvision.models.resnet18(pretrained=True)
        # self.net = nn.Sequential(OrderedDict(
        #     [('conv1', nn.Conv1d(in_channels=input_dims, out_channels=hidden_size, kernel_size=3, padding=1)),
        #     ('sigmoid1', nn.Sigmoid()),
        #     ('maxpool1d1', nn.MaxPool1d(2,2)),
        #     ('conv2', nn.Conv1d(in_channels=hidden_size, out_channels=2*hidden_size, kernel_size=3, padding=1)),
        #     ('sigmoid2', nn.Sigmoid()),
        #     ('maxpool1d2', nn.MaxPool1d(2,2))]))
        self.net = nn.Sequential(OrderedDict(
            [('conv1', nn.Conv1d(in_channels=input_dims, out_channels=hidden_size, kernel_size=3, padding=1))]))
        # self.n_features = self.net.fc.in_features
        
        self.net_x1 = nn.Sequential(OrderedDict(
            [('conv1', nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1)),
            ('relu', nn.ReLU()),
            ('maxpool1d1', nn.MaxPool1d(4,4))]))
        self.net_x2 = nn.Sequential(OrderedDict(
            [('conv1', nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1)),
            ('relu', nn.ReLU()),
            ('maxpool1d1', nn.MaxPool1d(4,4))]))
        self.GRU_x1 = nn.GRU(input_size=3,hidden_size=50,num_layers=4,batch_first=True,dropout=0,bidirectional=False)
        self.GRU_x2 = nn.GRU(input_size=3,hidden_size=50,num_layers=4,batch_first=True,dropout=0,bidirectional=False)
        self.n_features = hidden_size
        self.net.fc = nn.Identity()

        # self.net.fc1 = nn.Sequential(OrderedDict(
        #     # [('linear', nn.Linear(self.n_features,self.n_features)),
        #     [('linear', nn.Linear(7150,self.n_features)),
        #     ('relu1', nn.ReLU()),
        #     ('final', nn.Linear(self.n_features, 1))]))
 
        # self.net.fc2 = nn.Sequential(OrderedDict(
        #     [('linear', nn.Linear(7150,self.n_features)),
        #     ('relu1', nn.ReLU()),
        #     ('final', nn.Linear(self.n_features, 1))]))

        self.net.fc1 = nn.Sequential(OrderedDict(
            # [('linear', nn.Linear(self.n_features,self.n_features)),
            [('linear', nn.Linear(7150,1))]))
 
        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(7150,1))]))
        
         
    def forward(self, x):
        # x = x.transpose(1,2)
        # x1 = self.net(x)
        x = x.permute(0,2,1)  # [8, 576, 1] --> [8, 1, 576]
        # x1 = nn.Conv1d(in_channels=1,out_channels=self.hidden_size,kernel_size=3,stride=1)(x) #[8,8,574]
        # x1 = nn.Conv1d(in_channels=1,out_channels=3,kernel_size=3,stride=1)(x) #[8,8,574]
        # x1 = nn.ReLU()(x1)  # [8, 3, 574]
        # x1 = nn.MaxPool1d(4,4)(x1) # [8, 3, 143]
        x1 = self.net_x1(x)
        x1 = x1.permute(0,2,1) # [8, 287, 8]
        # x1 = nn.Conv1d(in_channels=self.hidden_size,out_channels=2*self.hidden_size,kernel_size=3,stride=1)(x1)
        # x1 = nn.ReLU()(x1)
        # x1 = nn.MaxPool1d(2,2)(x1)
        x1,hn1 = self.GRU_x1(x1) # [8, 143, 50]

        # x2 = nn.Conv1d(in_channels=1,out_channels=self.hidden_size,kernel_size=3,stride=1)(x)
        # x2 = nn.Conv1d(in_channels=1,out_channels=3,kernel_size=3,stride=1)(x)
        # x2 = nn.ReLU()(x2)
        # x2 = nn.MaxPool1d(4,4)(x2)
        x2 = self.net_x2(x)
        x2 = x2.permute(0,2,1)
        # x2 = nn.Conv1d(in_channels=self.hidden_size,out_channels=2*self.hidden_size,kernel_size=3,stride=1)(x2)
        # x2 = nn.ReLU()(x2)
        # x2 = nn.MaxPool1d(2,2)(x2)
        x2,hn2 = self.GRU_x2(x2)
        
        # x1 = x1.reshape(x1.size(0),-1)
        # x2 = x2.reshape(x2.size(0),-1)
        x1 = nn.Flatten()(x1)
        x2 = nn.Flatten()(x2)
        
        # hypertension_head = self.net.fc1(x1)  # 高血压输出头
        # retinopathy_head = self.net.fc2(x2)    # 视网膜病变输出头
        hypertension_head = self.net.fc1(x1)  # 高血压输出头
        # hypertension_head = nn.Sigmoid()(hypertension_head.view(hypertension_head.size(0),-1))
        retinopathy_head = self.net.fc2(x2)    # 视网膜病变输出头
        # retinopathy_head = nn.Sigmoid()(retinopathy_head.view(retinopathy_head.size(0),-1))
        return hypertension_head, retinopathy_head

class SLNet_0(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=256) -> None:
        super().__init__()
        self.input_dims = input_dims
        # self.output_dims = output_dims
        self.output_dims = 1
        self.hidden_size = hidden_size

        self.conv1d_x1 = nn.Conv1d(in_channels=self.input_dims,out_channels=3,kernel_size=3,padding=0)
        self.conv1d_x2 = nn.Conv1d(in_channels=self.input_dims,out_channels=3,kernel_size=3,padding=0)
        self.BN1 = nn.BatchNorm1d(3)
        self.BN2 = nn.BatchNorm1d(3)

        self.GRU_x1 = nn.GRU(input_size=3,hidden_size=100,num_layers=1,batch_first=True,dropout=0.5,bidirectional=True)
        self.GRU_x2 = nn.GRU(input_size=3,hidden_size=100,num_layers=1,batch_first=True,dropout=0.5,bidirectional=True)
        
        # self.GRU_x1 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)
        # self.GRU_x2 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)

        # self.x1_fc1 = nn.Linear(576, self.hidden_size)
        self.x1_fc1 = nn.Linear(114800, self.hidden_size)
        self.x1_fc2 = nn.Linear(self.hidden_size, self.output_dims)
        # self.x2_fc1 = nn.Linear(576, self.hidden_size)
        self.x2_fc1 = nn.Linear(114800, self.hidden_size)
        self.x2_fc2 = nn.Linear(self.hidden_size, self.output_dims)
         
    def forward(self, x):
        # x1,hn1 = self.GRU_x1(x)
        # x2,hn1 = self.GRU_x2(x)
        
        x1 = x.permute(0,2,1)
        x2 = x.permute(0,2,1)

        x1 = self.BN1(F.relu(self.conv1d_x1(x1)))
        x2 = self.BN2(F.relu(self.conv1d_x2(x2)))

        x1 = x1.permute(0,2,1)
        x2 = x2.permute(0,2,1)

        x1,hn1 = self.GRU_x1(x1)
        x2,hn1 = self.GRU_x2(x2)

        x1 = nn.Flatten()(x1)
        x2 = nn.Flatten()(x2)
        
        x1 = self.x1_fc1(x1)
        # x1 = self.x1_fc2(F.dropout(x1,0.5))
        x1 = self.x1_fc2(x1)
        x2 = self.x2_fc1(x2)
        # x2 = self.x2_fc2(F.dropout(x2,0.5))
        x2 = self.x2_fc2(x2)

        return x1, x2

class SLNet_1(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=256) -> None:
        super().__init__()
        self.input_dims = input_dims
        # self.output_dims = output_dims
        self.output_dims = 1
        self.hidden_size = hidden_size
        
        # self.GRU_x1 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)
        # self.GRU_x2 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)

        self.x1_fc1 = nn.Linear(576, self.hidden_size)
        # self.x1_fc1 = nn.Linear(23040, self.hidden_size)
        self.x1_fc2 = nn.Linear(self.hidden_size, self.output_dims)
        self.x2_fc1 = nn.Linear(576, self.hidden_size)
        # self.x2_fc1 = nn.Linear(23040, self.hidden_size)
        self.x2_fc2 = nn.Linear(self.hidden_size, self.output_dims)
         
    def forward(self, x):
        # x1,hn1 = self.GRU_x1(x)
        # x2,hn1 = self.GRU_x2(x)
        
        x1 = x
        x2 = x

        x1 = nn.Flatten()(x1)
        x2 = nn.Flatten()(x2)
        
        x1 = F.relu(self.x1_fc1(x1))
        x1 = self.x1_fc2(F.dropout(x1,0.1))
        # x1 = self.x1_fc2(x1)
        x2 = F.relu(self.x2_fc1(x2))
        x2 = self.x2_fc2(F.dropout(x2,0.1))
        # x2 = self.x2_fc2(x2)

        return x1, x2
    
class SLNet_2_0(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_size=256) -> None:
        super().__init__()
        self.input_dims = input_dims
        # self.output_dims = output_dims
        self.output_dims = 1
        self.hidden_size = hidden_size
        
        # self.GRU_x1 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)
        # self.GRU_x2 = nn.GRU(input_size=self.input_dims,hidden_size=20,num_layers=1,batch_first=True,dropout=0.2,bidirectional=True)

        self.x1_fc1 = nn.Linear(576, self.hidden_size)
        # self.x1_fc1 = nn.Linear(23040, self.hidden_size)
        self.x1_fc2 = nn.Linear(self.hidden_size, self.output_dims)
        self.x2_fc1 = nn.Linear(576, self.hidden_size)
        # self.x2_fc1 = nn.Linear(23040, self.hidden_size)
        self.x2_fc2 = nn.Linear(self.hidden_size, self.output_dims)
         
    def forward(self, x):
        # x1,hn1 = self.GRU_x1(x)
        # x2,hn1 = self.GRU_x2(x)
        
        x1 = x
        x2 = x

        x1 = nn.Flatten()(x1)
        x2 = nn.Flatten()(x2)
        
        x1 = F.relu(self.x1_fc1(x1))
        x1 = self.x1_fc2(F.dropout(x1,0.1))
        # x1 = self.x1_fc2(x1)
        x2 = F.relu(self.x2_fc1(x2))
        x2 = self.x2_fc2(F.dropout(x2,0.1))
        # x2 = self.x2_fc2(x2)

        return x1, x2

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        # embed_dim = head_dim * num_heads?
        self.input_fc = nn.Linear(args.input_size, args.d_model)
        self.output_fc = nn.Linear(args.input_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=8,
            dim_feedforward=4 * args.d_model,
            batch_first=True,
            dropout=0.1,
            device="cpu"
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=8,
            dropout=0.1,
            dim_feedforward=4 * args.d_model,
            batch_first=True,
            device="cpu"
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=5)
        self.fc = nn.Linear(args.output_size * args.d_model, args.output_size)
        self.fc1 = nn.Linear(args.seq_len * args.d_model, args.d_model)
        self.fc2 = nn.Linear(args.d_model, args.output_size)

    def forward(self, x):
        # print(x.size())  # (256, 24, 7)
        y = x[:, -self.args.output_size:, :]
        # print(y.size())  # (256, 4, 7)
        x = self.input_fc(x)  # (256, 24, 128)
        x = self.pos_emb(x)   # (256, 24, 128)
        x = self.encoder(x)
        # 不经过解码器
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)
        # y = self.output_fc(y)   # (256, 4, 128)
        # out = self.decoder(y, x)  # (256, 4, 128)
        # out = out.flatten(start_dim=1)  # (256, 4 * 128)
        # out = self.fc(out)  # (256, 4)

        return out

class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self,feature_size=259,num_layers=3,dropout=0):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=7, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        device = "cpu"
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src,mask)
        output = self.decoder(output)
        return 
import random
import numpy
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import iterative_train_test_split

class DataClass(Dataset):

    def __init__(self,data):
        super().__init__()
        self.data = data
    def __getitem__(self, index):
        return super().__getitem__(index)
    def __len__(self):
        return self.data.size(0)
    
class DiabetesDataset_v3_1_old(Dataset):
    def __init__(self, csv_file="Dia182_normProb_0.7_1") -> None:
        # 实现初始化方法，在初始化的时候将数据载入
        super().__init__()
        if csv_file == "Dia182_normProb_0.7_1": # sub_60_sample_182
            # self.df = pd.read_csv(csv_file)
            self.df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy2.csv",header=None).values
        elif csv_file == "Dia182_normProb_0.3_1": # sub_60_sample_182
            self.df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.3_classify_copy2.csv",header=None).values
        else:
            assert ""

    def __len__(self):
        # 返回df的长度
        return len(self.df)
    
    def __getitem__(self, index):
        # 根据index返回一行数据，即一个样本数据
        # return self.df[index,:]

        # 获取数据的X_fgm、X_biomarkers、y_class、y_biomarkers_binary
        X_df_fgm = self.df[:,1:-22]
        y_df = self.df[:,0:1]
        more_y_df_total = self.df[:,-22:]
        more_y_df_raw = more_y_df_total[:,::2]    
        more_y_df_binary = more_y_df_total[:,1::2]
        X_df_biomarkers = more_y_df_raw

        # Move the labels to {0, ..., L-1}
        labels = np.unique(y_df)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i
        y_df = np.vectorize(transform.get)(y_df)
        
        # return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
        #         'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}
        return {'X_fgm':X_df_fgm[index,:],'X_biomarkers':X_df_biomarkers[index,:],
                'y_class':y_df[index,:],'y_biomarkers_binary':more_y_df_binary[index,:]}
    
    def get_xy(self):
            # 获取数据的X_fgm、X_biomarkers、y_class、y_biomarkers_binary
        X_df_fgm = self.df[:,1:-22]
        y_df = self.df[:,0:1]
        more_y_df_total = self.df[:,-22:]
        more_y_df_raw = more_y_df_total[:,::2]    
        more_y_df_binary = more_y_df_total[:,1::2]
        X_df_biomarkers = more_y_df_raw

        # Move the labels to {0, ..., L-1}
        labels = np.unique(y_df)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i
        y_df = np.vectorize(transform.get)(y_df)
        
        # return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
        #         'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}
        return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
                'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}
    

class DiabetesDataset_v3_1(Dataset):
    def __init__(self, batchsz, n_label, k_shot, k_query, csv_file="Dia182_normProb_0.7_1",mode='train',seed=42) -> None:
        # 实现初始化方法，在初始化的时候将数据载入
        super().__init__()
        self.batchsz = batchsz
        self.n_label = n_label # MAML中是n_way,即n个类别，本实验中替换成n个标签
        self.k_shot = k_shot # MAML中k_shots是指SupportSet中每个类别的样本数，本实验中是指所有标签的总样本数
        self.k_query = k_query # MAML中k_shots是指QuerySet中每个类别的样本数，本实验中是指所有标签的总样本数
        self.labels_num = 11 # MAML中是cls_num,即所有样本存在的类别总数，本实验中替换成所有的标签总数,
        self.attr = 11 # 训练样本数据的特征数
        self.mode = mode
        self.train_ratio = 0.7
        self.seed = seed # 随机选取元训练集样本的种子，剩下部分作为元测试集
        self.df = self.load_csv(csv_file)
        self.create_batch(self.batchsz)

    def __len__(self):
        # 返回df的长度
        # return len(self.df)
        return self.batchsz
    
    # def __getitem__(self, index):
    #     # 根据index返回一行数据，即一个样本数据
    #     # return self.df[index,:]

    #     # 获取数据的X_fgm、X_biomarkers、y_class、y_biomarkers_binary
    #     X_df_fgm = self.df[:,1:-22]
    #     y_df = self.df[:,0:1]
    #     more_y_df_total = self.df[:,-22:]
    #     more_y_df_raw = more_y_df_total[:,::2]    
    #     more_y_df_binary = more_y_df_total[:,1::2]
    #     X_df_biomarkers = more_y_df_raw

    #     # Move the labels to {0, ..., L-1}
    #     labels = np.unique(y_df)
    #     transform = {}
    #     for i, l in enumerate(labels):
    #         transform[l] = i
    #     y_df = np.vectorize(transform.get)(y_df)
        
    #     # return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
    #     #         'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}
    #     return {'X_fgm':X_df_fgm[index,:],'X_biomarkers':X_df_biomarkers[index,:],
    #             'y_class':y_df[index,:],'y_biomarkers_binary':more_y_df_binary[index,:]}
    
    def __getitem__(self, task_idx):
        # 获取任务编号为task_idx的任务数据，因此task_idx的范围为[0,batchsz-1]
        temp_support_x = self.support_x_batch[task_idx,:,:]
        # support_x = temp_support_x.reshape((temp_support_x.shape[0], temp_support_x.shape[1], 1))  # [n_instance, n_feature, n_feature_num]
        support_x = temp_support_x  # [n_instance, n_feature]
        temp_query_x = self.query_x_batch[task_idx,:,:]
        # query_x = temp_query_x.reshape((temp_query_x.shape[0], temp_query_x.shape[1], 1))  # [n_instance, n_feature, n_feature_num]
        query_x = temp_query_x  # [n_instance, n_feature]

        support_y = self.support_y_batch[task_idx,:,:]  # [n_instance, selected_label]
        query_y = self.query_y_batch[task_idx,:,:]  # [n_instance, selected_label]

        return torch.FloatTensor(support_x), torch.FloatTensor(support_y), torch.FloatTensor(query_x), torch.FloatTensor(query_y)

    def get_xy(self):
        # 获取数据的X_fgm、X_biomarkers、y_class、y_biomarkers_binary
        np.random.seed(self.seed)
        if self.mode == 'train':
            metaTrainIdx = np.random.choice(self.df.shape[0], round(self.df.shape[0]*self.train_ratio), replace=False)
            self.df = self.df[metaTrainIdx,:]
        else:
            metaTestIdx = np.random.choice(self.df.shape[0], round(self.df.shape[0]*(1-self.train_ratio)), replace=False)
            self.df = self.df[metaTestIdx,:]
            
        X_df_fgm = self.df[:,1:-22]
        y_df = self.df[:,0:1]
        more_y_df_total = self.df[:,-22:]
        more_y_df_raw = more_y_df_total[:,::2]    
        more_y_df_binary = more_y_df_total[:,1::2]
        X_df_biomarkers = more_y_df_raw

        # Move the labels to {0, ..., L-1}
        labels = np.unique(y_df)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i
        y_df = np.vectorize(transform.get)(y_df)
        
        # return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
        #         'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}
        return {'X_fgm':X_df_fgm,'X_biomarkers':X_df_biomarkers,
                'y_class':y_df,'y_biomarkers_binary':more_y_df_binary}

    def load_csv(self,csv_file):
        if csv_file == "Dia182_normProb_0.7_1": # sub_60_sample_182
            # self.df = pd.read_csv(csv_file)
            df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy2.csv",header=None).values
        elif csv_file == "Dia182_normProb_0.3_1": # sub_60_sample_182
            df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.3_classify_copy2.csv",header=None).values
        else:
            assert ""
        
        return df

    # 构造可以被选择元学习的任务集合,batchsz:任务集合的总数
    def create_batch(self, batchsz):
        data = self.get_xy()
        self.support_x_batch = []
        self.query_x_batch = []
        self.support_y_batch = []
        self.query_y_batch = []
        for b in range(batchsz):
            # # 1. 随机选择多个标签中的几个
            # selected_label = np.random.choice(self.labels_num, self.n_label, False)  # False 表明选定的label中无重复项
            # np.random.shuffle(selected_label)
            # support_x = []
            # query_x = []
            # # 2. 为每个任务选择 k_shots 和 k_query
            # selected_sample_idx = np.random.choice(len(self.df), self.k_shot + self.k_query, False)
            # np.random.shuffle(selected_sample_idx)
            # indexDtrain = np.array(selected_sample_idx[:self.k_shot]) # idx for Dtrain
            # indexDtest = np.array(selected_sample_idx[self.k_shot:])  # idx for Dtest
            # support_x.append(np.array(data['X_biomarkers'])[indexDtrain].tolist()) # 为当前任务获取训练样本
            # query_x.append(np.array(data['X_biomarkers'])[indexDtest].tolist())    # 为当前任务获取测试样本

            #########################
            # 1. 为每个任务选择 k_shot 和k_query
            support_x = []
            query_x = []
            support_y = []
            query_y = []

            selected_sample_idx = np.random.choice(len(self.df), self.k_shot + self.k_query, False)
            np.random.shuffle(selected_sample_idx)
            indexDtrain = np.array(selected_sample_idx[:self.k_shot]) # idx for Dtrain
            indexDtest = np.array(selected_sample_idx[self.k_shot:])  # idx for Dtest
            support_x = np.array(data['X_biomarkers'])[indexDtrain,:].tolist() # 为当前任务获取训练样本
            query_x = np.array(data['X_biomarkers'])[indexDtest,:].tolist()    # 为当前任务获取测试样本

            selected_label = np.random.choice(self.labels_num, self.n_label, False)  # False 表明选定的label中无重复项
            np.random.shuffle(selected_label)
            # 2. 同时选定每个k_shot和k_query的标签
            if self.mode == 'train':
                temp_data = np.hstack((np.array(data['y_biomarkers_binary']),np.array(data['y_class'])))
            else:
                temp_data = np.array(data['y_class'])
                
            temp_train = temp_data[indexDtrain,:]
            temp_test = temp_data[indexDtest,:]

            if self.mode == 'train':
                temp_train_1 = temp_train[:,selected_label].tolist()
                temp_test_1 = temp_test[:,selected_label].tolist()
            else:
                temp_train_1 = temp_train
                temp_test_1 = temp_test

            support_y = temp_train_1 # 为当前任务获取训练标签
            query_y = temp_test_1    # 为当前任务获取测试标签

            # # 打乱支持集和查询集之间的对应关系
            # random.shuffle(support_x)
            # random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # [batchsz, indexDtrain, 11]
            self.query_x_batch.append(query_x)      # [batchsz, indexDtest, 11]
            self.support_y_batch.append(support_y)  # [batchsz, indexDtrain, selected_label]
            self.query_y_batch.append(query_y)      # [batchsz, indexDtest, selected_label]

        self.support_x_batch = np.array(self.support_x_batch) # [batchsz, indexDtrain(k_shot), self.attr]
        self.query_x_batch = np.array(self.query_x_batch)     # [batchsz, indexDtest(k_query), self.attr]
        self.support_y_batch = np.array(self.support_y_batch) # [batchsz, indexDtrain(k_shot), selected_label]
        self.query_y_batch = np.array(self.query_y_batch)     # [batchsz, indexDtest(k_query), selected_label]
            
        return

if __name__ == '__main__':
    # test
    # test = DiabetesDataset_v3_1_old()
    test = DiabetesDataset_v3_1(batchsz=100,n_label=5,k_shot=10,k_query=40,csv_file="Dia182_normProb_0.7_1",mode='train')


    support_x, support_y, query_x, query_y = test[1]
    m=1

    # data_xy = test.get_xy()
    # X_fgm = data_xy['X_fgm']
    # X_biomarkers = data_xy['X_biomarkers']
    # y_class = data_xy['y_class']
    # y_biomarkers_binary = data_xy['y_biomarkers_binary']

    # X = np.hstack((X_fgm,X_biomarkers))
    # y = np.hstack((y_class,y_biomarkers_binary))

    # X_train, y_train, X_test, y_test = iterative_train_test_split(X,y,test_size=0.2)
    # X_fgm_train, y_train, X_test, y_test = train_test_split(X_fgm,X_biomarkers,y_class,y_biomarkers_binary,test_size=0.2,random_state=42,shuffle=True,stratify=y_class)

    # # 划分训练集、测试集和验证集

    # dataloader = DataLoader(test, batch_size=4, shuffle=True, num_workers=4)

    # for i, data in enumerate(dataloader):
    #     X_fgm = data['X_fgm']
    #     X_biomarkers = data['X_biomarkers']
    #     y_class = data['y_class']
    #     y_biomarkers_binary = data['y_biomarkers_binary']

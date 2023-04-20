import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_Diabete_classification(dataset):
    # 无辅助任务的数据
    if dataset == "Dia454":
        df = pd.read_csv("datasets/Diabetes/T1DM333_T2DM121_align_1+576.csv",header=None).values
    elif dataset == "Dia242":
        df = pd.read_csv("datasets/Diabetes/T1DM121_T2DM121_align_1+576.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,1:]
    y_df = df[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    
    # Move the labels to {0, ..., L-1}
    labels = np.unique(y_df)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    
    
    # print(X_train.shape)  # (363, 576)
    # print(X_test.shape)  # (91,576)
    # print(np.sum(y_train))  # 460.0
    # print(np.sum(y_test))   # 115.0

    X_train = X_train.astype(np.float64)
    train_labels = np.vectorize(transform.get)(y_train)
    X_test = X_test[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(y_test)

    # mean = np.nanmean(X_train)
    # std = np.nanstd(X_train)
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    return X_train[..., np.newaxis], train_labels, X_test[..., np.newaxis], test_labels

def load_Diabete_classification_v2(dataset, normalize=False):
    # 有辅助任务的数据，使用的是高血压和视网膜病变数据做辅助任务的标签，Dia220 是数据类别均衡化之后的数据
    if dataset == "Dia437":
        df = pd.read_csv("datasets/Diabetes_v2/T1DM327_T2DM110_align_1+576.csv",header=None).values
    elif dataset == "Dia220":
        df = pd.read_csv("datasets/Diabetes_v2/T1DM110_T2DM110_align_1+576-1.csv",header=None).values
    elif dataset == "Dia220_noBalance":
        df = pd.read_csv("datasets/Diabetes_v2/T1DM110_T2DM110_align_1+576.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,3:]
    y_df = df[:,0:3]

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    
    y_train_1 = y_train[:,0]
    y_train_2 = y_train[:,1]
    y_train_3 = y_train[:,2]

    y_test_1 = y_test[:,0]
    y_test_2 = y_test[:,1]
    y_test_3 = y_test[:,2]

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train_1)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    # Move the labels to {0, ..., L-1}
    labels_2 = np.unique(y_train_2)
    transform_2 = {}
    for i, l in enumerate(labels_2):
        transform_2[l] = i

    # Move the labels to {0, ..., L-1}
    labels_3 = np.unique(y_train_3)
    transform_3 = {}
    for i, l in enumerate(labels_3):
        transform_3[l] = i
    
    
    # print(X_train.shape)  # (349, 576)
    # print(X_test.shape)  # (88,576)
    # print(np.sum(y_train),axis=0)  # [437. 135.  99.]
    # print(np.sum(y_test),axis=0)   # [110.  34.  24.]

    X_train = X_train.astype(np.float64)
    train_labels_1 = np.vectorize(transform_1.get)(y_train_1)
    train_labels_2 = np.vectorize(transform_2.get)(y_train_2)
    train_labels_3 = np.vectorize(transform_3.get)(y_train_3)
    # X_test = X_test[:, 1:].astype(np.float64)
    X_test = X_test.astype(np.float64)
    test_labels_1 = np.vectorize(transform_1.get)(y_test_1)
    test_labels_2 = np.vectorize(transform_2.get)(y_test_2)
    test_labels_3 = np.vectorize(transform_3.get)(y_test_3)

    if normalize == True:
        mean = np.nanmean(X_train)
        std = np.nanstd(X_train)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return X_train[..., np.newaxis], train_labels_1, train_labels_2, train_labels_3, X_test[..., np.newaxis], test_labels_1, test_labels_2, test_labels_3 

def load_Diabete_classification_v2_1(dataset, normalize=False):
    # p2中使用的数据Dia182以及Dia175，没有辅助任务的数据，仅使用FGM数据
    if dataset == "Dia182_FGM":
        df = pd.read_csv("datasets/Diabetes_v2/All_DM_sub_60_sample_182.csv",header=None).values
    elif dataset == "Dia175_FGM":
        df = pd.read_csv("datasets/Diabetes_v2/All_DM_sub_110_sample_175.csv",header=None).values
    else:
        assert ""
    X_df = df[:,1:-11]
    y_df = df[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    X_train = X_train.astype(np.float64)
    train_labels = np.vectorize(transform_1.get)(y_train)
    X_test = X_test.astype(np.float64)
    test_labels = np.vectorize(transform_1.get)(y_test)

    if normalize == True:
        mean = np.nanmean(X_train)
        std = np.nanstd(X_train)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
    return X_train[..., np.newaxis], train_labels, X_test[..., np.newaxis], test_labels 

def load_Diabete_classification_v3(dataset):
    # 有辅助任务的数据，使用paper2使用的数据
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia182_normProb_0.7": # sub_60_sample_182
        # df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy.csv",header=None).values
        df = pd.read_csv("datasets/Diabetes_v4/All_DM_sub_60_sample_182_classify_copy.csv",header=None).values
    elif dataset == "Dia175_normProb_0.7":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.7_classify_copy.csv",header=None).values
    elif dataset == "Dia182_normProb_0.3": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.3_classify_copy.csv",header=None).values
    elif dataset == "Dia175_normProb_0.3":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.3_classify_copy.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,1:-11]
    y_df = df[:,0:1]
    more_y_df = df[:,-11:]
    all_y_df = np.append(y_df,more_y_df,axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_df, all_y_df, test_size=0.2, random_state=42, stratify=all_y_df[:,0], shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # 总共有12个指标，将指标label中数值变为{0, ..., L-1}
    # 例如，label原来是1和2，变为0和1
    for j in range(12):
        y_train_1 = y_train[:,j]
        y_test_1 = y_test[:,j]
        labels_1 = np.unique(y_train_1)
        transform_1 = {}
        for i, l in enumerate(labels_1):
            transform_1[l] = i
        train_labels_1 = np.vectorize(transform_1.get)(y_train_1)
        test_label_1 = np.vectorize(transform_1.get)(y_test_1)
        train_labels[:,j] = train_labels_1
        test_labels[:,j] = test_label_1

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # mean = np.nanmean(X_train)
    # std = np.nanstd(X_train)
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    return X_train[..., np.newaxis], train_labels, X_test[..., np.newaxis], test_labels

def load_Diabete_classification_v3_1(dataset):
    # 有辅助任务的数据，使用paper2使用的数据
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    dataset = "Dia182_normProb_0.7_1"
    if dataset == "Dia182_normProb_0.7_1": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy2.csv",header=None).values
        # df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_classify_copy.csv",header=None).values
    elif dataset == "Dia175_normProb_0.7_1":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.7_classify_copy.csv",header=None).values
    elif dataset == "Dia182_normProb_0.3_1": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.3_classify_copy2.csv",header=None).values
    elif dataset == "Dia175_normProb_0.3_1":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_110_sample_175_normProb_0.3_classify_copy.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,1:-22]
    y_df = df[:,0:1]
    more_y_df_total = df[:,-22:]
    more_y_df_raw = more_y_df_total[:,::2]
    more_y_df_binary = more_y_df_total[:,1::2]
    more_y_df = more_y_df_total
    all_y_df = np.append(y_df,more_y_df,axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_df, all_y_df, test_size=0.2, random_state=42, stratify=all_y_df[:,0], shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # 总共有12个指标，将指标label中数值变为{0, ..., L-1}
    # 例如，label原来是1和2，变为0和1
    for j in range(12):
        y_train_1 = y_train[:,j]
        y_test_1 = y_test[:,j]
        labels_1 = np.unique(y_train_1)
        transform_1 = {}
        for i, l in enumerate(labels_1):
            transform_1[l] = i
        train_labels_1 = np.vectorize(transform_1.get)(y_train_1)
        test_label_1 = np.vectorize(transform_1.get)(y_test_1)
        train_labels[:,j] = train_labels_1
        test_labels[:,j] = test_label_1

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # mean = np.nanmean(X_train)
    # std = np.nanstd(X_train)
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    return X_train[..., np.newaxis], train_labels, X_test[..., np.newaxis], test_labels

def load_Diabete_classification_v4(dataset):
    # 有辅助任务的数据，使用paper2使用的数据
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia182_LPA": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v4/All_DM_sub_60_sample_182_classify_copy_LPA_1.csv",header=None).values
    elif dataset == "Dia175_LPA":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v4/All_DM_sub_110_sample_175_classify_copy_LPA.csv",header=None).values
    else:
        assert ""
    # print(df.values.shape)    
    X_df = df[:,1:-11]
    y_df = df[:,0:1]
    more_y_df = df[:,-11:]
    all_y_df = np.append(y_df,more_y_df,axis=1)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_df, all_y_df, test_size=0.2, random_state=42, stratify=all_y_df[:,0], shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # 总共有12个指标，将指标label中数值变为{0, ..., L-1}
    # 例如，label原来是1和2，变为0和1
    for j in range(12):
        y_train_1 = y_train[:,j]
        y_test_1 = y_test[:,j]
        labels_1 = np.unique(y_train_1)
        transform_1 = {}
        for i, l in enumerate(labels_1):
            transform_1[l] = i
        train_labels_1 = np.vectorize(transform_1.get)(y_train_1)
        test_label_1 = np.vectorize(transform_1.get)(y_test_1)
        train_labels[:,j] = train_labels_1
        test_labels[:,j] = test_label_1

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # mean = np.nanmean(X_train)
    # std = np.nanstd(X_train)
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    return X_train[..., np.newaxis], train_labels, X_test[..., np.newaxis], test_labels

def load_Diabete_classification_v5(dataset, isBioNormalized=True):
    # 没有辅助任务，FGM使用ts2vec+sfa，Biomarkers使用VAE算法
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia182": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v5/All_DM_sub_60_sample_182_classify_copy.csv", header=None, encoding='latin-1').values
    elif dataset == "Dia175":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v5/All_DM_sub_110_sample_175_classify_copy.csv", header=None, encoding='latin-1').values
    else:
        assert ""
    # print(df.values.shape)    
    X_fgm_df = df[:,1:-11]
    y_df = df[:,0:1]
    X_biomark_df = df[:,-11:]

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_test, X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_fgm_df, X_biomark_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    train_labels = np.vectorize(transform_1.get)(y_train)
    test_labels = np.vectorize(transform_1.get)(y_test)

    # 对biomarker数据进行标准化处理
    if isBioNormalized == True:
        mean = np.nanmean(X_bio_train)
        std = np.nanstd(X_bio_train)
        X_bio_train = (X_bio_train - mean) / std
        X_bio_test = (X_bio_test - mean) / std

    return X_fgm_train[..., np.newaxis], X_bio_train, train_labels, X_fgm_test[..., np.newaxis], X_bio_test, test_labels

# 给生理数据加噪声
def addNoise(df):
    # 去除零元素之后求均值和方差
    def non_zero_mean_std(np_arr):
        exist = (np_arr != 0)
        num = np_arr.sum(axis=0)
        den = exist.sum(axis=0)
        
        res = num/den
        stds = []
        for j in range(np_arr.shape[1]):
            std = 0
            for i in range(np_arr.shape[0]):
                if np_arr[i,j] != 0:
                    std = std + (np_arr[i,j] - res[j]) ** 2
            std = (std/den[j]) ** 0.5
            stds.append(std)
        return res,stds

    # 给原数组加上服从列向高斯分布的随机噪声
    def addNoise_util(arr_old):
        means, stds = non_zero_mean_std(arr_old)
        for j in range(arr_old.shape[1]):
            # x = np.random.normal(loc=means[j], scale=stds[j], size=(arr_old.shape[0]))
            x = np.random.normal(loc=0, scale=1, size=(arr_old.shape[0]))
            # print(x)
            # print(arr_old[:,j])
            arr_old[:,j] = arr_old[:,j] + x
            
        return arr_old
    # tails = df[:,577:588]
    tails = df
    tails_new = addNoise_util(tails)
    # df_new = np.hstack((df[:,0:577],tails_new))
    # df_new = np.abs(tails_new)
    df_new = tails_new
    return df_new
    
def load_Diabete_classification_v5_with_val(dataset, isBioNormalized=True):
    # 没有辅助任务，FGM使用ts2vec+sfa，Biomarkers使用VAE算法
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia182": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v5/All_DM_sub_60_sample_182_classify_copy.csv", header=None, encoding='latin-1').values
    elif dataset == "Dia175":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v5/All_DM_sub_110_sample_175_classify_copy.csv", header=None, encoding='latin-1').values
    else:
        assert ""
    # print(df.values.shape)    
    X_fgm_df = df[:,1:-11]
    y_df = df[:,0:1]
    X_biomark_df = df[:,-11:]
    # 生理数据是否需要加噪声
    flag_addNoise = True
    if flag_addNoise == True:
        X_biomark_df = addNoise(X_biomark_df)
        # np.random.shuffle(df_new)
    else:
        df_new = df.to_numpy()
        # np.random.shuffle(df_new)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_test, X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_fgm_df, X_biomark_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_val, X_bio_train, X_bio_val, y_train, y_val = train_test_split(X_fgm_train, X_bio_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    val_labels = np.empty(y_test.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    train_labels = np.vectorize(transform_1.get)(y_train)
    val_labels = np.vectorize(transform_1.get)(y_val)
    test_labels = np.vectorize(transform_1.get)(y_test)

    # 对biomarker数据进行标准化处理
    if isBioNormalized == True:
        mean = np.nanmean(X_bio_train)
        std = np.nanstd(X_bio_train)
        X_bio_train = (X_bio_train - mean) / std
        X_bio_val = (X_bio_val - mean) / std
        X_bio_test = (X_bio_test - mean) / std

    return X_fgm_train[..., np.newaxis], X_bio_train, train_labels, X_fgm_val[..., np.newaxis], X_bio_val, val_labels, X_fgm_test[..., np.newaxis], X_bio_test, test_labels
    # return X_fgm_train[..., np.newaxis], X_bio_train[..., np.newaxis], train_labels, X_fgm_val[..., np.newaxis], X_bio_val[..., np.newaxis], val_labels, X_fgm_test[..., np.newaxis], X_bio_test[..., np.newaxis], test_labels

def load_Diabete_classification_v5_with_val_origin(dataset, isBioNormalized=True, num_CV=0):
    # 没有辅助任务，FGM使用ts2vec+sfa，Biomarkers使用VAE算法
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia182": # sub_60_sample_182
        train_filename = "datasets/Diabetes_v5/data-new-2_types/synthetic_detrended_6d_2types_f"+ str(num_CV) +"_train.csv"
        test_filename = "datasets/Diabetes_v5/data-new-2_types/synthetic_detrended_6d_2types_f"+ str(num_CV) +"_test.csv"
        df_train  = pd.read_csv(train_filename, header=None, encoding='latin-1').values
        df_test  = pd.read_csv(test_filename, header=None, encoding='latin-1').values
    elif dataset == "Dia175":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v5/All_DM_sub_110_sample_175_classify_copy.csv", header=None, encoding='latin-1').values
    else:
        assert ""
    
    # 生理数据是否需要加噪声
    flag_addNoise = True
    if flag_addNoise == True:
        df_train = addNoise(df_train)  
        df_test = addNoise(df_test)  

    y_train = df_train[:,0:1]
    y_test = df_test[:,0:1]

    
    # print(df.values.shape)    
    X_fgm_df = df[:,1:-11]
    y_df = df[:,0:1]
    X_biomark_df = df[:,-11:]

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_test, X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_fgm_df, X_biomark_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_val, X_bio_train, X_bio_val, y_train, y_val = train_test_split(X_fgm_train, X_bio_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    val_labels = np.empty(y_test.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    train_labels = np.vectorize(transform_1.get)(y_train)
    val_labels = np.vectorize(transform_1.get)(y_val)
    test_labels = np.vectorize(transform_1.get)(y_test)

    # 对biomarker数据进行标准化处理
    if isBioNormalized == True:
        mean = np.nanmean(X_bio_train)
        std = np.nanstd(X_bio_train)
        X_bio_train = (X_bio_train - mean) / std
        X_bio_val = (X_bio_val - mean) / std
        X_bio_test = (X_bio_test - mean) / std

    # return X_fgm_train[..., np.newaxis], X_bio_train, train_labels, X_fgm_val[..., np.newaxis], X_bio_val, val_labels, X_fgm_test[..., np.newaxis], X_bio_test, test_labels
    return X_fgm_train[..., np.newaxis], X_bio_train[..., np.newaxis], train_labels, X_fgm_val[..., np.newaxis], X_bio_val[..., np.newaxis], val_labels, X_fgm_test[..., np.newaxis], X_bio_test[..., np.newaxis], test_labels

def load_Diabete_classification_v6(dataset, isBioNormalized=True):
    # 没有辅助任务，FGM使用ts2vec+sfa，Biomarkers使用VAE算法
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia220": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v6/T1DM110_T2DM110_align_1+576.csv", header=None, encoding='latin-1').values
    elif dataset == "Dia437":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v6/T1DM327_T2DM110_align_1+576.csv", header=None, encoding='latin-1').values
    else:
        assert ""
    # print(df.values.shape)    
    X_fgm_df = df[:,14:]
    y_df = df[:,0]
    X_biomark_df = df[:,3:14]
    # 生理数据是否需要加噪声
    flag_addNoise = False
    if flag_addNoise == True:
        X_biomark_df = addNoise(X_biomark_df)
        np.random.shuffle(X_biomark_df)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_test, X_bio_train, X_bio_test, y_train, y_test = train_test_split(X_fgm_df, X_biomark_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_val, X_bio_train, X_bio_val, y_train, y_val = train_test_split(X_fgm_train, X_bio_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    val_labels = np.empty(y_test.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    train_labels = np.vectorize(transform_1.get)(y_train)
    val_labels = np.vectorize(transform_1.get)(y_val)
    test_labels = np.vectorize(transform_1.get)(y_test)

    # 对biomarker数据进行标准化处理
    if isBioNormalized == True:
        mean = np.nanmean(X_bio_train)
        std = np.nanstd(X_bio_train)
        X_bio_train = (X_bio_train - mean) / std
        X_bio_val = (X_bio_val - mean) / std
        X_bio_test = (X_bio_test - mean) / std

        # min_value = np.min(X_bio_train,axis=0)
        # max_value = np.max(X_bio_train,axis=0)

        # X_bio_train = (X_bio_train - min_value) / (max_value - min_value)
        # X_bio_val = (X_bio_val - min_value) / (max_value - min_value)
        # X_bio_test = (X_bio_test - min_value) / (max_value - min_value)

    return X_fgm_train[..., np.newaxis], X_bio_train, train_labels, X_fgm_val[..., np.newaxis], X_bio_val, val_labels, X_fgm_test[..., np.newaxis], X_bio_test, test_labels
    # return X_fgm_train[..., np.newaxis], X_bio_train[..., np.newaxis], train_labels, X_fgm_val[..., np.newaxis], X_bio_val[..., np.newaxis], val_labels, X_fgm_test[..., np.newaxis], X_bio_test[..., np.newaxis], test_labels

def load_Diabete_classification_v6_Neighbor(dataset, isBioNormalized=True):
    # 没有辅助任务，FGM使用ts2vec+sfa，Biomarkers使用VAE算法
    # Dia182：train:test = 145:37
    # Dia175：train:test = 140:35
    if dataset == "Dia220": # sub_60_sample_182
        df = pd.read_csv("datasets/Diabetes_v6/T1DM110_T2DM110_align_1+576.csv", header=None, encoding='latin-1').values
    elif dataset == "Dia437":  # sub_110_sample_175
        df = pd.read_csv("datasets/Diabetes_v6/T1DM327_T2DM110_align_1+576_Neighbor.csv", header=None, encoding='latin-1').values
    else:
        assert ""
    # print(df.values.shape)    
    X_fgm_df = df[:,15:]
    y_df = df[:,0]
    neighbor = df[:,1]
    X_biomark_df = df[:,4:15]
    # 生理数据是否需要加噪声
    flag_addNoise = False
    if flag_addNoise == True:
        X_biomark_df = addNoise(X_biomark_df)
        np.random.shuffle(X_biomark_df)

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_test, X_bio_train, X_bio_test, y_train, y_test, n_train, n_test = train_test_split(X_fgm_df, X_biomark_df, y_df, neighbor, test_size=0.2, random_state=42, stratify=y_df, shuffle=True)
    X_fgm_train, X_fgm_val, X_bio_train, X_bio_val, y_train, y_val, n_train, n_val = train_test_split(X_fgm_train, X_bio_train, y_train, n_train, test_size=0.2, random_state=42, stratify=y_train, shuffle=True)
    
    train_labels = np.empty(y_train.shape,dtype=int)
    val_labels = np.empty(y_test.shape,dtype=int)
    test_labels = np.empty(y_test.shape,dtype=int)
    train_neighbor = np.empty(n_train.shape,dtype=int)
    val_neighbor = np.empty(n_test.shape,dtype=int)
    test_neighbor = np.empty(n_test.shape,dtype=int)



    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_train)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i

    train_labels = np.vectorize(transform_1.get)(y_train)
    val_labels = np.vectorize(transform_1.get)(y_val)
    test_labels = np.vectorize(transform_1.get)(y_test)

    # Move the labels to {0, ..., L-1}
    labels_2 = np.unique(n_train)
    transform_2 = {}
    for i, l in enumerate(labels_2):
        transform_2[l] = i

    train_neighbor = np.vectorize(transform_2.get)(n_train)
    val_neighbor = np.vectorize(transform_2.get)(n_val)
    test_neighbor = np.vectorize(transform_2.get)(n_test)

    # 对biomarker数据进行标准化处理
    if isBioNormalized == True:
        mean = np.nanmean(X_bio_train)
        std = np.nanstd(X_bio_train)
        X_bio_train = (X_bio_train - mean) / std
        X_bio_val = (X_bio_val - mean) / std
        X_bio_test = (X_bio_test - mean) / std

        # min_value = np.min(X_bio_train,axis=0)
        # max_value = np.max(X_bio_train,axis=0)

        # X_bio_train = (X_bio_train - min_value) / (max_value - min_value)
        # X_bio_val = (X_bio_val - min_value) / (max_value - min_value)
        # X_bio_test = (X_bio_test - min_value) / (max_value - min_value)

    return X_fgm_train[..., np.newaxis], X_bio_train, train_labels, train_neighbor, X_fgm_val[..., np.newaxis], X_bio_val, val_labels, val_neighbor, X_fgm_test[..., np.newaxis], X_bio_test, test_labels, test_neighbor
    # return X_fgm_train[..., np.newaxis], X_bio_train[..., np.newaxis], train_labels, X_fgm_val[..., np.newaxis], X_bio_val[..., np.newaxis], val_labels, X_fgm_test[..., np.newaxis], X_bio_test[..., np.newaxis], test_labels


def load_Diabete_prediction(dataset):


    return

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    # data = np.load(f'datasets/{name}.npy')    
    data = np.load(f'datasets/Diabetes/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


if __name__ == "__main__":
    dataset = "Dia437"
    # x_train, y_train, x_test, y_test = load_Diabete_classification_v5(dataset)
    load_Diabete_classification_v6(dataset)
    pass
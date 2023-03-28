import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def data_loader_v2():
    # df = pd.read_csv("datasets/Diabetes_v2/T1DM110_T2DM110_align_1+576.csv",header=None).values
    df = pd.read_csv("datasets/Diabetes_v2/T1DM110_T2DM110_align_1+576-1.csv",header=None).values
    
    X_df = df[:,3:]
    y_df = df[:,0:3]

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_df[:,0])
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i
    y_df[:,0] = np.vectorize(transform_1.get)(y_df[:,0])
    
    return X_df, y_df

def data_loader_v3_1():

    df = pd.read_csv("datasets/Diabetes_v3/All_DM_sub_60_sample_182_normProb_0.7_classify_copy2.csv",header=None).values
    
    X_df = df[:,1:-22]
    y_df = df[:,0:1]
    more_y_df_total = df[:,-22:]
    more_y_df_raw = more_y_df_total[:,::2]
    more_y_df_binary = more_y_df_total[:,1::2]
    more_y_df = more_y_df_total
    all_y_df = np.append(y_df,more_y_df,axis=1)

    # Move the labels to {0, ..., L-1}
    labels_1 = np.unique(y_df)
    transform_1 = {}
    for i, l in enumerate(labels_1):
        transform_1[l] = i
    labels_1 = np.vectorize(transform_1.get)(y_df)
    
    return X_df, more_y_df_raw, labels_1, more_y_df_binary


if __name__ == '__main__':

    # fgm_X, fgm_y = data_loader_v2()
    fgm_X1, fgm_X2, fgm_y0, fgm_y_more = data_loader_v3_1()
    
    parameters = {'kernel':('linear', 'rbf'), 'C':[1,10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, cv=5)
    # for i in range(3):
    #     clf.fit(fgm_X, fgm_y[:,i])
    #     print(clf.best_score_)  
    #     print(clf.best_params_)
    #     print()

    for i in range(11):
        clf.fit(fgm_X2, fgm_y_more[:,i])
        print(clf.best_score_)  
        print(clf.best_params_)
        print()
    pass
## 使用Label Propagation Algorithm(LPA) 对糖尿病数据中生化指标缺失数据进行处理。
## 把生化指标当成标签，缺失值相当于无标签数据，使用LPA算法对无标签数据打上伪标签

import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelPropagation

def test(filePath):
    df = pd.read_csv(filePath,header=None).values
    y_df_0 = df[:,0:1]
    X_df = df[:,1:-11]
    y_df = df[:,-11:].astype('int64')

    new_y_df = y_df.copy()

    for i in range(11):
        label_prop_model = LabelPropagation()
        label_prop_model.fit(X_df,y_df[:,i])
        pseudo_labels = label_prop_model.transduction_
        new_y_df[:,i] = pseudo_labels

        print(pseudo_labels)
    
    # new_y_df = pd.DataFrame(new_y_df)
    result = np.hstack((y_df_0,X_df,new_y_df))
    result_df = pd.DataFrame(result)
    result_df.to_csv("datasets/Diabetes_v4/All_DM_sub_110_sample_175_classify_copy_LPA.csv",index=None,header=None)
    return

if __name__=='__main__':
    dirPath = "datasets/Diabetes_v4/"
    fileName1 = "All_DM_sub_60_sample_182_classify_copy.csv"
    fileName2 = "All_DM_sub_110_sample_175_classify_copy.csv"

    test(dirPath+fileName1)
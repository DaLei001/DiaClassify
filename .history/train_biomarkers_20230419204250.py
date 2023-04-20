from sklearn import svm
import torch
import datautils
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# def 

if __name__ == '__main__':
    task_type = 'classification'
    isBioNormalized = True
    train_data, train_bio_data, train_labels,\
            val_data, val_bio_data, val_labels,\
            test_data, test_bio_data, test_labels\
            = datautils.load_Diabete_classification_v6("Dia437", isBioNormalized)
    
    model1 = SelectKBest(mutual_info_classif,k=2)
    model1.fit(train_bio_data,train_labels)
#     X_new = model1.transform(train_bio_data)
    X_new = model1.transform(val_bio_data)
    X_new1 = model1.transform(test_bio_data)

    model = svm.SVC()
#     model.fit(train_bio_data,train_labels)
    model.fit(val_bio_data,val_labels)
    s1 = model.score(test_bio_data, test_labels)

#     model.fit(X_new, train_labels)
    model.fit(X_new, val_labels)
    s2 = model.score(X_new1, test_labels)


    pass
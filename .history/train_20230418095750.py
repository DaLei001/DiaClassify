import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
from ts2vec_SFA_Neighbor import TS2Vec_SFA_Neighbor
from ts2vec_wl import TS2Vec_wl
from ts2vec_SFA import TS2Vec_SFA
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
# Import comet_ml at the top of your file
from comet_ml import Experiment
from sklearn.model_selection import GridSearchCV

# # Create an experiment with your api key
# experiment = Experiment(
#     api_key="imiTOqzHp0lZLqDOHdoPcm678",
#     project_name="general",
#     workspace="dalei001",
# )


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    # parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    # parser.add_argument('--epochs', type=int, default=40, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--dataNormalization', type=bool, default=False, help='the flag to determine whether to add the normalization operation for data')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    if args.gpu == 0 and torch.cuda.is_available():
        rec_device = "cuda:0"
    else:
        rec_device = 'cpu'
    
    device = init_dl_program(rec_device, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'Diabetes_Classification':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels \
            = datautils.load_Diabete_classification(args.dataset)
    
    elif args.loader == 'Diabetes_Classification_v2':
        task_type = 'classification'
        isNormalized = args.dataNormalization
        train_data, train_labels, train_labels_1, train_labels_2, test_data, test_labels, \
            test_labels_2, test_labels_3 = datautils.load_Diabete_classification_v2(args.dataset, isNormalized)
    
    elif args.loader == 'Diabetes_Classification_v2_1':
        task_type = 'classification'
        isNormalized = args.dataNormalization
        train_data, train_labels, test_data, test_labels \
            = datautils.load_Diabete_classification_v2_1(args.dataset, isNormalized)
    
    elif args.loader == 'Diabetes_Classification_v3':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_Diabete_classification_v3(args.dataset)    
        more_train_laebls = train_labels[:,1:]
        train_labels = train_labels[:,0]
        more_test_laebls = test_labels[:,1:]
        test_labels = test_labels[:,0]

    elif args.loader == 'Diabetes_Classification_v3_1':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_Diabete_classification_v3_1(args.dataset)    
        more_train_laebls = train_labels[:,1:]
        train_labels = train_labels[:,0]
        more_test_laebls = test_labels[:,1:]
        test_labels = test_labels[:,0]
    
    elif args.loader == 'Diabetes_Classification_v4':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_Diabete_classification_v4(args.dataset)    
        more_train_laebls = train_labels[:,1:]
        train_labels = train_labels[:,0]
        more_test_laebls = test_labels[:,1:]
        test_labels = test_labels[:,0]
    
    elif args.loader == 'Diabetes_Classification_v5':
        task_type = 'classification'
        # isNormalized = args.dataNormalization
        isBioNormalized = True
        train_data, train_bio_data, train_labels, test_data, test_bio_data, test_labels\
            = datautils.load_Diabete_classification_v5(args.dataset, isBioNormalized)
    
    elif args.loader == 'Diabetes_Classification_v6':
        task_type = 'classification'
        isBioNormalized = True
        train_data, train_bio_data, train_labels,\
              val_data, val_bio_data, val_labels,\
              test_data, test_bio_data, test_labels\
                = datautils.load_Diabete_classification_v6(args.dataset, isBioNormalized)
    
    elif args.loader == 'Diabetes_Classification_v6_neighbor':
        task_type = 'classification'
        isBioNormalized = True
        train_data, train_bio_data, train_labels, train_neighbor,\
              val_data, val_bio_data, val_labels, val_neighbor,\
              test_data, test_bio_data, test_labels, test_neighbor\
                = datautils.load_Diabete_classification_v6_Neighbor(args.dataset, isBioNormalized)
        
    elif args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    # 实验结果保存的文件夹
    run_dir = 'training/exp02_1-Dia182_FGM/' + args.dataset + '__' + name_with_datetime(args.run_name)
    # run_dir = 'training/exp_ACSF1/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()

    # optimal:AdamW, Adam, SGD, SGD_momentum0.9
    optimal="Adam"
    # TS2Vec, SlowFeature,构建样本对的方式，TS2Vec：使用TS2Vec原文中的方法，SlowFeature：使用慢特征分析的方法
    # TS2Vec_SFA_Neighbor:使用慢特征+邻域对比损失
    sample_pair="TS2Vec_SFA_Neighbor" # TS2Vec, SlowFeature,构建样本对的方式，TS2Vec：使用TS2Vec原文中的方法，SlowFeature：使用慢特征分析的方法

    if sample_pair == "SlowFeature" or sample_pair == "SlowFeature":
        model = TS2Vec_SFA(input_dims=train_data.shape[-1], device=device, **config)
        loss_log = model.fit(train_data, n_epochs=args.epochs, n_iters=args.iters, \
                             verbose=True, optimal=optimal, sample_pair=sample_pair)
    elif sample_pair == "TS2Vec_SFA_Neighbor":
        model = TS2Vec_SFA_Neighbor(input_dims=train_data.shape[-1], device=device, **config)
        loss_log = model.fit(train_data, train_neighbor, n_epochs=args.epochs, n_iters=args.iters, \
                             verbose=True, optimal=optimal, sample_pair=sample_pair)

    # Dia484,Dia220,Diabetes_v2
    # loss_log = model.fit(
    #     train_data,
    #     train_labels_1, 
    #     train_labels_2, 
    #     n_epochs=args.epochs,
    #     n_iters=args.iters,
    #     verbose=True
    # )

    ## Dia175,Dia182,Diabetes_v3
    # loss_log = model.fit(
    #     train_data,
    #     more_train_laebls,
    #     n_epochs=args.epochs,
    #     n_iters=args.iters,
    #     verbose=True
    # )

    model.save(f'{run_dir}/model.pkl')
    # run_dir = 'training/' + 'Dia454__Classify_20221228_153224'
    # model.load(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            # out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
            out, eval_res = tasks.eval_classification(model, val_data, val_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")

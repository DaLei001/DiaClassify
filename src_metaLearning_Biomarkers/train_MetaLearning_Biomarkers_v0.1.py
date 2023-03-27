## 本文件是MetaLearning方法用于Biomarkers的主函数文件 

import os
import argparse
import time
import torch
import numpy as np
from maml import MAML
from torch.utils.data import DataLoader
from data_loader import DiabetesDataset_v3_1
from ml_utils import name_with_datetime
# from learner import MLP, SimpleCNN, MiddleCNN, ComplicatedCNN

def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    MLP = [
        ('linear', [500, args.n_attr]),
        ('relu', [True]),
        ('linear', [args.n_label, 500])
    ]

    SimpleCNN = [
        ('unsqueeze',[]),
        ('conv1d',[32,1,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        ('flatten',[]),
        ('linear',[args.n_label, 32*4]),
    ]

    # 最佳性能：test acc:0.79
    MiddleCNN = [   
        ('unsqueeze',[]),
        ('conv1d',[32,1,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        ('conv1d',[32,32,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        ('flatten',[]),
        ('linear',[8, 32*1]),
        ('relu', [True]),
        ('linear', [args.n_label, 8])
    ]

    # 最佳性能：test acc:
    ComplicatedCNN = [
        ('unsqueeze',[]),
        ('conv1d',[16,1,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[16]),
        ('max_pool1d',[2,2,0]),
        ('conv1d',[32,16,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        # ('conv1d',[128,128,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        # ('relu',[True]),
        # ('bn',[128]),
        # ('max_pool1d',[2,2,0]),
        ('flatten',[]),
        ('linear',[8, 32*1]),
        ('relu', [True]),
        ('linear', [args.n_label, 8])
    ]
    
    # config = ComplicatedCNN
    config = MiddleCNN

    # device = torch.device('cuda')
    device = torch.device('cpu')
    maml = MAML(args, config).to(device) # 定义MAML模型

    print(maml)

    run_dir = 'src_metaLearning_Biomarkers/training_log/' + args.dataset + '__' + name_with_datetime("T")
    os.makedirs(run_dir, exist_ok=True)

    train_dataset = DiabetesDataset_v3_1(batchsz=5000,n_label=args.n_label,k_shot=args.k_spt,k_query=args.k_qry,csv_file="Dia182_normProb_0.7_1",mode='train')
    test_dataset = DiabetesDataset_v3_1(batchsz=100,n_label=args.n_label,k_shot=args.k_spt,k_query=args.k_qry,csv_file="Dia182_normProb_0.7_1",mode='test')
    
    time_start = time.time()
    
    max_test_acc = 0
    for epoch in range(args.epoch//1000):
        print("***********epoch: ",epoch)

        # fetch meta_batchsz num of episode each time
        db = DataLoader(dataset=train_dataset, batch_size=args.task_num, shuffle=True, num_workers=2, pin_memory=True)

        # 每个 epoch 中的迭代数（step）total_step = data_total_batchsz(即定义数据集时传入的batchsz)/batch_size = 1000/4 = 250
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 5 == 0:  # total_step = data_total_batchsz(即定义数据集时传入的batchsz)/batch_size
                print('step:', step, '\ttraining acc:', accs)

            if step % 50 == 0:  # evaluation
                db_test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                # 如果此时模型的test_accs＞max_test_acc,那么就保存最好模型，并更新max_test_acc
                if accs > max_test_acc:
                    max_test_acc = accs
                    print("save model")
                    # 保存模型语句
                    torch.save(maml.state_dict(), f"{run_dir}/best_model.pkl")
                    torch.save({"accs:": max_test_acc}, f"{run_dir}/eval_res.pkl")

        print("The current usage time is: %s s." % np.round(time.time()-time_start))
        print('\n')
        pass
    print("The total usage time is: %s s." % np.round(time.time()-time_start))
    return

def test(run_dir):
    # 最佳性能：test acc:0.79
    MiddleCNN = [   
        ('unsqueeze',[]),
        ('conv1d',[32,1,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        ('conv1d',[32,32,3,1,0]), # [weight_out_channels,weight_in_channels, weight_filtersz, stride, padding]
        ('relu',[True]),
        ('bn',[32]),
        ('max_pool1d',[2,2,0]),
        ('flatten',[]),
        ('linear',[8, 32*1]),
        ('relu', [True]),
        ('linear', [args.n_label, 8])
    ]

    config = MiddleCNN

    device = torch.device('cpu')
    maml = MAML(args, config).to(device) # 定义MAML模型

    train_dataset = DiabetesDataset_v3_1(batchsz=5000,n_label=args.n_label,k_shot=args.k_spt,k_query=args.k_qry,csv_file="Dia182_normProb_0.7_1",mode='train')
    test_dataset = DiabetesDataset_v3_1(batchsz=100,n_label=args.n_label,k_shot=args.k_spt,k_query=args.k_qry,csv_file="Dia182_normProb_0.7_1",mode='test')
    
    
    maml.load_state_dict(torch.load(f"{run_dir}/best_model.pkl"))
    
    db_test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    accs_all_test = []

    for x_spt, y_spt, x_qry, y_qry in db_test:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                        x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    print('Test acc:', accs)
    
def load_test_res(run_dir):
    a = torch.load(f"{run_dir}/eval_res.pkl")
    print(a)

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset', default="Dia182_normProb_0.7_1")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--task_num', type=int, help='meta batch szie, namely task number', default=8)
    argparser.add_argument('--n_label', type=int, help='n label', default=5) # MAML中是n_way,即n个类别，本实验中替换成n个标签
    argparser.add_argument('--n_attr', type=int, help='num of feature', default=11) # 每个样本的特征数
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=20)  # MAML 中默认是1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=30)    # MAML 中默认是15
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)  # 5-->10
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetuning', default=10)

    args = argparser.parse_args()

    main()
    # test("src_metaLearning_Biomarkers/training_log/Dia182_normProb_0.7_1__T_20230316_112347")
    # load_test_res("src_metaLearning_Biomarkers/training_log/Dia182_normProb_0.7_1__T_20230316_173134")
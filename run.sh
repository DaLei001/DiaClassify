#!/bin/bash
# python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --seed 10 --eval 
# python train.py Dia220 Classify\
#         --loader Diabetes_Classification_v2\
#         --batch-size 8\
#         --lr 0.001\
#         --repr-dims 320\
#         --epochs None\
#         --seed None\
#         --irregular 0\
#         --gpu 0\
#         --eval
# python train.py ACSF1 ACSF1 --loader UCR --batch-size 8 --gpu 0 --repr-dims 320 --seed 10 --eval
# nohup python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --eval >file1.out&
# python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --eval
# for loop in 0 1 2 3 4 5 6 7 8 9 10
# for loop in 3
for loop in 0 1 2 3
do
    # python train.py Dia182_FGM Classify --loader Diabetes_Classification_v2_1 --batch-size 8 --gpu 0 --repr-dims 320 --seed $loop --eval
    # python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --seed $loop --eval
    # python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --seed 4 --eval
    # python train.py Dia437 Classify --loader Diabetes_Classification_v6 --batch-size 8 --gpu 0 --repr-dims 320 --seed $loop --eval
    python train.py Dia437 Classify --loader Diabetes_Classification_v6_neighbor --batch-size 8 --gpu 0 --repr-dims 320 --seed $loop --eval
    # python train.py ACSF1 ACSF1 --loader UCR --batch-size 8 --gpu 0 --repr-dims 320 --seed 10 --eval
done
# python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --seed 0 --eval
# python train.py Dia182_FGM Classify --loader Diabetes_Classification_v2_1 --batch-size 8 --gpu 0 --repr-dims 320 --seed 8 --eval

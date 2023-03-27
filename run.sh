#!/bin/bash

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

for loop in 2 4 8
do
    python train.py Dia220 Classify --loader Diabetes_Classification_v2 --batch-size 8 --gpu 0 --repr-dims 320 --eval
    # python train.py ACSF1 ACSF1 --loader UCR --batch-size $loop --gpu 0 --repr-dims 320 --seed 10 --eval
done



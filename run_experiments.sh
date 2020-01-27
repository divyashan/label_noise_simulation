#!/bin/bash

python evaluate.py --config configs/cifar_test.yaml
python evaluate.py --config configs/cifar_test_rank_pruning.yaml
#python train.py --config configs/cifar_swap2.yaml
#python train.py --config configs/cifar_swap3.yaml
#python train.py --config configs/cifar_swap5.yaml
#python train.py --config configs/cifar_clean.yaml

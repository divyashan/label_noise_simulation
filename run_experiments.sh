#!/bin/bash

python train.py --config configs/cifar_swap2.yaml
python train.py --config configs/cifar_swap3.yaml
python train.py --config configs/cifar_swap5.yaml
python train.py --config configs/cifar_clean

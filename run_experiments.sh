#!/bin/bash

python train.py --config configs/blob.yaml
python train.py --config configs/mv_gaussian.yaml
python train.py --config configs/mv_gaussian2.yaml
python train.py --config configs/mv_gaussian3.yaml
python train.py --config configs/exponential.yaml
python train.py --config configs/exponential2.yaml
python train.py --config configs/exponential3.yaml
python train.py --config configs/geometric.yaml
python train.py --config configs/uniform.yaml

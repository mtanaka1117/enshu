#!/bin/sh
echo "Adaptive Linear Single"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding single_group --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation Single"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding single_group --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Linear Unshifted"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding group_unshifted --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation Unshifted"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding group_unshifted --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Linear Pruned"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding pruned --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation Pruned"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding pruned --encryption stream --collection-rate 0.3 1.0 0.1

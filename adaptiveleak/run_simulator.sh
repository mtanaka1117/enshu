#!/bin/sh
echo "Uniform Standard"
python simulator.py --dataset $1 --policy uniform --encoding standard --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Linear Standard"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding standard --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation Standard"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding standard --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Linear AGE"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding group --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation AGE"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding group --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Linear Padded"
python simulator.py --dataset $1 --policy adaptive_heuristic --encoding padded --encryption stream --collection-rate 0.3 1.0 0.1

echo "Adaptive Deviation Padded"
python simulator.py --dataset $1 --policy adaptive_deviation --encoding padded --encryption stream --collection-rate 0.3 1.0 0.1


#! /bin/bash

CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops 0 --encoder cnn --epochs 5 --learning_rate 0.01 >tmp_cnn_lr0.01.txt
CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops 0 --encoder cnn --epochs 5 --learning_rate 0.05 >tmp_cnn_lr0.05.txt
CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops 0 --encoder cnn --epochs 5 --learning_rate 0.1 > tmp_cnn_lr0.1.txt

#! /bin/bash

for i in {0..3} 
do
	echo $i
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset trec --memory_size 20 --embedding_size 50 --hops $i --encoder avg --epochs 5  >./results/log_trec_avg_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset trec --memory_size 20 --embedding_size 50 --hops $i --encoder rnn --epochs 5  >./results/log_trec_rnn_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset trec --memory_size 20 --embedding_size 50 --hops $i --encoder cnn --epochs 5  >./results/log_trec_cnn_mem_20_hub_$i".txt"

done 

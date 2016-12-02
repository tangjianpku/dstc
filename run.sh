#! /bin/bash

for i in {0..3} 
do
	echo $i
	CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops $i --encoder avg --epochs 10  >./results/log_dblp3_avg_mem_20_hub_$i".txt.tmp4"
	#CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops $i --encoder rnn --epochs 20  >./results/log_dblp3_rnn_mem_20_hub_$i".txt"
	#CUDA_VISIBLE_DEVICES=3 python main.py --dataset dblp3 --memory_size 20 --embedding_size 50 --hops $i --encoder cnn --epochs 20  >./results/log_dblp3_cnn_mem_20_hub_$i".txt"

done 

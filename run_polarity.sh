#! /bin/bash

for i in {0..3} 
do
	echo $i
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset polarity --memory_size 20 --embedding_size 50 --hops $i --encoder avg --epochs 5 --batch_size 32 \
	--class_size 2  >./results/log_polarity_avg_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset polarity --memory_size 20 --embedding_size 50 --hops $i --encoder rnn --epochs 5 --batch_size 32 \
	--class_size 2 	>./results/log_polarity_rnn_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset polarity --memory_size 20 --embedding_size 50 --hops $i --encoder cnn --epochs 5 --batch_size 32 \
	--class_size 2 >./results/log_polarity_cnn_mem_20_hub_$i".txt"

done 

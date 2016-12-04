#! /bin/bash

for i in {0..5} 
do
	echo $i
	CUDA_VISIBLE_DEVICES=3 python main.py --dataset tweets --memory_size 20 --embedding_size 50 --hops $i --encoder avg --epochs 10 --batch_size 128 \
	--class_size 2  >./results/log_tweets_avg_mem_20_hub_$i".txt.tmp"
#	CUDA_VISIBLE_DEVICES=3 python main.py --dataset tweets --memory_size 20 --embedding_size 50 --hops $i --encoder rnn --epochs 10 --batch_size 128 \
#	--class_size 2 	>./results/log_tweets_rnn_mem_20_hub_$i".txt"
#	CUDA_VISIBLE_DEVICES=3 python main.py --dataset tweets --memory_size 20 --embedding_size 50 --hops $i --encoder cnn --epochs 10 --batch_size 128 \
#	--class_size 2 >./results/log_tweets_cnn_mem_20_hub_$i".txt"

done 

#! /bin/bash

for i in {0..3} 
do
	echo $i
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset 20ng --memory_size 20 --embedding_size 50 --hops $i --encoder avg --epochs 5  --sentence_size 100 \
        --class_size 20 >./results/log_20ng_avg_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset 20ng --memory_size 20 --embedding_size 50 --hops $i --encoder rnn --epochs 5  --sentence_size 100 \
	--class_size 20 >./results/log_20ng_rnn_mem_20_hub_$i".txt"
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset 20ng --memory_size 20 --embedding_size 50 --hops $i --encoder cnn --epochs 5  --sentence_size 100\
	--class_size 20 >./results/log_20ng_cnn_mem_20_hub_$i".txt"

done 

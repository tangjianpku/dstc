#! /bin/bash

##### finding the K-nearest neighbors

doc_embedding=doc.txt
knn_graph=knn_graph.txt
knn_graph_fmt=knn_graph_fmt.txt
train_size=61479
neighbor_size=20

./KNNLargeVis -input $doc_embedding -output $knn_graph

perl fmt.pl $knn_graph $knn_graph_fmt $train_size $neighbor_size 



#!/bin/bash

datasets=("sqa") # select from "hybridqa" "wtq" "tabfact"
n_clusters=(3) #  select from 5 10 20
ks=(50) # select from 100 150 200
embedding_method=contriever # select from contriever e5 sentencetransformer




for k in "${ks[@]}"; do
    for n_cluster in "${n_clusters[@]}"; do
        for dataset in "${datasets[@]}"; do

            mkdir -p "logs/${dataset}/"
            log_file="logs/${dataset}/${dataset}_cluster_k${k}_n${n_cluster}_${embedding_method}.log"

            echo "Logging to $log_file" > "$log_file"
            > "$log_file"

            echo "Clustering table in Dataset: $dataset with k=$k and n_clusters=$n_cluster" | tee -a "$log_file"

            python cluster/table_cluster_$embedding_method.py --dataset $dataset --n_clusters $n_cluster --k $k >> "$log_file" 2>&1
        done
    done
done

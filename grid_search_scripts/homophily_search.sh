#!/bin/bash

dataset=$1

wd_sca_lst=(0.0 0.0001 0.001 0.01 0.1)
wd_fc1_lst=(0.0 0.0001 0.001)
wd_fc2_lst=(0.0 0.001 0.0001)
lr_sca_lst=(0.04 0.02 0.01 0.005)
lr_fc_lst=(0.01 0.005)
dropout_lst=(0.5 0.6 0.7)


for wd_sca in "${wd_sca_lst[@]}"; do
    for wd_fc1 in "${wd_fc1_lst[@]}"; do
        for wd_fc2 in "${wd_fc2_lst[@]}"; do
            for lr_sca in "${lr_sca_lst[@]}"; do
                for lr_fc in "${lr_fc_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        echo "Running"
                        python pegfan_node_class.py --data $dataset --feat_type homophily --layer 3 --w_att $wd_sca --w_fc2 $wd_fc2 --w_fc1 $wd_fc1 --dropout $dropout --lr_fc $lr_fc --lr_att $lr_sca --layer_norm 1 --dev 0 --type c --h 8
                    done 
                done 
            done
        done
    done 
done 


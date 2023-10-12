import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import ShuffleSplit
from utils import sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor,sys_normalized_adjacency_i
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.io as io
import torch
import math

def load(datastr, table = 1):
    g = io.loadmat('synthetic/'+datastr+'.mat')['A']
    if table == 1:
        features = io.loadmat('synthetic/features_table_1.mat')['X3']
    else:
        features = io.loadmat('synthetic/features_table_2.mat')['X1']
    
    n = g.shape[0]//4
    
    train_mask = np.array([False for x in range(4*n)])
    val_mask = np.array([False for x in range(4*n)])
    test_mask = np.array([False for x in range(4*n)])
    
    labels = io.loadmat('synthetic/labels_00.mat')['labels']
    temp_label = []
    for i in range(labels.shape[0]):
        temp_label.append(labels[i][0])
    labels = np.array(temp_label)
    
    train_num = math.floor(n *0.48)
    val_num = math.floor(n*0.32)
    test_num = n - train_num - val_num
    
    for i in range(train_num):
        for j in range(4):
            train_mask[n*j+i]=True
    
    for i in range(train_num,train_num+val_num):
        for j in range(4):
            val_mask[n*j+i]=True
    
    for i in range(train_num+val_num,train_num+val_num+test_num):
        for j in range(4):
            test_mask[n*j+i]=True
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    adj = sys_normalized_adjacency(g)
    adj_i = sys_normalized_adjacency_i(g)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)

    return g, adj,adj_i, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse.csgraph import shortest_path
from sknetwork.hierarchy import Paris, LouvainHierarchy, Ward, cut_balanced
import numpy as np
import scipy.sparse as sp
import scipy.io as io
import math
import torch
import os
import pickle

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_spatial_partitions(adj, h = 4):
    
    print("=======Generating hierachical partitions======")

    n = adj.shape[0]
    cluster_size_bound = h
    
    paris = Ward()
    partitions= []
    partitions.append([i for i in range(n)])
    prev_cluster_set = {}
    for i in range(n):
        temp_set = set()
        temp_set.add(i)
        prev_cluster_set[i] = temp_set
    prev_adj = adj
    while(prev_adj.shape[0]>cluster_size_bound and np.abs(np.sum(prev_adj))>1e-6):
        print("Current graph size：",prev_adj.shape)
        dendrogram = paris.fit_transform(prev_adj)
        cluster_id = cut_balanced(dendrogram, cluster_size_bound)
        cluster_num = max(cluster_id) + 1
        
        temp_cluster_set = {}
        temp_cluster_list = {}
        for j in range(len(cluster_id)):
            temp_id = cluster_id[j]
            if temp_id not in temp_cluster_set.keys():
                temp_cluster_set[temp_id] = prev_cluster_set[j]
                temp_cluster_list[temp_id] = [j]
            else:
                temp_cluster_set[temp_id] = temp_cluster_set[temp_id].union(prev_cluster_set[j])
                temp_cluster_list[temp_id].append(j)
        
        prev_cluster_set = temp_cluster_set
        temp_adj = np.zeros((cluster_num,cluster_num))
        for j in range(cluster_num):
            for k in range(j+1,cluster_num):
                edge_weight = 0.0
                for p in temp_cluster_list[j]:
                    for q in temp_cluster_list[k]:
                        edge_weight += prev_adj[p][q]
                temp_adj[j][k]=temp_adj[k][j] = edge_weight
        prev_adj = temp_adj
        
        temp_partition = [0 for j in range(n)]
        for j in range(cluster_num):
            temp_list = list(temp_cluster_set[j])
            for p in temp_list:
                temp_partition[p] = j
        print("Current level cluster num:",max(temp_partition)+1)
        
        partitions.append(temp_partition)
    
    print("=======Generation completed=======")
        
    return partitions

def one_step_refinement(coarse, fine, Partition_previous):
    label = np.array(coarse)
    num = np.max(label) + 1
    
    Partition = {}
    for i in range(num):
        Partition["%d"%(i)] = {}
        
    label = np.array(fine)
    
    num = 0
    for i in range(len(Partition_previous)):
        for j in range(len(Partition_previous["%d"%(i)])):
            
            parent = Partition_previous["%d"%(i)]["%d"%(j)]
            parentlabel = label[parent]
            
            num_parent = np.unique(parentlabel)
            
            
            for k in range(len(num_parent)):
                indx = np.where(parentlabel == num_parent[k])
                indx = indx[0].tolist()
                A = np.array(parent)
                # print(k)
                # print(num)
                Partition["%d"%(num)]["%d"%(k)] = A[indx].tolist()
                
            num = num + 1
            
    return Partition
        
def step_one_partition(partition1):
    
    label = np.array(partition1)
    
    Partition = {}
    Partition["%d"%(0)] = {}
    
    num_clusters = np.max(label) + 1
    for j in range(num_clusters):
        indx = np.where(label == j)
        indx = indx[0].tolist()
        Partition["0"]["%d"%(j)] = indx
    
    return Partition
    
def partition_matrix2partition_list(PartitionMatrix):
    
    Partitionlist = []
    
    partition1 = PartitionMatrix[0]
    Partition = step_one_partition(partition1)
    Partitionlist.append(Partition)
    
    for i in range(1, len(PartitionMatrix)):
        coarse = PartitionMatrix[i-1]
        fine = PartitionMatrix[i]
        Partition_previous = Partitionlist[i-1]
        Partition = one_step_refinement(coarse, fine, Partition_previous)
        
        Partitionlist.append(Partition)
        
    
    return Partitionlist[::-1]

def matrix_A(d):
    if d == 1:
        A = np.array([[1]])
    else:
        n_rows = int(d * (d-1)/2)
        
        A = np.zeros((n_rows, d))
        
        row = 0
        i = 0
        while i < d:
            for j in range(i+1, d):
                A[row,i] = 1
                A[row,j] = -1
                row = row + 1
            i = i + 1

    return A

# output A which generate tight frame with frame bound 1
def normalize_framebound(A):
    C = np.matmul(A,A.T)
    D = np.matmul(C,A)
    A = A / math.sqrt(D[0,0])
    C = np.matmul(A,A.T)
    D = np.matmul(C,A)
    
    if np.linalg.norm(D-A, ord=np.inf) > 1e-5:
        print(np.linalg.norm(D-A, ord=np.inf))
        print("There is something wrong.")
    
    return A


def step_one_framelet(children, n):
    ell = len(children)
    Phi = np.zeros((ell,n))
    
    if ell > 1:
        for i in range(ell):
            Phi[i, children["%d"%(i)]] = 1/math.sqrt(len(children["%d"%(i)]))
        Phi0 = np.sum(Phi/math.sqrt(ell), axis=0)
        
        A = matrix_A(ell)
        A = normalize_framebound(A)
        
        Psi = np.matmul(A,Phi)
    else:
        Phi[0, children["0"]] = 1
        Psi = Phi
        Phi0 = Phi
    
    return Phi0, Psi, Phi


def step_next_framelet(children, Phi, n):
    ell = len(children)
    
    if ell > 1:
        
        Phi0 = np.sum(Phi/math.sqrt(ell), axis=0)
        
        A = matrix_A(ell)
        #print("^^^",ell,A.shape)
        A = normalize_framebound(A)
        
        Psi = np.matmul(A,Phi)
    else:
        Psi = Phi
        Phi0 = Phi
    
    return Phi0, Psi, Phi

def Framelet(Partition, n):
    k = len(Partition)
    PSI = []
    PHI0 = []
    PHI = []
    
    indx = []
    
    for i in range(k):
        
        if i == 0:
            Partitioni = Partition[i]
            for j in range(len(Partitioni)):
                children = Partitioni["%d"%(j)]
                ell = len(children)
                if ell > 1:
                    Phi0, Psi, Phi = step_one_framelet(children, n)
                    
                    PHI0.append(Phi0)
                    PSI.append(Psi)
                    
                else:
                    Phi0, Psi, Phi = step_one_framelet(children, n)
                    
                    PHI0.append(Phi0)
                    #PSI.append(Psi)
        else:
            Partitioni = Partition[i]
            leng = []
            for j in range(len(Partitioni)):
                if j == 0:
                    ell = len(Partitioni["%d"%(j)])
                    leng.append(ell)
                else:
                    ell = len(Partitioni["%d"%(j)])
                    ell = ell + leng[j-1]
                    leng.append(ell)
                
            for j in range(len(Partitioni)):
                children = Partitioni["%d"%(j)]
                ell = len(children)
                
                Phi = np.zeros((ell, n))
                for i in range(ell):
                    if j == 0:
                        Phi[i,:] = PHI[0+i]
                    else:
                        Phi[i,:] = PHI[leng[j-1]+i]
                
                if ell > 1:
                    Phi0, Psi, Phi = step_next_framelet(children, Phi, n)
                    
                    PHI0.append(Phi0)
                    PSI.append(Psi)
                    
                else:
                    Phi0, Psi, Phi = step_next_framelet(children, Phi, n)
                    
                    PHI0.append(Phi0)
        
        indx.append(len(PSI))
                    
        PHI = PHI0
        PHI0 = []
    
    return PHI, PSI, indx

def foo(x):
    pre = x.shape[0]
    x = sp.csr_matrix(x,dtype=np.float32)
    cur = x.todense().shape[0]
    x = sparse_mx_to_torch_sparse_tensor(x)

    return x

def get_spatial_framelets_list(adj, dataset, h = 4):
    file_path = 'framelets/' + str(h) + '/' + dataset + '.pickle'
    if os.path.exists(file_path):
        with open(file_path, "rb") as fp:
            FrameletMatrix = pickle.load(fp)
        framelets_list = []
        framelets_T_list = []
        layers = len(FrameletMatrix)
        for i in range(layers):
            framelets_list.append(FrameletMatrix[i])
            framelets_T_list.append(torch.transpose(FrameletMatrix[i], 0, 1))
    else:
        partitions = get_spatial_partitions(adj, h)

        partition_num = len(partitions)
        n = len(partitions[0])
        partitions.reverse()

        processed_partitions = partition_matrix2partition_list(partitions)
        
        PHI, PSI, indx = Framelet(processed_partitions, n)
        
        PHI = [np.array(x) for x in PHI]
        Lowpass = np.zeros((len(PHI),PHI[0].shape[0]))
        
        for i in range(len(PHI)):
            Lowpass[i,:] = PHI[i]
        
        
        cnt = []
        for i in range(len(PSI)):
            cnt.append(PSI[i].shape[0])
        PSI_tot = np.sum(cnt)
        Highpass = np.zeros((PSI_tot,PSI[0].shape[1]))
        temp_cnt = 0
        
        temp_idx = [x for x in range(PSI[0].shape[1])]
        for i in range(len(PSI)):
            Highpass[np.ix_([x for x in range(temp_cnt,temp_cnt+cnt[i])],temp_idx)] = PSI[i]
            temp_cnt += cnt[i]
        
        tempMatrix = np.vstack((Lowpass, Highpass))
        print("==========Framelet generation completed==========")
        print("Framelet matrix size:",tempMatrix.shape)
        print("l_2 Difference between F^t*F and I:",np.linalg.norm(np.transpose(tempMatrix).dot(tempMatrix)-np.eye(tempMatrix.shape[1])))
        print("l_infty Difference between F^t*F and I:",np.max(np.abs(np.transpose(tempMatrix).dot(tempMatrix)-np.eye(tempMatrix.shape[1]))))
        print("=================================================")

        PHI[0] = PHI[0][None, :]
    
        framelets_list = []
        framelets_T_list = []
        
        
        framelets_list.append(foo(PHI[0]))
        framelets_T_list.append(foo(np.transpose(PHI[0])))
        
        tot = 0
        for i in range(len(indx)):
            num = indx[i]-tot
            row_num = 0
            
            for j in range(num):
                row_num += PSI[tot+j].shape[0]
            
            temp_m = np.zeros((row_num, Highpass.shape[1]))
            temp_row_num = 0

            """ avoid using np.concatenate, which incurs heavy storage and computation burden""" 
            
            for j in range(num):
                id_x = [x for x in range(temp_row_num,temp_row_num + PSI[tot + j].shape[0])]
                id_y = [y for y in range(Highpass.shape[1])]
                
                temp_m[np.ix_(id_x,id_y)] = PSI[tot + j]
                temp_row_num+=PSI[tot + j].shape[0]
               
            framelets_list.append(foo(temp_m))
            framelets_T_list.append(foo(np.transpose(temp_m)))
            tot = indx[i]
        
        with open(file_path, "wb") as fp:
            pickle.dump(framelets_list, fp)

    return framelets_list, framelets_T_list


            

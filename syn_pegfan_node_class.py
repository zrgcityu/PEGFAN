from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
from framelets_utils import get_spatial_framelets_list
import synthetic_input
import uuid
import pickle
import scipy.sparse as sp
import os
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_att',type=float, default=0.0005, help='Weight decay scalar')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.02, help='Learning rate Scalar')
parser.add_argument('--feat_type',type=str, default='all', help='Type of features to be used')
parser.add_argument('--type',type=str, default='a', help='Channel type')
parser.add_argument('--h',type=int, default=4, help='Cluster-size upperbound')
parser.add_argument('--gamma',type=str,default='00')
parser.add_argument('--table',type=int,default=1)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
num_layer = args.layer
feat_type = args.feat_type

layer_norm = bool(int(args.layer_norm))
print("==========================")
print(f"Gamma: {args.gamma}")
print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
print(f"w_att:{args.w_att}, w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, lr_fc:{args.lr_fc}, lr_att:{args.lr_att}")


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'


def train_step(model,optimizer,labels,list_mat,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,list_mat,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,labels,list_mat,idx_test):
    #model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        
        return loss_test.item(),acc_test.item()


def train():
        
    adj_raw, adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = synthetic_input.load('A4_'+ args.gamma,args.table)
    features = features.to(device)
    
    framelets_list, framelets_T_list = get_spatial_framelets_list(np.asarray(adj_raw), 'A4_'+args.gamma)
    framelet_layer = len(framelets_list)

    
    
    adj = adj.to(device)
    adj_i = adj_i.to(device)
    list_mat = []
    #list_mat.append(features)
    no_loop_mat = features
    loop_mat = features
    
    if args.type != 'a':
        for ii in range(args.layer):
            no_loop_mat = torch.spmm(adj, no_loop_mat)
            loop_mat = torch.spmm(adj_i, loop_mat)
            list_mat.append(no_loop_mat)
            list_mat.append(loop_mat)


        # Select X and self-looped features 
        if feat_type == "homophily":
            #select_idx = [0] + [2*ll for ll in range(1,num_layer+1)]
            select_idx = [2*ll-1 for ll in range(1,num_layer+1)]
            list_mat = [list_mat[ll] for ll in select_idx]

        #Select X and no-loop features
        elif feat_type == "heterophily":
            #select_idx = [0] + [2*ll-1 for ll in range(1,num_layer+1)]
            select_idx = [2*ll-2 for ll in range(1,num_layer+1)]
            list_mat = [list_mat[ll] for ll in select_idx]
     
    if args.type == 'c':
        if feat_type == "homophily":
            x = torch.spmm(adj_i,features)
        elif feat_type == "heterophily":
            x = torch.spmm(adj,features)
    else:
        x = features
    
    """to get results of fsgnn on synthetic data, comment the following loop and select proper channel type"""

    for i in range(framelet_layer):
        framelets_list[i] = framelets_list[i].to(device)
        framelets_T_list[i] = framelets_T_list[i].to(device)
        coeff = torch.spmm(framelets_list[i],x)
        temp_x = torch.spmm(framelets_T_list[i],coeff)
        list_mat.append(temp_x)
    
    
    
    
    
    
    model = FSGNN(nfeat=num_features,
                nlayers=len(list_mat),
                nhidden=args.hidden,
                nclass=num_labels,
                dropout=args.dropout).to(device)


    optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
        {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
    ]

    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999
    best_acc_val = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,labels,list_mat,idx_train)
        loss_val,acc_val = validate_step(model,labels,list_mat,idx_val)
        test_out = test_step(model,labels,list_mat,idx_test)
        #Uncomment following lines to see loss and accuracy values
        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''        

        if loss_val < best:
            best = loss_val
            best_acc_val = acc_val
            #torch.save(model.state_dict(), checkpt_file)
            acc =  test_out[1]
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    return acc*100, best_acc_val


t_total = time.time()
accuracy_data, best_val = train()
print("###",accuracy_data,best_val)
print("Train cost: {:.4f}s".format(time.time() - t_total))
#print("Test acc.:{:.2f}".format(np.mean(acc_list)))

filename = f'syn_results/A4_{args.gamma}' + '_' + '.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    write_obj.write(f"{best_val:.3f}," +
                    f"{accuracy_data:.3f}\n")
#os.remove(checkpt_file)

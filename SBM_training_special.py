import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import warnings
import scipy as sp
import time, copy, pickle, matplotlib, os, sys, math, random, itertools
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, get_laplacian, to_dense_adj

from utils import *

# ------------------------SETUP------------------------

np.seterr(all="ignore")
sys.path.insert(0, os.path.abspath('../../'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)
    
# ------------------------CREATE DATA------------------------

n = 500
d = int(np.ceil(n/(np.log(n)**2)))
p = 0.5
q = 0.1

sizes = [int(n/2), int(n/2)]
probs = [[p,q], [q,p]]
g = nx.stochastic_block_model(sizes, probs)
data = tg.utils.from_networkx(g)
data = data.to(device)
gadj = tg.utils.to_dense_adj(data.edge_index).squeeze(0)

g_test = nx.stochastic_block_model(sizes, probs)
data_test = tg.utils.from_networkx(g_test)
data_test = data_test.to(device)
gadj_test = tg.utils.to_dense_adj(data_test.edge_index).squeeze(0)

assert torch.equal(gadj, gadj_test) == False, "Training and testing graphs are the same"

data.adj_matrix = gadj
data.edge_index_test = data_test.edge_index

EDGE_ATTR_2 = extract_eigen(1, gadj.unsqueeze(0))
EDGE_ATTR_2_test = extract_eigen(1, gadj_test.unsqueeze(0))

which_class = 1
ground_truth = torch.from_numpy(np.concatenate((np.zeros(int(n/2)), np.ones(int(n/2)))))
ground_truth_test = torch.from_numpy(np.concatenate((np.zeros(int(n/2)), np.ones(int(n/2)))))
data.ynew = ground_truth
data.y = ground_truth
data.y_test = ground_truth_test
idx = data.y == which_class

degree = tg.utils.degree(data.edge_index[1], n)

std_ = 0.1
mu_up = 20*std_*np.sqrt(np.log(n**2))/(2*np.sqrt(d))
mu_lb = 0.01*std_/(2*np.sqrt(d))

Nmus = 15
mus = torch.tensor(np.geomspace(mu_lb.item(), mu_up.item(), Nmus, endpoint=True)).to(device)

epochs = 5000
trials = 7

print ("---------CONFIG---------")
print (f"number of nodes: {n}")
print (f"p: {p}")
print (f"q: {q}")
print (f"number of mus: {Nmus}")
print (f"trials: {trials}")
print (f"epochs: {epochs}")

FIGURESAVEPATH = f"figures/SBM_varying_means_training_smallk_special"
FILEPICKLESAVEPATH = f"pickle/SBM_varying_means_training_eigen1_smallk_special"
METAPICKLESAVEPATH = f"pickle/SBM_varying_means_training_meta_smallk_special"

print (f"Figures can be found: {FIGURESAVEPATH}")
print (f"Model pickle files can be found: {FILEPICKLESAVEPATH}")
print (f"Meta pickle files can be found: {METAPICKLESAVEPATH}")
print ("------------------------\n\n\n")

# ------------------------TRAINING------------------------

test_acc_mlp_gat = []
test_acc_gatv3_eigen1 = []
test_acc_gcn = []

intra_gamma_eigen1 = []
inter_gamma_eigen1 = []
intra_gamma_std_eigen1 = []
inter_gamma_std_eigen1 = []
acc_intra_edges_all_eigen1 = []
acc_inter_edges_all_eigen1 = []
psi_intra_attn_00_eigen1 = []
psi_intra_attn_11_eigen1 = []        
psi_inter_attn_01_eigen1 = []
psi_inter_attn_10_eigen1 = []
psi_intra_attn_00_std_eigen1 = []
psi_intra_attn_11_std_eigen1 = []        
psi_inter_attn_01_std_eigen1 = []
psi_inter_attn_10_std_eigen1 = []

phi_intra_attn_00_retro = []
phi_intra_attn_11_retro = []        
phi_inter_attn_01_retro = []
phi_inter_attn_10_retro = []
phi_intra_attn_00_std_retro = []
phi_intra_attn_11_std_retro = []        
phi_inter_attn_01_std_retro = []
phi_inter_attn_10_std_retro = []        

for muidx, mu_ in enumerate(mus):

    sum_test_acc_gcn = 0
    sum_test_acc_mlp_gat = 0
    sum_test_acc_gatv3_eigen1 = 0

    sum_intra_gamma_eigen1 = 0
    sum_inter_gamma_eigen1 = 0
    sum_intra_gamma_std_eigen1 = 0
    sum_inter_gamma_std_eigen1 = 0
    sum_acc_intra_edges_all_eigen1 = 0
    sum_acc_inter_edges_all_eigen1 = 0
    sum_psi_intra_attn_00_eigen1 = 0
    sum_psi_intra_attn_11_eigen1 = 0        
    sum_psi_inter_attn_01_eigen1 = 0
    sum_psi_inter_attn_10_eigen1 = 0
    sum_psi_intra_attn_00_std_eigen1 = 0
    sum_psi_intra_attn_11_std_eigen1 = 0        
    sum_psi_inter_attn_01_std_eigen1 = 0
    sum_psi_inter_attn_10_std_eigen1 = 0
    
    sum_phi_intra_attn_00_retro = 0
    sum_phi_intra_attn_11_retro = 0        
    sum_phi_inter_attn_01_retro = 0
    sum_phi_inter_attn_10_retro = 0
    sum_phi_intra_attn_00_std_retro = 0
    sum_phi_intra_attn_11_std_retro =  0       
    sum_phi_inter_attn_01_std_retro = 0
    sum_phi_inter_attn_10_std_retro = 0
    
    weight_decay = 1.0e-3
    loss_tol = 1.0e-2

    for trial in range(trials):
        print (f"mu index: {muidx}/{len(mus)}, trial: {trial+1}/{trials}, current mu: {mu_}")
        
        # ------------------------NODE FEATURES------------------------

        X = torch.zeros((n, d))
        X[:int(n/2)] = -mu_
        X[int(n/2):] = mu_
        noise = std_ * torch.from_numpy(np.random.randn(n, d))
        X = X + noise
        X = X.float()

        X_test = torch.zeros((n, d))
        X_test[:int(n/2)] = -mu_
        X_test[int(n/2):] = mu_
        noise = std_ * torch.from_numpy(np.random.randn(n, d))
        X_test = X_test + noise
        X_test = X_test.float()
        
        assert torch.equal(X, X_test) == False, "Training and testing node features are the same"

        data.x = X
        data.x_test = X_test
        data = data.to(device)
        
        # ------------------------GCN------------------------

        loss, train_acc, test_acc = run_gcn(data, d=data.x.shape[1], out_d=1, device=device, weight_decay=weight_decay, loss_tol=loss_tol, epochs=epochs)
        print(f"GCN \t\t Loss: {loss:.7f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
        sum_test_acc_gcn += test_acc

        # ------------------------RetroGAT------------------------

        loss, train_acc, test_acc, model_mlp_gat = run_mlp_gat(data, d=data.x.shape[1], out_d=1, device=device, weight_decay=weight_decay, loss_tol=loss_tol, epochs=epochs)
        print(f"RetroGAT \t\t Loss: {loss:.7f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
        sum_test_acc_mlp_gat += test_acc
        
        logits, attn_weights, pair_pred = model_mlp_gat(data)
        phi_intra_weight_00, phi_intra_weight_11, _, phi_inter_weight_01, phi_inter_weight_10, _ = info_mlp_gat_scores_phi(attn_weights, idx)
        
        sum_phi_intra_attn_00_retro += np.asarray(phi_intra_weight_00).mean()
        sum_phi_intra_attn_11_retro += np.asarray(phi_intra_weight_11).mean()   
        sum_phi_inter_attn_01_retro += np.asarray(phi_inter_weight_01).mean()   
        sum_phi_inter_attn_10_retro += np.asarray(phi_inter_weight_10).mean()   

        sum_phi_intra_attn_00_std_retro += np.asarray(phi_intra_weight_00).std()
        sum_phi_intra_attn_11_std_retro += np.asarray(phi_intra_weight_11).std()       
        sum_phi_inter_attn_01_std_retro += np.asarray(phi_inter_weight_01).std() 
        sum_phi_inter_attn_10_std_retro += np.asarray(phi_inter_weight_10).mean()
        
        # ------------------------GATv3------------------------

        data.edge_attr = EDGE_ATTR_2.to(device)
        data.edge_attr_test = EDGE_ATTR_2_test.to(device)

        loss, train_acc, test_acc, model_GATv3 = run_GATv3(data, d=data.x.shape[1], out_d=1, k=1, device=device, weight_decay=weight_decay, loss_tol=loss_tol, epochs=epochs)
        print(f"GATv3 k=1 \t\t Loss: {loss:.7f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
        sum_test_acc_gatv3_eigen1 += test_acc

        logits, attn_weights, pair_pred = model_GATv3(data) 
        intra_weight, idx_intra, inter_weight, idx_inter = info_gatv3_gamma(attn_weights, idx)
        psi_intra_weight_00, psi_intra_weight_11, _, psi_inter_weight_01, psi_inter_weight_10, _ = info_gatv3_scores_psi(attn_weights, idx)

        sum_intra_gamma_eigen1 += np.asarray(intra_weight).mean()
        sum_inter_gamma_eigen1 += np.asarray(inter_weight).mean()
        sum_intra_gamma_std_eigen1 += np.asarray(intra_weight).std()
        sum_inter_gamma_std_eigen1 += np.asarray(inter_weight).std()
        acc_intra_edges, acc_inter_edges = pair_acc(attn_weights[0][0].shape[0], pair_pred, idx_intra, idx_inter, 0)
        sum_acc_intra_edges_all_eigen1 += acc_intra_edges.cpu()
        sum_acc_inter_edges_all_eigen1 += acc_inter_edges.cpu()

        sum_psi_intra_attn_00_eigen1 += np.asarray(psi_intra_weight_00).mean()
        sum_psi_intra_attn_11_eigen1 += np.asarray(psi_intra_weight_11).mean()   
        sum_psi_inter_attn_01_eigen1 += np.asarray(psi_inter_weight_01).mean()   
        sum_psi_inter_attn_10_eigen1 += np.asarray(psi_inter_weight_10).mean()   

        sum_psi_intra_attn_00_std_eigen1 += np.asarray(psi_intra_weight_00).std()
        sum_psi_intra_attn_11_std_eigen1 += np.asarray(psi_intra_weight_11).std()       
        sum_psi_inter_attn_01_std_eigen1 += np.asarray(psi_inter_weight_01).std() 
        sum_psi_inter_attn_10_std_eigen1 += np.asarray(psi_inter_weight_10).mean()  

        print ("\n")

    test_acc_gcn.append(sum_test_acc_gcn/trials)
    test_acc_mlp_gat.append(sum_test_acc_mlp_gat/trials)
    test_acc_gatv3_eigen1.append(sum_test_acc_gatv3_eigen1/trials)

    intra_gamma_eigen1.append(sum_intra_gamma_eigen1/trials)
    inter_gamma_eigen1.append(sum_inter_gamma_eigen1/trials)
    intra_gamma_std_eigen1.append(sum_intra_gamma_std_eigen1/trials)
    inter_gamma_std_eigen1.append(sum_inter_gamma_std_eigen1/trials)

    acc_intra_edges_all_eigen1.append(sum_acc_intra_edges_all_eigen1/trials)
    acc_inter_edges_all_eigen1.append(sum_acc_inter_edges_all_eigen1/trials)
    psi_intra_attn_00_eigen1.append(sum_psi_intra_attn_00_eigen1/trials)    
    psi_intra_attn_11_eigen1.append(sum_psi_intra_attn_11_eigen1/trials) 
    psi_inter_attn_01_eigen1.append(sum_psi_inter_attn_01_eigen1/trials) 
    psi_inter_attn_10_eigen1.append(sum_psi_inter_attn_10_eigen1/trials)
    psi_intra_attn_00_std_eigen1.append(sum_psi_intra_attn_00_std_eigen1/trials)    
    psi_intra_attn_11_std_eigen1.append(sum_psi_intra_attn_11_std_eigen1/trials) 
    psi_inter_attn_01_std_eigen1.append(sum_psi_inter_attn_01_std_eigen1/trials) 
    psi_inter_attn_10_std_eigen1.append(sum_psi_inter_attn_10_std_eigen1/trials)

    phi_intra_attn_00_retro.append(sum_phi_intra_attn_00_retro/trials)    
    phi_intra_attn_11_retro.append(sum_phi_intra_attn_11_retro/trials) 
    phi_inter_attn_01_retro.append(sum_phi_inter_attn_01_retro/trials) 
    phi_inter_attn_10_retro.append(sum_phi_inter_attn_10_retro/trials)
    phi_intra_attn_00_std_retro.append(sum_phi_intra_attn_00_std_retro/trials)    
    phi_intra_attn_11_std_retro.append(sum_phi_intra_attn_11_std_retro/trials) 
    phi_inter_attn_01_std_retro.append(sum_phi_inter_attn_01_std_retro/trials) 
    phi_inter_attn_10_std_retro.append(sum_phi_inter_attn_10_std_retro/trials)

    print ("\n--------------------------------------------------------------------------------\n")
    
# ------------------------SAVEING FILES------------------------    

allfiles2 = {
    "test_acc_gatv3_eigen1": test_acc_gatv3_eigen1,
    "intra_gamma_eigen1": intra_gamma_eigen1,
    "inter_gamma_eigen1": inter_gamma_eigen1,
    "intra_gamma_std_eigen1": intra_gamma_std_eigen1,
    "inter_gamma_std_eigen1": inter_gamma_std_eigen1,
    "acc_intra_edges_all_eigen1": acc_intra_edges_all_eigen1,
    "acc_inter_edges_all_eigen1": acc_inter_edges_all_eigen1,
    "psi_intra_attn_00_eigen1": psi_intra_attn_00_eigen1,
    "psi_intra_attn_11_eigen1": psi_intra_attn_11_eigen1,
    "psi_inter_attn_01_eigen1": psi_inter_attn_01_eigen1,
    "psi_inter_attn_10_eigen1": psi_inter_attn_10_eigen1,
    "psi_intra_attn_00_std_eigen1": psi_intra_attn_00_std_eigen1,
    "psi_intra_attn_11_std_eigen1": psi_intra_attn_11_std_eigen1,
    "psi_inter_attn_01_std_eigen1": psi_inter_attn_01_std_eigen1,
    "psi_inter_attn_10_std_eigen1": psi_inter_attn_10_std_eigen1,
}

metafiles = {
    "test_acc_gcn": test_acc_gcn,
    "test_acc_mlp_gat": test_acc_mlp_gat,
    "phi_intra_attn_00_retro": phi_intra_attn_00_retro,
    "phi_intra_attn_11_retro": phi_intra_attn_11_retro,
    "phi_inter_attn_01_retro": phi_inter_attn_01_retro,
    "phi_inter_attn_10_retro": phi_inter_attn_10_retro,
    "phi_intra_attn_00_std_retro": phi_intra_attn_00_std_retro,
    "phi_intra_attn_11_std_retro": phi_intra_attn_11_std_retro,
    "phi_inter_attn_01_std_retro": phi_inter_attn_01_std_retro,
    "phi_inter_attn_10_std_retro": phi_inter_attn_10_std_retro,
    "mus": mus.cpu()
}



print ("SAVING ALL ARRAYS")

SAVEALL(which_class, allfiles2, FILEPICKLESAVEPATH)
SAVEALL(which_class, metafiles, METAPICKLESAVEPATH)

allfiles_plotting = {}
allfiles_plotting.update(allfiles2)
allfiles_plotting.update(metafiles)

print ("Number of files:", len(allfiles_plotting.keys()))

my_plot(
    "SBM", 
    which_class, 
    allfiles_plotting, 
    FIGURESAVEPATH
)
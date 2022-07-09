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
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, get_laplacian, to_dense_adj

from my_retro_gat import RetroPhiLayer
from my_gatv3 import GATv3PsiLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model_GCN(torch.nn.Module):
    def __init__(self, d, out_d):
        super(Model_GCN, self).__init__()
        
        self.conv1 = GCNConv(d,out_d, bias=False)

    def forward(self, data, train=True):
        if train:
            x = data.x
            x = self.conv1(x, data.edge_index)
        else:
            x = data.x_test
            x = self.conv1(x, data.edge_index_test)
        
        return x.squeeze(-1)
    
class RetroGATPhi(torch.nn.Module):
    def __init__(self, d ,out_d):
        super(RetroGATPhi, self).__init__()
        
        self.conv1 = RetroPhiLayer(d, out_d, 2, 50, share_weights=True, bias=False)

    def forward(self, data, train=True):
        if train:
            x = data.x
            x, attn_weights, pair_pred = self.conv1(x, data.edge_index, return_attention_info=True)
        else:
            x = data.x_test
            x, attn_weights, pair_pred = self.conv1(x, data.edge_index_test, return_attention_info=True)
        
        return x.squeeze(-1), attn_weights, pair_pred    

def extract_eigen(k, edge_adj):
    # lap_idx, lap_wt = get_laplacian(edge_idx, normalization="sym")
    eigenvals, eigenvecs = torch.linalg.eig(edge_adj)
    top_eig = eigenvecs.squeeze(0)[:, 1:k+1]
    top_eig = torch.real(top_eig)

    return top_eig
    
class GATv3Psi(torch.nn.Module):
    def __init__(self, d, out_d, k):
        super(GATv3Psi, self).__init__()
        self.conv1 = GATv3PsiLayer(
            in_channels=d, 
            out_channels=out_d,
            num_eigenvectors=k,
            node_att_in_channels=2,
            node_att_out_channels=2,
            edge_att_in_channels = 2*k,
            edge_att_out_channels = 2,
            share_weights=True, 
            bias=False
        )

    def forward(self, data, train=True):
        if train:
            x = data.x
            x, attn_weights, pair_pred = self.conv1(x, edge_index=data.edge_index, edge_feat=data.edge_attr, return_attention_info=True)
        else:
            x = data.x_test
            x, attn_weights, pair_pred = self.conv1(x, data.edge_index_test, edge_feat=data.edge_attr_test, return_attention_info=True)
        
        return x.squeeze(-1), attn_weights, pair_pred    

@torch.no_grad()
def measure_accuracy(model, data):
    model.eval()

    logits = model(data) # forward operation
    preds = torch.sigmoid(logits) > 0.5

    test_logits = model(data, train=False) # forward operation
    test_preds = torch.sigmoid(test_logits) > 0.5
    
    # calculate training accuracy
    correct = preds == data.ynew
    train_acc = int(correct.sum()) / int(data.x.size(0))
    
    # calculate training accuracy
    correct = test_preds == data.y_test
    test_acc = int(correct.sum()) / int(data.x_test.size(0))
        
    return train_acc, test_acc 

def run_gcn(data, d, out_d, device, weight_decay, loss_tol, epochs):
    # Define the model
    model = Model_GCN(d, out_d=out_d).to(device)

    # Define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define the solver, check documentation in pytorch for how to set the learning rate.
    if weight_decay == None:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=weight_decay)

    # Test using the randomly initialized parameters.
    train_acc, test_acc = measure_accuracy(model, data)

    for epoch in range(1, epochs):
        loss = train(model, data, criterion, opt) # Performs an Adam step etc.
        train_acc, test_acc = measure_accuracy(model, data) # Test at each epoch
#         if loss <= 1.0e-2 or train_acc > 0.99:
        if loss <= loss_tol:
            break
#         print(f"q: {epoch:0.1f} | Loss: {loss:0.15f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
    return loss, train_acc, test_acc

def train_gat(model, data, criterion, opt):
    model.train()
    opt.zero_grad()
    logits, attn_weights, pair_pred = model(data) # does a forward computation
    loss = criterion(logits, data.ynew)
    loss.backward() # this computes the stochastic gradient (or whatever gradient you are using based on the solver)
    opt.step() # this updates the parameters using the gradient that has been computed above.
    return loss

def train(model, data, criterion, opt):
    model.train()
    opt.zero_grad()
    logits = model(data) # does a forward computation
    loss = criterion(logits, data.ynew)
    loss.backward() # this computes the stochastic gradient (or whatever gradient you are using based on the solver)
    opt.step() # this updates the parameters using the gradient that has been computed above.
    return loss
        
@torch.no_grad()
def measure_accuracy_gat(model, data):
    model.eval()

    logits, attn_weights, pair_pred = model(data) # forward operation
    preds = torch.sigmoid(logits) > 0.5

    test_logits, _, _ = model(data, train=False) # forward operation
    test_preds = torch.sigmoid(test_logits) > 0.5
    
    # calculate training accuracy
    correct = preds == data.ynew
    train_acc = int(correct.sum()) / int(data.x.size(0))
    
    # calculate training accuracy
    correct = test_preds == data.y_test
    test_acc = int(correct.sum()) / int(data.x_test.size(0))
        
    return train_acc, test_acc    

def run_GATv3(data, d, out_d, k, device, weight_decay, loss_tol, epochs):
    # Define the model
    model = GATv3Psi(d, out_d=out_d, k=k).to(device)

    # Define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define the solver, check documentation in pytorch for how to set the learning rate.
    if weight_decay == None:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    # Test using the randomly initialized parameters.
    train_acc, test_acc = measure_accuracy_gat(model, data)

    for epoch in range(1, epochs):
        loss = train_gat(model, data, criterion, opt) # Performs an Adam step etc.
        train_acc, test_acc = measure_accuracy_gat(model, data) # Test at each epoch
#         if loss <= 1.0e-2 or train_acc > 0.99:
        if loss <= loss_tol:
            break
#         print(f"q: {epoch:0.1f} | Loss: {loss:0.15f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
    return loss, train_acc, test_acc, model

def run_mlp_gat(data, d, out_d, device, weight_decay, loss_tol, epochs):
    # Define the model
    model = RetroGATPhi(d, out_d=out_d).to(device)

    # Define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Define the solver, check documentation in pytorch for how to set the learning rate.
    if weight_decay == None:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=weight_decay)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')

    # Test using the randomly initialized parameters.
    train_acc, test_acc = measure_accuracy_gat(model, data)

    for epoch in range(1, epochs):
        loss = train_gat(model, data, criterion, opt) # Performs an Adam step etc.
        train_acc, test_acc = measure_accuracy_gat(model, data) # Test at each epoch
#         if loss <= 1.0e-2 or train_acc > 0.99:
        if loss <= loss_tol:
            break
#         print(f"q: {epoch:0.1f} | Loss: {loss:0.15f} | Train: {train_acc:0.4f} | Test: {test_acc:0.4f}")
    return loss, train_acc, test_acc, model

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def info_mlp_gat_scores_phi(attn_weights, idx):
    intra_weight = []
    idx_intra = []
    inter_weight = []
    idx_inter = []
    
    intra_weight_00 = []
    intra_weight_11 = []
    
    inter_weight_01 = []
    inter_weight_10 = []

    weights = attn_weights[2] # get phi values

    idx_class0 = idx.nonzero(as_tuple=True)[0]
    idx_class1 = (~idx).nonzero(as_tuple=True)[0]
    
    for i in range(attn_weights[0].shape[1]):
        edge = [attn_weights[0][0][i].item(),attn_weights[0][1][i].item()]
        if ((edge[0] in idx_class0) and (edge[1] in idx_class0)):
            intra_weight_00.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class1)):
            intra_weight_11.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class0) and (edge[1] in idx_class1)):
            inter_weight_01.append(weights[i].item())
            idx_inter.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class0)):
            inter_weight_10.append(weights[i].item())
            idx_inter.append(i)
            
    return intra_weight_00, intra_weight_11, idx_intra, inter_weight_01, inter_weight_10, idx_inter

def info_gatv3_betas(attn_weights, idx):
    intra_weight = []
    idx_intra = []
    inter_weight = []
    idx_inter = []

    weights = attn_weights[4]
            
    return weights

def info_gatv3_gamma(attn_weights, idx):
    intra_weight = []
    idx_intra = []
    inter_weight = []
    idx_inter = []

    weights = attn_weights[1] # get gamma values

    idx_class0 = idx.nonzero(as_tuple=True)[0]
    idx_class1 = (~idx).nonzero(as_tuple=True)[0]
    
    for i in range(attn_weights[0].shape[1]):
        edge = [attn_weights[0][0][i].item(),attn_weights[0][1][i].item()]
        if ((edge[0] in idx_class0) and (edge[1] in idx_class0)) or ((edge[0] in idx_class1) and (edge[1] in idx_class1)):
            intra_weight.append(weights[i].item())
            idx_intra.append(i)
        else:
            inter_weight.append(weights[i].item())
            idx_inter.append(i)
            
    return intra_weight, idx_intra, inter_weight, idx_inter

def info_gatv3_scores_phi(attn_weights, idx):
    idx_intra = []
    idx_inter = []
    
    intra_weight_00 = []
    intra_weight_11 = []
    
    inter_weight_01 = []
    inter_weight_10 = []

    weights = attn_weights[3] # get phi scores

    idx_class0 = idx.nonzero(as_tuple=True)[0]
    idx_class1 = (~idx).nonzero(as_tuple=True)[0]
    
    for i in range(attn_weights[0].shape[1]):
        edge = [attn_weights[0][0][i].item(),attn_weights[0][1][i].item()]
        if ((edge[0] in idx_class0) and (edge[1] in idx_class0)):
            intra_weight_00.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class1)):
            intra_weight_11.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class0) and (edge[1] in idx_class1)):
            inter_weight_01.append(weights[i].item())
            idx_inter.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class0)):
            inter_weight_10.append(weights[i].item())
            idx_inter.append(i)
            
    return intra_weight_00, intra_weight_11, idx_intra, inter_weight_01, inter_weight_10, idx_inter

def info_gatv3_scores_psi(attn_weights, idx):
    idx_intra = []
    idx_inter = []
    
    intra_weight_00 = []
    intra_weight_11 = []
    
    inter_weight_01 = []
    inter_weight_10 = []

    weights = attn_weights[2] # get psi scores

    idx_class0 = idx.nonzero(as_tuple=True)[0]
    idx_class1 = (~idx).nonzero(as_tuple=True)[0]
    
    for i in range(attn_weights[0].shape[1]):
        edge = [attn_weights[0][0][i].item(),attn_weights[0][1][i].item()]
        if ((edge[0] in idx_class0) and (edge[1] in idx_class0)):
            intra_weight_00.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class1)):
            intra_weight_11.append(weights[i].item())
            idx_intra.append(i)
        elif ((edge[0] in idx_class0) and (edge[1] in idx_class1)):
            inter_weight_01.append(weights[i].item())
            idx_inter.append(i)
        elif ((edge[0] in idx_class1) and (edge[1] in idx_class0)):
            inter_weight_10.append(weights[i].item())
            idx_inter.append(i)
            
    return intra_weight_00, intra_weight_11, idx_intra, inter_weight_01, inter_weight_10, idx_inter

def pair_acc(n_edges, pair_pred, idx_intra, idx_inter, head):
    
    tmp = torch.zeros(n_edges).to(device)
    tmp[pair_pred[:,head].reshape(len(pair_pred)) > 0] = 1

    gt = torch.zeros(n_edges).to(device)
    gt[idx_intra] = 1

    acc_intra_edges = 1 - torch.sum(torch.abs(gt[idx_intra] - tmp[idx_intra]))/len(idx_intra)
    acc_inter_edges = 1 - torch.sum(torch.abs(gt[idx_inter] - tmp[idx_inter]))/len(idx_inter)
    print("\tHead: ", head, " acc intra edges: ", acc_intra_edges.item(), " acc inter edges: ", acc_inter_edges.item())
    
    return acc_intra_edges, acc_inter_edges

def my_plot(name, which_class, allfiles_plotting, parent_path, show_linear=False):
    
    allqs = allfiles_plotting["mus"].cpu()
    
    # check if folder exist
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        print ("Created directory\n\n")
    else:
        print ("Directory exists\n\n")
        
    XAXISLABEL = "Distance between means"
    
    # ---------------------------------------GAMMA---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["intra_gamma_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='Average $\gamma$, test, intra edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["inter_gamma_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='Average $\gamma$, test, inter edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["train_intra_gamma_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='Average $\gamma$, train, intra edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["train_inter_gamma_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='Average $\gamma$, train, inter edges, GATv3 k=1')

    plt.title(f"Class: {which_class} | Gammas")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('$\gamma$ value', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/gatv3_gammas_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')
    
    # ---------------------------------------GAMMA STD---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["intra_gamma_std_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='Stand. dev. $\gamma$, test, intra edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["inter_gamma_std_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='Stand. dev. $\gamma$, test, inter edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["train_intra_gamma_std_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='Stand. dev. $\gamma$, train, intra edges, GATv3 k=1')
    plt.plot(allqs, allfiles_plotting["train_inter_gamma_std_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='Stand. dev. $\gamma$, train, inter edges, GATv3 k=1')
    
    plt.title(f"SBM | Gammas Std Dev")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('Value', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/gatv3_gammas_std_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')
    
    fig = plt.figure(figsize=(16, 12), dpi=80)

    marker_intra = ['v','^','>','<']
    marker_inter = ['1','2','3','4']
    
    # ---------------------------------------NODE CLASSIFICATION TEST---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["test_acc_gcn"], linewidth=2, linestyle='-', marker='s', markersize=9, label='GCN')
    plt.plot(allqs, allfiles_plotting["test_acc_mlp_gat"], linewidth=2, linestyle='-', marker='+', markersize=9,label='GAT Retro')
    plt.plot(allqs, allfiles_plotting["test_acc_gatv3_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='GATv3 k=1')

    plt.title(f"SBM | Testing Node Classification")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('Classification accuracy', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/node_classification_test_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')

    # ---------------------------------------NODE CLASSIFICATION TRAIN---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["train_acc_gcn"], linewidth=2, linestyle='-', marker='s', markersize=9, label='GCN')
    plt.plot(allqs, allfiles_plotting["train_acc_mlp_gat"], linewidth=2, linestyle='-', marker='+', markersize=9,label='GAT Retro')
    plt.plot(allqs, allfiles_plotting["train_acc_gatv3_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='GATv3 k=1')

    plt.title(f"SBM | Training Node Classification")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('Classification accuracy', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/node_classification_train_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')
    
    # ---------------------------------------EDGE CLASSIFICATION TEST---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["acc_intra_edges_all_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='GATv3 k=1, intra edge classification')
    plt.plot(allqs, allfiles_plotting["acc_inter_edges_all_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='GATv3 k=1, inter edge classification')

    plt.title(f"SBM | Testing Edge Classification")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('Classification accuracy', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/edge_classification_test_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')

    # ---------------------------------------EDGE CLASSIFICATION TEST---------------------------------------
    
    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["train_acc_intra_edges_all_eigen1"], linewidth=2, linestyle='-', marker='*', markersize=9, label='GATv3 k=1, intra edge classification')
    plt.plot(allqs, allfiles_plotting["train_acc_inter_edges_all_eigen1"], linewidth=2, linestyle='-', marker='X', markersize=9, label='GATv3 k=1, inter edge classification')

    plt.title(f"SBM | Training Edge Classification")
    plt.grid(linestyle='dashed')
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('Classification accuracy', fontsize=20)
    # plt.show()
    
    fig.savefig(parent_path + "/edge_classification_train_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')
    
    # ---------------------------------------PSI k=1 TEST---------------------------------------

    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["psi_intra_attn_00_eigen1"], label="$i \in C_0, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="*")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["psi_intra_attn_00_eigen1"])-np.asarray(allfiles_plotting["psi_intra_attn_00_std_eigen1"]), np.asarray(allfiles_plotting["psi_intra_attn_00_eigen1"])+np.asarray(allfiles_plotting["psi_intra_attn_00_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["psi_intra_attn_11_eigen1"], label="$i \in C_1, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="X")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["psi_intra_attn_11_eigen1"])-np.asarray(allfiles_plotting["psi_intra_attn_11_std_eigen1"]), np.asarray(allfiles_plotting["psi_intra_attn_11_eigen1"])+np.asarray(allfiles_plotting["psi_intra_attn_11_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["psi_inter_attn_10_eigen1"], label="$i \in C_1, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="+")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["psi_inter_attn_10_eigen1"])-np.asarray(allfiles_plotting["psi_inter_attn_10_std_eigen1"]), np.asarray(allfiles_plotting["psi_inter_attn_10_eigen1"])+np.asarray(allfiles_plotting["psi_inter_attn_10_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["psi_inter_attn_01_eigen1"], label="$i \in C_0, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="o")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["psi_inter_attn_01_eigen1"])-np.asarray(allfiles_plotting["psi_inter_attn_01_std_eigen1"]), np.asarray(allfiles_plotting["psi_inter_attn_01_eigen1"])+np.asarray(allfiles_plotting["psi_inter_attn_01_std_eigen1"]), alpha=0.4)

    plt.grid(linestyle="dashed")
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('$\Psi$ Values', fontsize=20)
    plt.title("Testing $\Psi$ Values for GATv3 k=1")
    # plt.show()

    fig.savefig(parent_path + "/gatv3_psi_values_test_real_data_ansatz_"+name+"k1.pdf", dpi=400, bbox_inches='tight')

    # ---------------------------------------PSI k=1 TRAIN---------------------------------------

    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["train_psi_intra_attn_00_eigen1"], label="$i \in C_0, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="*")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_psi_intra_attn_00_eigen1"])-np.asarray(allfiles_plotting["train_psi_intra_attn_00_std_eigen1"]), np.asarray(allfiles_plotting["train_psi_intra_attn_00_eigen1"])+np.asarray(allfiles_plotting["train_psi_intra_attn_00_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_psi_intra_attn_11_eigen1"], label="$i \in C_1, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="X")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_psi_intra_attn_11_eigen1"])-np.asarray(allfiles_plotting["train_psi_intra_attn_11_std_eigen1"]), np.asarray(allfiles_plotting["train_psi_intra_attn_11_eigen1"])+np.asarray(allfiles_plotting["train_psi_intra_attn_11_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_psi_inter_attn_10_eigen1"], label="$i \in C_1, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="+")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_psi_inter_attn_10_eigen1"])-np.asarray(allfiles_plotting["train_psi_inter_attn_10_std_eigen1"]), np.asarray(allfiles_plotting["train_psi_inter_attn_10_eigen1"])+np.asarray(allfiles_plotting["train_psi_inter_attn_10_std_eigen1"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_psi_inter_attn_01_eigen1"], label="$i \in C_0, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="o")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_psi_inter_attn_01_eigen1"])-np.asarray(allfiles_plotting["train_psi_inter_attn_01_std_eigen1"]), np.asarray(allfiles_plotting["train_psi_inter_attn_01_eigen1"])+np.asarray(allfiles_plotting["train_psi_inter_attn_01_std_eigen1"]), alpha=0.4)

    plt.grid(linestyle="dashed")
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('$\Psi$ Values', fontsize=20)
    plt.title("Training $\Psi$ Values for GATv3 k=1")
    # plt.show()

    fig.savefig(parent_path + "/gatv3_psi_values_train_real_data_ansatz_"+name+"k1.pdf", dpi=400, bbox_inches='tight')
    
    # ---------------------------------------PHI k=1 TEST---------------------------------------

    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["phi_intra_attn_00_retro"], label="$i \in C_0, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="*")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["phi_intra_attn_00_retro"])-np.asarray(allfiles_plotting["phi_intra_attn_00_std_retro"]), np.asarray(allfiles_plotting["phi_intra_attn_00_retro"])+np.asarray(allfiles_plotting["phi_intra_attn_00_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["phi_intra_attn_11_retro"], label="$i \in C_1, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="X")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["phi_intra_attn_11_retro"])-np.asarray(allfiles_plotting["phi_intra_attn_11_std_retro"]), np.asarray(allfiles_plotting["phi_intra_attn_11_retro"])+np.asarray(allfiles_plotting["phi_intra_attn_11_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["phi_inter_attn_10_retro"], label="$i \in C_1, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="+")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["phi_inter_attn_10_retro"])-np.asarray(allfiles_plotting["phi_inter_attn_10_std_retro"]), np.asarray(allfiles_plotting["phi_inter_attn_10_retro"])+np.asarray(allfiles_plotting["phi_inter_attn_10_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["phi_inter_attn_01_retro"], label="$i \in C_0, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="o")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["phi_inter_attn_01_retro"])-np.asarray(allfiles_plotting["phi_inter_attn_01_std_retro"]), np.asarray(allfiles_plotting["phi_inter_attn_01_retro"])+np.asarray(allfiles_plotting["phi_inter_attn_01_std_retro"]), alpha=0.4)

    plt.grid(linestyle="dashed")
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('$\Phi$ Values', fontsize=20)
    plt.title("Testing $\Phi$ Values for RetroGAT")
    # plt.show()

    fig.savefig(parent_path + "/retro_phi_values_test_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')
    
    # ---------------------------------------PHI k=1 TRAIN---------------------------------------

    fig = plt.figure(figsize=(11, 7), dpi=80)

    plt.plot(allqs, allfiles_plotting["train_phi_intra_attn_00_retro"], label="$i \in C_0, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="*")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_phi_intra_attn_00_retro"])-np.asarray(allfiles_plotting["train_phi_intra_attn_00_std_retro"]), np.asarray(allfiles_plotting["train_phi_intra_attn_00_retro"])+np.asarray(allfiles_plotting["train_phi_intra_attn_00_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_phi_intra_attn_11_retro"], label="$i \in C_1, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="X")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_phi_intra_attn_11_retro"])-np.asarray(allfiles_plotting["train_phi_intra_attn_11_std_retro"]), np.asarray(allfiles_plotting["train_phi_intra_attn_11_retro"])+np.asarray(allfiles_plotting["train_phi_intra_attn_11_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_phi_inter_attn_10_retro"], label="$i \in C_1, j \in C_0$", linewidth=2, linestyle="-", markersize=9, marker="+")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_phi_inter_attn_10_retro"])-np.asarray(allfiles_plotting["train_phi_inter_attn_10_std_retro"]), np.asarray(allfiles_plotting["train_phi_inter_attn_10_retro"])+np.asarray(allfiles_plotting["train_phi_inter_attn_10_std_retro"]), alpha=0.4)

    plt.plot(allqs, allfiles_plotting["train_phi_inter_attn_01_retro"], label="$i \in C_0, j \in C_1$", linewidth=2, linestyle="-", markersize=9, marker="o")
    plt.fill_between(allqs, np.asarray(allfiles_plotting["train_phi_inter_attn_01_retro"])-np.asarray(allfiles_plotting["train_phi_inter_attn_01_std_retro"]), np.asarray(allfiles_plotting["train_phi_inter_attn_01_retro"])+np.asarray(allfiles_plotting["train_phi_inter_attn_01_std_retro"]), alpha=0.4)

    plt.grid(linestyle="dashed")
    plt.legend(fontsize=20)
    plt.xscale('log')
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(XAXISLABEL, fontsize=20)
    plt.ylabel('$\Phi$ Values', fontsize=20)
    plt.title("Training $\Phi$ Values for RetroGAT")
    # plt.show()

    fig.savefig(parent_path + "/retro_phi_values_train_real_data_ansatz_"+name+".pdf", dpi=400, bbox_inches='tight')

#     # ---------------------------------------BETA VALUES---------------------------------------

#     fig = plt.figure(figsize=(11, 7), dpi=80)

#     plt.plot(allqs, allfiles_plotting["betas_eigen1"], label="beta", linewidth=2, linestyle="-", markersize=9, marker="*")

#     plt.grid(linestyle="dashed")
#     plt.legend(fontsize=20)
#     plt.xscale('log')
#     plt.tick_params(axis='x', labelsize=18)
#     plt.tick_params(axis='y', labelsize=18)
#     plt.xlabel(XAXISLABEL, fontsize=20)
#     plt.ylabel('Beta values', fontsize=20)
#     plt.title("Beta values for k=1")
#     # plt.show()

#     fig.savefig(parent_path + "/beta_values_real_data_ansatz_"+name+"k1.pdf", dpi=400, bbox_inches='tight')

def get_graph_stats(edge_idx, y, x0, x1):
    assert edge_idx.size(0) == 2
    new_edge_idx = edge_idx.T
    np = 0
    nq = 0
    
    for idx, pair in enumerate(new_edge_idx):
        i, j = pair
        if y[i] == y[j]:
            np += 1
        else:
            nq += 1
    
    total_intra = math.comb(x0.size(0), 2) + math.comb(x1.size(0), 2)
    total_inter = (x0.size(0) * x1.size(0))
    
    p = np/total_intra
    q = nq/total_inter
        
    return p, q, np, nq

def SAVEALL(which_class, allfiles, parent_path):

    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
        print ("Created list saving directory\n\n")
    else:
        print ("List saving directory exists\n\n")

    for filename, array in allfiles.items():
        with open(parent_path + "/" + f"{filename}.pkl", 'wb') as f:
            pickle.dump(array, f)

    print (f"SAVED ALL FILES")

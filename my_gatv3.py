from typing import Optional, Tuple, Union
import numpy as np

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, get_laplacian, to_dense_adj
from torch_geometric.datasets import Planetoid
    
class GATv3Layer(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        node_att_in_channels: int,
        node_att_out_channels: int,
        edge_att_in_channels: int,
        edge_att_out_channels: int,
        num_eigenvectors: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_att_in_channels = node_att_in_channels
        self.node_att_out_channels = node_att_out_channels
        self.edge_att_in_channels = edge_att_in_channels
        self.edge_att_out_channels = edge_att_out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.k = num_eigenvectors

        self.node_att_in = Linear(node_att_in_channels, node_att_out_channels, bias=bias, weight_initializer='glorot') 

        self.node_att_out = Linear(node_att_out_channels, 1, bias=False, weight_initializer='glorot') 

        self.edge_att_in = Linear(edge_att_in_channels, edge_att_out_channels, bias=bias, weight_initializer='glorot')
        
        self.edge_att_out = Linear(edge_att_out_channels, 1, bias=False, weight_initializer='glorot')

        self.lin = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')


        self._alpha = None
        self._pair_pred = None
        self._psi_vals = None
        self._phi_vals = None

        self.reset_parameters()

    def reset_parameters(self):
        self.node_att_in.reset_parameters()
        self.node_att_out.reset_parameters()
        self.edge_att_in.reset_parameters()
        self.edge_att_out.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_feat, return_attention_info=None):
            
        x = self.lin(x)        
        
        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index=edge_index, x=x, size=None, E=edge_feat)

        alpha = self._alpha
        pair_pred = self._pair_pred
        psi_vals = self._psi_vals
        phi_vals = self._phi_vals

        self._alpha = None
        self._pair_pred = None
        self._psi_vals = None
        self._phi_vals = None

        out = out.mean(dim=1)

        if isinstance(return_attention_info, bool):
            assert alpha is not None
            assert pair_pred is not None 
            assert psi_vals is not None             
            assert phi_vals is not None
            return out, (edge_index, alpha, psi_vals, phi_vals), pair_pred
        else:
            return out

    def message(self, x_j, x_i, E_i, E_j, index, ptr, size_i):
                
        # [E, 1] -> r(LRelu(S(Wx)))
        cat = torch.cat([x_i, x_j], dim=1)
        cat2 = torch.cat([E_i, E_j], dim=1)
        
        node_att = self.node_att_out(F.leaky_relu(self.node_att_in(cat), 0.2)) 
        edge_att = self.edge_att_out(F.leaky_relu(self.edge_att_in(cat2), 0.2)) 

        attn = node_att + edge_att

        self._pair_pred = attn
        self._psi_vals = edge_att
        self._phi_vals = node_att
        
        gamma = tg.utils.softmax(attn, index, ptr, size_i) # [E, 1]
        msg = x_j * gamma
        
        self._alpha = gamma # edge-wise score
        
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class GATv3PsiLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        node_att_in_channels: int,
        node_att_out_channels: int,
        edge_att_in_channels: int,
        edge_att_out_channels: int,
        num_eigenvectors: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_att_in_channels = node_att_in_channels
        self.node_att_out_channels = node_att_out_channels
        self.edge_att_in_channels = edge_att_in_channels
        self.edge_att_out_channels = edge_att_out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.k = num_eigenvectors

        # self.node_att_in = Linear(node_att_in_channels, node_att_out_channels, bias=bias, weight_initializer='glorot') 

        # self.node_att_out = Linear(node_att_out_channels, 1, bias=False, weight_initializer='glorot') 

        self.edge_att_in = Linear(edge_att_in_channels, edge_att_out_channels, bias=bias, weight_initializer='glorot')
        
        self.edge_att_out = Linear(edge_att_out_channels, 1, bias=False, weight_initializer='glorot')

        self.lin = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')

        self._alpha = None
        self._pair_pred = None
        self._psi_vals = None
        # self._phi_vals = None

        self.reset_parameters()

    def reset_parameters(self):
        # self.node_att_in.reset_parameters()
        # self.node_att_out.reset_parameters()
        self.edge_att_in.reset_parameters()
        self.edge_att_out.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_feat, return_attention_info=None):
            
        x = self.lin(x)     
        
        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index=edge_index, x=x, size=None,  numnodes=x.size(0), E = edge_feat)

        alpha = self._alpha
        pair_pred = self._pair_pred
        psi_vals = self._psi_vals
        # phi_vals = self._phi_vals

        self._alpha = None
        self._pair_pred = None
        self._psi_vals = None
        # self._phi_vals = None

        out = out.mean(dim=1)

        if isinstance(return_attention_info, bool):
            assert alpha is not None
            assert pair_pred is not None 
            assert psi_vals is not None             
            # assert phi_vals is not None
            return out, (edge_index, alpha, psi_vals), pair_pred
        else:
            return out

    def message(self, x_j, x_i, E_i, E_j, index, ptr, size_i):
                
        # [E, 1] -> r(LRelu(S(Wx)))
        cat2 = torch.cat([E_i, E_j], dim=1)
        
        edge_att = self.edge_att_out(F.leaky_relu(self.edge_att_in(cat2), 0.2)) 

        attn = edge_att

        self._pair_pred = attn
        self._psi_vals = edge_att
        # self._phi_vals = node_att
        
        gamma = tg.utils.softmax(attn, index, ptr, size_i) # [E, 1]
        msg = x_j * gamma
        
        self._alpha = gamma # edge-wise score
        
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')                
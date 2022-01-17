import logging
import time
import itertools
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from scipy import sparse
from genome_utils import *


class GCNLayer(nn.Module):
    def __init__(self, adj, in_dim=1, channels=1, cuda=False, id_layer=None, centroids=None):
        super(GCNLayer, self).__init__()

        self.my_layers = []
        self.cuda = cuda
        self.nb_nodes = adj.shape[0]
        self.in_dim = in_dim
        self.channels = channels
        self.id_layer = id_layer
        self.centroids = centroids


        self.linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)
        self.eye_linear = nn.Conv1d(in_channels=self.in_dim, out_channels=int(self.channels/2), kernel_size=1, bias=True)

        self.adj = adj

        # edges = torch.LongTensor(np.array(self.adj.nonzero()))
        #
        # self.sparse_adj = torch.sparse.FloatTensor(edges, torch.FloatTensor(self.adj.data),
        #                                       torch.Size([self.nb_nodes, self.nb_nodes]))
        # # self.register_buffer('sparse_adj', sparse_adj)
        # self.sparse_adj = self.sparse_adj.cuda() if self.cuda else self.sparse_adj
        # self.centroids = self.centroids.cuda() if self.cuda else self.centroids
        # self.dense_adj = (self.sparse_adj.to_dense() > 0.).float()
        # self.dense_adj = self.dense_adj.cuda() if self.cuda else self.dense_adj

    def register_buf(self):
        self.register_buffer('sparse_adj', self.sparse_adj)

    def forward_adj(self, x, adj):
        #         adj = Variable(self.sparse_adj, requires_grad=False).to_dense()

        x = x.permute(0, 2, 1).contiguous()

        # 5129*5129 despite aggr set to default : 2

        eye_x = self.eye_linear(x)
        nb_examples, nb_channels, nb_nodes = x.size()
        x_1 = x.view(-1, nb_nodes)

        # bs*dim*n, n*n
        x = torch.matmul(x_1, adj)
        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)

        x = torch.cat([self.linear(x), eye_x], dim=1).contiguous()
        x = F.relu(x)
        x = x.permute(0, 2, 1).contiguous()

        return x

    def _adj_mul(self, x, D):
        nb_examples, nb_channels, nb_nodes = x.size()
        x_1 = x.view(-1, nb_nodes)

        # Needs this hack to work: https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
        #x = D.mm(x.t()).t()
        x = SparseMM(D)(x_1.t()).t()

        x = x.contiguous().view(nb_examples, nb_channels, nb_nodes)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        adj = Variable(self.sparse_adj, requires_grad=False)
        
        #5129*5129 despite aggr set to default : 2

        eye_x = self.eye_linear(x)
        
#         print("x:adj", x.shape, adj.shape)


        x = self._adj_mul(x, adj)


        linear_x = self.linear(x)
        


        x = torch.cat([self.linear(x), eye_x], dim=1).contiguous()
        x = F.relu(x)

        index = self.centroids
        x = torch.index_select(x, 2, self.centroids)

        x = x.permute(0, 2, 1).contiguous()

        return x


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    From: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class EmbeddingLayer(nn.Module):
    def __init__(self, nb_emb, emb_size=32):
        self.emb_size = emb_size
        super(EmbeddingLayer, self).__init__()
        self.emb_size = emb_size
        self.nb_emb = nb_emb
        self.emb = nn.Parameter(torch.rand(nb_emb, emb_size))
        self.reset_parameters()

    def forward(self, x):
        #print(x)
        #print(x.size)

        emb = x * self.emb
        #print(emb)
        #print(emb.shape)
        return emb

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.emb.size(1))
        self.emb.data.uniform_(-stdv, stdv)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, nb_attention_head=1):
        self.in_dim = in_dim
        self.nb_attention_head = nb_attention_head
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, nb_attention_head)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)

        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, self.nb_attention_head)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(1)  # normalizing

        x = x.view(nb_examples, nb_nodes, nb_channels)
        attn_applied = x.unsqueeze(-1) * attn_weights.unsqueeze(-2)
        attn_applied = attn_applied.sum(dim=1)
        attn_applied = attn_applied.view(nb_examples, -1)

        return attn_applied, attn_weights


class SoftPoolingLayer(nn.Module):
    def __init__(self, in_dim, nb_attention_head=10):
        self.in_dim = in_dim
        self.nb_attention_head = nb_attention_head
        super(SoftPoolingLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, self.nb_attention_head)
        self.temperature = 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)

        attn_weights = torch.exp(self.attn(x)*self.temperature)
        attn_weights = attn_weights.view(nb_examples, nb_nodes, self.nb_attention_head)
        attn_weights = attn_weights / attn_weights.sum(dim=1).unsqueeze(1)  # normalizing
        attn_weights = attn_weights.sum(dim=-1)

        return attn_weights.unsqueeze(-1)


class ElementwiseGateLayer(nn.Module):
    def __init__(self, in_dim):
        self.in_dim = in_dim
        super(ElementwiseGateLayer, self).__init__()
        self.attn = nn.Linear(self.in_dim, 1, bias=True)

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        x = x.view(-1, nb_channels)
        gate_weights = torch.sigmoid(self.attn(x))
        gate_weights = gate_weights.view(nb_examples, nb_nodes, 1)
        return gate_weights


class StaticElementwiseGateLayer(nn.Module):
    def __init__(self, in_dim):
        self.in_dim = in_dim
        super(StaticElementwiseGateLayer, self).__init__()
        self.attn = nn.Parameter(torch.zeros(50), requires_grad=True) + 1.

    def forward(self, x):
        nb_examples, nb_nodes, nb_channels = x.size()
        gate_weights = torch.sigmoid(self.attn)
        gate_weights = gate_weights.view(nb_nodes, 1)
        return gate_weights

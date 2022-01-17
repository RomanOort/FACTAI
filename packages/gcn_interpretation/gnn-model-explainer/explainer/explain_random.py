""" explain.py

    Implementation of the explainer. 
"""

import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import random

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb
import pickle

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils
import utils.accuracy_utils as accuracy_utils
import utils.noise_utils as noise_utils

import explainer.explain as explain
from scipy.sparse import coo_matrix

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

flips = 0.
incorrect_preds = 0
use_comb_mask = False
avg_add_edges = 0.
avg_removed_edges = 0.
global_noise_count = 0.
global_mask_dense_count = 0.
global_mask_density = 0.
avg_noise_diff = 0.
noise_diff_count = 0.
avg_adj_diff = 0.
noise_percent = 0.0



nbr_data = None

sub_label_nodes = pickle.load(open(
    "synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p",
    "rb"))['sub_label_nodes']

class ExplainerRandom(explain.Explainer):
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        train_idx,
        args,
        writer=None,
        print_training=False, # default false for runtime
        graph_mode=False,
        graph_idx=False,
        num_nodes = None,
        device='cpu',
    ):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer, print_training, graph_mode, graph_idx, num_nodes, device)

    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        pass


    def explain_nodes_gnn_stats(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):

        pass



    # GRAPH EXPLAINER
    def explain_graphs(self, args, graph_indices, test_graph_indices=None):
        stats = accuracy_utils.Stats("Random", self)
        for graph_idx in graph_indices:
            print("doing for graph index: ", graph_idx)
            if len(self.adj.shape) < 3:
                sub_adj = self.adj
            else:
                sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            if self.num_nodes is not None:
                sub_nodes = self.num_nodes[graph_idx]
            else:
                sub_nodes = None

            sub_adj = sub_adj.cpu().detach().numpy()
            sub_feat = sub_feat.cpu().detach().numpy()
            masked_adj = torch.rand(sub_adj.shape)
            masked_adj = masked_adj * sub_adj
            print(masked_adj.shape)
            print(masked_adj)

            imp_nodes = explain.getImportantNodes(masked_adj, 8)
            stats.update(masked_adj, imp_nodes, graph_idx)

        pass


    
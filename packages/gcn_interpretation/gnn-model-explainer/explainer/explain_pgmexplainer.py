""" explain_pgmexplainer.py

    Implementation of the PGMExplainer paper. 
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

from scipy.special import softmax
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils
import utils.accuracy_utils as accuracy_utils
import utils.noise_utils as noise_utils

import explainer.explain as explain

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

flips = 0.
incorrect_preds = 0

nbr_data = None


class ExplainerPgmExplainer(explain.Explainer):
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
        print_training=True,
        graph_mode=False,
        graph_idx=False,
        num_nodes = None,
        snorm_n = None, 
        snorm_e = None, 
        perturb_feature_list = None,
        perturb_mode = "mean", # mean, zero, max or uniform
        perturb_indicator = "diff", # diff or abs
        device='cpu'
    ):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer, print_training, graph_mode, graph_idx, num_nodes, device)
        # self.snorm_n = snorm_n
        # self.snorm_e = snorm_e
        
        self.perturb_feature_list = list(range(self.feat.shape[-1]))
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        
        # constants
        self.num_samples = 1000
        self.percentage = 50
        self.top_node = 4
        self.p_threshold = 0.05
        self.pred_threshold = 0.1


    '''
    Code taken directly from PGMExplainer repository, ported to our GCN model
    '''

    def perturb_features_on_node(self, feat, adj, num_nodes, feature_matrix, node_idx, random = 0):
        
        X_perturb = feature_matrix.clone()
        perturb_array = X_perturb[node_idx].clone().detach()
        epsilon = 0.05*np.max(feat.clone().detach().numpy(), axis = 0)
        seed = np.random.randint(2)

        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if i in self.perturb_feature_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = torch.mean(feature_matrix[:,i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = np.max(feature_matrix[:,i])
                        elif self.perturb_mode == "uniform":
                            perturb_array[i] = perturb_array[i] + np.random.uniform(low=-epsilon[i], high=epsilon[i])
                            if perturb_array[i] < 0:
                                perturb_array[i] = 0
                            elif perturb_array[i] > np.max(feat, axis = 0)[i]:
                                perturb_array[i] = np.max(feat, axis = 0)[i]

        
        X_perturb[node_idx] = perturb_array

        return X_perturb 
    
    def batch_perturb_features_on_node(self, feat, adj, num_nodes, num_samples, index_to_perturb,
                                            percentage, p_threshold, pred_threshold):

        X_torch = feat#torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float)
        E_torch = adj#torch.tensor(np.expand_dims(adj, axis=0), dtype=torch.float)
        pred, _ = self.model(X_torch.to(self.device), E_torch.to(self.device), batch_num_nodes=[num_nodes])
        pred = softmax(pred.detach().cpu().numpy())
        pred_label = np.argmax(pred)
        num_nodes = num_nodes
        Samples = [] 
        for iteration in range(num_samples):
            X_perturb = feat[0].clone().detach()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(feat, adj, num_nodes, X_perturb, node, random = latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch =  torch.tensor(np.expand_dims(X_perturb, axis=0), dtype=torch.float)
            pred_perturb, _ = self.model(X_perturb_torch.to(self.device), E_torch.to(self.device), batch_num_nodes=[num_nodes])
            pred_perturb = softmax(pred_perturb.detach().cpu().numpy())
            pred_change = np.max(pred) - pred_perturb[0][pred_label]
            sample.append(pred_change)
            Samples.append(sample)
        
        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)
        
        top = int(num_samples/8)

        top_idx = np.argsort(Samples[:,num_nodes])[-top:] 
        for i in range(num_samples):
            if i in top_idx:
                Samples[i,num_nodes] = 1
            else:
                Samples[i,num_nodes] = 0
            
        return Samples
    
    # NODE EXPLAINER
    def explain_nodes_gnn_stats(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):
        pass

    def explain_graph(self, feat, adj, num_nodes, sparsity=-1):
        num_samples = self.num_samples
        percentage = self.percentage
        top_node = self.top_node
        p_threshold = self.p_threshold
        pred_threshold = self.pred_threshold

        num_nodes = int(num_nodes)
        if top_node is None:
            top_node = int(num_nodes/20)

        # process varying sparsities
        if sparsity is not -1:
            top_node = int(num_nodes * sparsity / 4)
        #         Round 1
        Samples = self.batch_perturb_features_on_node(feat, adj, num_nodes, int(num_samples/2), range(num_nodes),percentage, 
                                                            p_threshold, pred_threshold)         
        
        data = pd.DataFrame(Samples)
        est = ConstraintBasedEstimator(data)
        
        p_values = []
        candidate_nodes = []
        
        target = num_nodes # The entry for the graph classification data is at "num_nodes"
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
        
        number_candidates = int(top_node*4)
        # process varying sparsities
        number_candidates = min(number_candidates, len(p_values) - 1)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]
        
        #         Round 2
        Samples = self.batch_perturb_features_on_node(feat, adj, num_nodes, num_samples, candidate_nodes, percentage, 
                                                            p_threshold, pred_threshold)          
        data = pd.DataFrame(Samples)
        est = ConstraintBasedEstimator(data)
        
        p_values = []
        dependent_nodes = []
        
        target = num_nodes
        for node in range(num_nodes):
            chi2, p = chi_square(node, target, [], data)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)

        top_p = np.min((top_node,num_nodes-1))

        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)
        
        return pgm_nodes, p_values, candidate_nodes
    
    # GRAPH EXPLAINER
    def explain_graphs(self, args, graph_indices, test_graph_indices=None):

        graph_indices = list(graph_indices)
        stats = accuracy_utils.Stats("PGMExplainer", self)

        graph_mode = self.graph_mode

        def get_graph_pred_changes(m_adj, m_x, node_mask=None):
            if graph_mode:
                logits_masked, _ = self.model(m_x, m_adj, batch_num_nodes=batch_num_nodes, node_mask=node_mask)
            else:
                logits_masked, _ = self.model(m_x, m_adj, batch_num_nodes=batch_num_nodes, new_node_idx=[node_idx_new])
                
            if not graph_mode:
                logits_masked = logits_masked[0][node_idx_new]
            else:
                logits_masked = logits_masked[0]
            pred_masked_label = np.argmax(logits_masked.cpu().detach().numpy())

            pred_change = pred_masked_label != pred_label
            pred = torch.softmax(logits, dim=-1)
            pred_masked = torch.softmax(logits_masked, dim=-1)
            pred_prob_change = pred[pred_label] - pred_masked[pred_label]
            return pred_change, pred_prob_change

        def get_graph_inputs(self, graph_idx):
            sub_nodes = None
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
            neighbors = np.asarray(range(self.adj.shape[0]))  # 1,2,3....num_nodes

            sub_adj = np.expand_dims(sub_adj, axis=0)
            sub_feat = np.expand_dims(sub_feat, axis=0)

            adj = torch.tensor(sub_adj, dtype=torch.float)
            x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
            label = torch.tensor(sub_label, dtype=torch.long)

            return adj, x, label, sub_nodes

        def get_node_inputs(self, node_idx):
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood_from_saved_data(
                node_idx, self.args.bmname
            )
            sub_adj = np.expand_dims(sub_adj, axis=0)
            sub_feat = np.expand_dims(sub_feat, axis=0)

            adj   = torch.tensor(sub_adj, dtype=torch.float)
            x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
            label = torch.tensor(sub_label, dtype=torch.long)
            
            return adj, x, label, node_idx_new

        
        nodeFidelityPredChange =  explain.AverageMeter(size=len(stats.nodesparsity))
        nodeFidelityPredProbChange =  explain.AverageMeter(size=len(stats.nodesparsity))
        nodeFidelitySparsity =  explain.AverageMeter(size=len(stats.nodesparsity))
        edgeFidelitySparsity =  explain.AverageMeter(size=len(stats.nodesparsity))
        numNodes =  explain.AverageMeter(size=len(stats.nodesparsity))
         # noise stats
        noise_iters = 1
        noise_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        noise_handlers = [noise_utils.NoiseHandler("PGMExplainer", self.model, self, noise_percent=x) for x in noise_range]
        noise_acc = {}
        for x in noise_range:
            for y in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                noise_acc[(x, y)] = (0, 0)
        
        times = []

        for graph_idx in graph_indices:
            print("Doing for: " + str(graph_idx))
            
            # evaluate original input
            if graph_mode:
                adj, x, label, sub_nodes = self.get_graph_inputs(graph_idx)
            else:
                adj, x, label, node_idx_new = self.get_node_inputs(node_idx)
                sub_nodes = None

            start = time.time()
            pgm_nodes, p_values, candidate_nodes = self.explain_graph(x, adj, sub_nodes, 0.5)

            end = time.time()
            times.append(end - start)
            print(np.mean(times), np.std(times))
            batch_num_nodes = [sub_nodes.cpu().numpy()] if sub_nodes is not None else None

            logits, _ = self.model(x, adj, batch_num_nodes=batch_num_nodes)

            if not graph_mode:
                logits = logits[0][node_idx_new]
            else:
                logits = logits[0]

            pred_label = np.argmax(logits.cpu().detach().numpy())
            
            pred_changes = []
            pred_prob_changes = []
            sparsities = []
            esparsities = []
            numnodes = []
            for max_nodes in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                sparsity = max_nodes / sub_nodes
                num_nodes = sub_nodes.cpu().numpy()
                top_node = max_nodes
                top_p = np.min((top_node,num_nodes-1))
                ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
                pgm_nodes_sp = list(ind_top_p)         

                mask = torch.ones(adj[0].shape)
                for idx in pgm_nodes_sp:
                    mask[idx, :] = 0
                    mask[:, idx] = 0
                fid_adj = torch.multiply(adj, mask)

                node_mask = torch.ones(fid_adj.shape[-1]).to(self.device)
                for idx in pgm_nodes_sp:
                    node_mask[idx] = 0

                pred_change, pred_prob_change = get_graph_pred_changes(fid_adj, x, node_mask=node_mask)

                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())
                sparsities.append(len(pgm_nodes_sp) / sub_nodes)
                esparsities.append(torch.sum(fid_adj) / torch.sum(adj))
                numnodes.append(len(pgm_nodes_sp))
                
                # p_values = torch.tensor(p_values, dtype=torch.float)
                # masked_adj = (torch.mul(adj[0], p_values) + torch.mul(p_values.T, adj[0])) / 2
                # masked_adj = masked_adj.numpy()
                # thresh_nodes = 15
                # imp_nodes = explain.getImportantNodes(masked_adj, 8)
                # stats.update(masked_adj, imp_nodes, graph_idx)

            if self.args.noise:
                for n_iter in range(noise_iters):
                    for nh in noise_handlers:
                        try:
                            feat_n = x[0].clone().detach().numpy()
                            adj_n = adj[0].clone().detach().numpy()
                            noise_feat, noise_adj = nh.sample(feat_n, adj_n, sub_nodes)
                        except: 
                            continue
                        noise_adj = noise_adj.unsqueeze(0)
                        noise_feat = noise_feat.unsqueeze(0)
                        pgm_nodes_n, p_values_n, candidate_nodes_n = self.explain_graph(noise_feat, noise_adj, sub_nodes, 0.5)
                        for max_nodes in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                            sparsity = max_nodes / sub_nodes
                            num_nodes = sub_nodes.cpu().numpy()
                            top_node = max_nodes
                            top_p = np.min((top_node,num_nodes-1))
                            ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
                            pgm_nodes_sp_gt = list(ind_top_p)  

                            num_nodes = sub_nodes.cpu().numpy()
                            top_node = max_nodes
                            top_p = np.min((top_node,num_nodes-1))
                            ind_top_p = np.argpartition(p_values_n, top_p)[0:top_p]
                            pgm_nodes_sp_n = list(ind_top_p) 

                            print(sparsity, nh.noise_percent, pgm_nodes_sp_gt, pgm_nodes_sp_n)
                        
                            # p_values = torch.tensor(p_values, dtype=torch.float32)
                            # masked_adj_n = (torch.mul(adj[0], p_values) + torch.mul(p_values.T, adj[0])) / 2
                            # masked_adj_n = masked_adj_n.numpy()
                            # nh.update(masked_adj, masked_adj_n, adj[0].clone().detach().numpy(), noise_adj[0].cpu().detach().numpy(), None, graph_idx)
                            acc = 0

                            for n in pgm_nodes_sp_gt:
                                if n in pgm_nodes_sp_n:
                                    acc += 1
                            noise_acc[(nh.noise_percent, max_nodes)] = (noise_acc[(nh.noise_percent, max_nodes)][0] + acc, noise_acc[(nh.noise_percent, max_nodes)][1] + len(pgm_nodes_sp_gt))

                            
            nodeFidelityPredChange.update(pred_changes)
            nodeFidelityPredProbChange.update(pred_prob_changes)
            nodeFidelitySparsity.update(sparsities)
            edgeFidelitySparsity.update(esparsities)
            numNodes.update(numnodes)

            print("Node Pred Change: {}".format(nodeFidelityPredChange.avg))
            print("Node Fidelity: {}".format(nodeFidelityPredProbChange.avg))
            print("Node Sparsity: {}".format(nodeFidelitySparsity.avg))
            print("Edge Sparsity: {}".format(edgeFidelitySparsity.avg))
            print("Num nodes: {}".format(numNodes.avg))

            print("noise results organized by: x axis as noise range, y axis as sparsity (check num nodes above to map to real node cnt")
            noise_output = ""
            for x in noise_range:
                for y in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    if noise_acc[(x, y)][0] == 0 or noise_acc[(x, y)][1] == 0:
                        noise_output += "0\t"
                    else:
                        noise_output += "{:.4f}".format(noise_acc[(x, y)][0] / noise_acc[(x, y)][1]) + "\t"

                noise_output += "\n"

            print(noise_output)
            
            # print(noise_output)
            # print(stats)
            # if self.args.noise:
            #     # for nh in noise_handlers:
            #     #     print(nh)
            #     print("SUMMARY")
            #     for nh in noise_handlers:
            #         print(nh.summary())
                
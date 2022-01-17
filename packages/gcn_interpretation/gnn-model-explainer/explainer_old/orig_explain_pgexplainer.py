""" explain_pgexplainer.py

    Implementation of the NN-parameterized GNNExplainer explainer. 
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

import explainer.explain as explain

from scipy.sparse import coo_matrix

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

flips = 0.
incorrect_preds = 0

nbr_data = None


class ExplainerPGExplainer(explain.Explainer):
    def __init__(
        self,
        model,
        adj,
        emb,
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
        device='cpu'
    ):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer, print_training, graph_mode, graph_idx, num_nodes,device)
        self.emb = emb
        

        # self.model = model
        # self.model.eval()
        # self.adj = adj
        # self.feat = feat
        # self.label = label
        # self.pred = pred
        # self.train_idx = train_idx
        # self.num_nodes = num_nodes
        # self.n_hops = args.num_gc_layers
        # self.graph_mode = graph_mode
        # self.graph_idx = graph_idx
        # self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        # self.args = args
        # self.writer = writer
        # self.print_training = print_training
    def extract_neighborhood_emb(self, node_idx, graph_idx=0):
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = super().extract_neighborhood(node_idx, graph_idx)
        sub_embs = self.emb[graph_idx, neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_embs

    def get_nbr_data(self, args, node_indices, graph_idx=0):
        torch_data = super().get_nbr_data(args, node_indices, graph_idx)

        emb_l = []
        for node_idx in node_indices:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_emb(node_idx, graph_idx)# get this
            emb_l.append(sub_emb)
            # TODO: port over rest of preprocessing, concat to torch_data, then use! 
            # Need embeddings for 
        torch_data['emb'] = emb_l

    def extract_neighborhood_from_saved_data_emb(self, node_idx, dataset):
        new_idx, adj, feat, label, nbrs = super().extract_neighborhood_from_saved_data(node_idx, dataset)
        offset_idx = node_idx
        emb = self.nbr_data['emb'][offset_idx]
        return new_idx, adj, feat, label, nbrs, emb


    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained 
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs


    def explain_nodes_gnn_stats(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):

        global nbr_data

        # if self.args.bmname == "syn1":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn1/torch_data.pth")
        # elif self.args.bmname == "syn2":
        #     # nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn2/torch_data.pth")
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn2/torch_data_all_binary.pth")

        # elif self.args.bmname == "syn3":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn3/torch_data_all.pth")
        # elif self.args.bmname == "syn4":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn4/torch_data_all.pth")

        nbr_data = self.get_nbr_data(args, graph_node_indices, graph_idx)

        # torch.save(nbr_data,"../data/syn2/torch_data_all_dense.pth")


        # masked_adjs = [
        #     self.explain(node_idx, graph_idx=graph_idx, model=model)
        #     for node_idx in node_indices
        # ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        avg_map = 0.0
        map_d = {}
        count_d = {}

        node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
            node_indices[0], self.args.bmname
        ) 

        explainer = ExplainModule(
            model = self.model, 
            num_nodes = self.adj.shape[1],
            emb_dims = sub_emb.shape[1] * 3,
            device = self.device,
            args = self.args
        )

        scheduler, optimizer = train_utils.build_optimizer(self.args, explainer.parameters())

        for epoch in range(self.args.num_epochs):
            avg_map = 0.0
            loss = 0
            for i, node_idx in enumerate(node_indices):

                # new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
                # new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood(idx)

                node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
                    node_idx, self.args.bmname
                ) 
                sub_label = np.expand_dims(sub_label, axis=0)
                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)
                sub_emb = np.expand_dims(sub_emb, axis=0)

                adj   = torch.tensor(sub_adj, dtype=torch.float)
                x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)
                emb   = torch.tensor(sub_emb, dtype=torch.float)

                pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)

                t0 = 0.5
                t1 = 4.99
                tmp = float(t0 * np.power(t1 / t0, epoch /self.args.num_epochs))

                pred, masked_adj = explainer((x[0], emb[0], adj[0], tmp, label, None), node_idx=node_idx_new, training=True)
                loss = loss + explainer.loss(pred, pred_label, node_idx=node_idx_new)
                
                new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood_from_saved_data(node_idx, self.args.bmname)
                labels = self.label[graph_idx][nbrs]

                masked_adj = masked_adj.clone().detach().numpy()

                # G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
                # print(masked_adjs[i].shape, i, idx, new_idx)
                if self.args.bmname == 'syn3':
                    map_score = accuracy_utils.getmAPsyn3(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])

                elif self.args.bmname == 'syn4':
                    # map_score = getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                    map_score = accuracy_utils.getmAPsyn4(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                elif self.args.bmname == 'syn1':
                    map_score = accuracy_utils.getmAPsyn1(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                elif self.args.bmname == 'syn2':
                    map_score = accuracy_utils.getmAPsyn2(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                else:
                    map_score = accuracy_utils.getmAPNodes(masked_adj, n_adj, labels, nbrs, new_idx)

                avg_map += map_score
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            avg_map_score = avg_map / len(node_indices)
            print("\n\navg map score: ", avg_map_score, "\n\n")
            # for k in map_d.keys():
            #     print("label: ", k, "  map: ", map_d[k] / count_d[k], "  count: ", count_d[k])
        # if args.fname != "":
        #     full_path = "./tuning/explain/" + self.args.fname
        #     file1 = open(full_path, "a")  # write mode
        #     file1.write(str(args.size_c) + " " + str(args.lap_c) + " " + str(args.ent_c) + "\n")
        #     file1.write(str(avg_map_score) + "\n")
        #     for k in map_d.keys():
        #         file1.write(str(k) + " " + str(map_d[k] / count_d[k]) + " " + str(count_d[k]) + " ")
        #     file1.write("\n\n")
        #     file1.close()

        return None

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, args, graph_indices):
        """
        Explain graphs.
        """

        graph_indices = list(graph_indices)
        masked_adjs = []
        rule_top4_acc = 0.
        rule_top6_acc = 0.
        rule_top8_acc = 0.

        rule_acc_count = 0.
        avg_mask_density = 0.
        mAP = 0.


        explainer = ExplainModule(
            model = self.model, 
            num_nodes = self.adj.shape[1],
            emb_dims = self.emb[0, :].shape[1] * 2, # TODO: fixme!
            device=self.device,
            args = self.args
        )
        
        explainer.load_state_dict(torch.load( "./ckpt/explainer_" + self.args.bmname + ".pth"))
        
        scheduler, optimizer = train_utils.build_optimizer(self.args, explainer.parameters())

        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        def shuffle_backward(l):
            l_out = [0] * len(l)
            for i, j in enumerate(l):
                l_out[j] = l[i]
            return l_out

        # orders = [[None] * (max(graph_indices)+1)][0]
        # rand_orders = [[None] * (max(graph_indices)+1)][0]
        # for graph_idx in graph_indices:
        #     orders[graph_idx] = list(range(self.adj.shape[1]))
        #     order = orders[graph_idx]
        #     rand_orders[graph_idx] = shuffle_forward(order)
        #     rand_order = rand_orders[graph_idx]
        #     self.feat[graph_idx, rand_order, :] = self.feat[graph_idx,order,:]
        #     self.adj[graph_idx, rand_order, :] = self.adj[graph_idx,order,:]
        #     self.adj[graph_idx, :, rand_order] = self.adj[graph_idx,:,order]

        for epoch in range(self.args.num_epochs):
            loss = 0
            logging_graphs=False


            masked_adjs = []

            np.random.shuffle(graph_indices)
            explainer.train()
            

            for graph_idx in graph_indices:
                # print("doing for graph index: ", graph_idx)

                # extract features and
                global flips
                global incorrect_preds
                """Explain a single node prediction
                """
                # index of the query node in the new adj
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
                neighbors = np.asarray(range(self.adj.shape[0])) #1,2,3....num_nodes
                sub_emb = self.emb[graph_idx, :]
                
                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)
                sub_emb = np.expand_dims(sub_emb, axis=0)

                

                adj   = torch.tensor(sub_adj, dtype=torch.float)
                x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)
                emb   = torch.tensor(sub_emb, dtype=torch.float)

                

                pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
                if pred_label != label.item():
                    incorrect_preds += 1
                # print("Graph predicted label: ", pred_label)
                t0 = 0.5
                t1 = 4.99
                tmp = float(t0 * np.power(t1 / t0, epoch /self.args.num_epochs))

                pred, masked_adj = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes))
                loss = loss + explainer.loss(pred, pred_label)

                masked_adj = masked_adj.clone().detach()

                # masked_adj[orders[graph_idx], :] = masked_adj[rand_orders[graph_idx], :]
                # masked_adj[:, orders[graph_idx]] = masked_adj[:, rand_orders[graph_idx]]

                #why node_idx is set to 0?
                if self.args.bmname == 'synthetic' or self.args.bmname == 'Mutagenicity':
                    masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
                    thresh_nodes = 15

                    if logging_graphs and epoch == 250:
                        G_denoised = io_utils.denoise_graph(
                            masked_adj,
                            0,
                            threshold_num=thresh_nodes,
                            feat=self.feat[graph_idx],
                            max_component=False,
                        )
                        print("denoising done")

                        label = self.label[graph_idx]
                        print("logging denoised graph ...")
                        if logging_graphs:
                            io_utils.log_graph(
                                self.writer,
                                G_denoised,
                                "graph/graphidx_{}_label={}".format(graph_idx, label),
                                identify_self=False,
                                nodecolor="feat",
                                args=self.args
                            )
                        print("logging denoised graph done")

                    if self.args.bmname == 'synthetic':
                        imp_nodes = explain.getImportantNodes(masked_adj, 8)
                        h_nodes = explain.getHnodes(graph_idx, 0)
                        h_nodes.extend(explain.getHnodes(graph_idx, 1))
                        mAP_s = accuracy_utils.getmAP(masked_adj, h_nodes)
                        mAP += mAP_s


                        top4_acc, top6_acc, top8_acc = accuracy_utils.getAcc(imp_nodes, h_nodes)
                        rule_top4_acc += top4_acc
                        rule_top6_acc += top6_acc
                        rule_top8_acc += top8_acc

                    rule_acc_count += 1.0
                    mask_density = np.sum(masked_adj) / np.sum(adj.cpu().numpy())
                    avg_mask_density += mask_density
                    label = self.label[graph_idx]
                    masked_adjs.append(masked_adj)

                else:
                    label = self.label[graph_idx]
                    masked_adjs.append(masked_adj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(explainer.state_dict(), "./ckpt/explainer_" + self.args.bmname + ".pth")
            if scheduler is not None:
                scheduler.step()

            # for graph_idx in graph_indices:
            #     self.feat[graph_idx, orders[graph_idx], :] = self.feat[graph_idx,rand_orders[graph_idx],:]
            #     self.adj[graph_idx, orders[graph_idx], :] = self.adj[graph_idx,rand_orders[graph_idx],:]
            #     self.adj[graph_idx, :, orders[graph_idx]] = self.adj[graph_idx,:,rand_orders[graph_idx]]

            if self.print_training:
                print(
                    "epoch: ",
                    epoch,
                    "; loss: ",
                    loss.item(),
                )

            # plot cmap for graphs' node features
            io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")
            if self.args.bmname =='synthetic':
                print(
                    "Rule wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(rule_top4_acc / rule_acc_count,
                                                                                rule_top6_acc / rule_acc_count,
                                                                                rule_top8_acc / rule_acc_count)
                )

                print(
                    "mAP score: {}".format(mAP / rule_acc_count)
                )


            print(
                "Average mask density: {}".format(avg_mask_density / rule_acc_count)
            )
            print("Total graphs optimized: ", len(graph_indices))

        #if self.args.bmname == 'Mutagenicity':
            #pickle.dump(masked_adjs, open("../data/Mutagenicity/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        #else:
            #pickle.dump(masked_adjs, open("../data/synthetic_data_3label_3sublabel/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        print("Flips: ", flips)
        print("Incorrect preds: ", incorrect_preds)

        return masked_adjs

class ExplainModule(nn.Module):
    def __init__(
        self, 
        model,
        num_nodes,
        emb_dims,
        device,
        args
    ):
        super(ExplainModule, self).__init__()
        self.device = device

        self.model = model.to(self.device)
        self.num_nodes = num_nodes

        input_dim = np.sum(emb_dims)

        self.elayers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        rc = torch.unsqueeze(torch.arange(0, self.num_nodes), 0).repeat([self.num_nodes,1]).to(torch.float32)
        # rc = torch.repeat(rc,[nodesize,1])
        self.row = torch.reshape(rc.T,[-1]).to(self.device)
        self.col = torch.reshape(rc,[-1]).to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.mask_act = 'sigmoid'
        self.args = args

        # self.mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        # std = nn.init.calculate_gain("relu") * math.sqrt(
        #     2.0 / (num_nodes + num_nodes)
        # )
        # with torch.no_grad():
        #     self.mask.normal_(1.0, std)
        #     # mask.clamp_(0.0, 1.0)


        self.coeffs = {
            "size": 0.01,#syn1, 0.009
            # "size": 0.012,#mutag
            # "size": 0.009,#synthetic graph
            #"size": 0.007, #0.04 for synthetic,#0.007=>40%,#0.06,#0.005,
            "feat_size": 0.0,
            "ent": 0.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
            "weight_decay": 0,
        }
    
    def _masked_adj(self,mask,adj):

        mask = mask.to(self.device)
        sym_mask = mask
        sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2

        # Create sparse tensor TODO: test and "maybe" a transpose is needed somewhere
        sparseadj = torch.sparse_coo_tensor(
            indices=torch.transpose(torch.cat([torch.unsqueeze(torch.Tensor(adj.row),-1), torch.unsqueeze(torch.Tensor(adj.col),-1)], dim=-1), 0, 1).to(torch.int64),
            values=adj.data,
            size=adj.shape
        )

        adj = sparseadj.coalesce().to_dense().to(torch.float32).to(self.device) #FIXME: tf.sparse.reorder was also applied, but probably not necessary. Maybe it needs a .coalesce() too tho?
        self.adj = adj

        masked_adj = torch.mul(adj,sym_mask)

        num_nodes = adj.shape[0]
        ones = torch.ones((num_nodes, num_nodes))
        diag_mask = ones.to(torch.float32) - torch.eye(num_nodes)
        diag_mask = diag_mask.to(self.device)
        return torch.mul(masked_adj,diag_mask)

    def mask_density(self, adj):
        mask_sum = torch.sum(self.masked_adj).cpu()
        adj_sum = torch.sum(adj)
        return mask_sum / adj_sum
    
    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""

        if training:
            debug_var = 0.0
            bias = 0.0
            random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(debug_var, 1.0-debug_var)
            random_noise = random_noise.to(self.device)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.clone() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        return gate_inputs

    def forward(self,inputs,node_idx=None, training=None):
        x, embed, adj, tmp, label, sub_nodes = inputs
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        adj = adj.to(self.device)
        # embed = embed.to('cpu')
        self.label = label
        self.tmp = tmp
        
        if node_idx is not None:
            adjs = coo_matrix(adj)
            if not isinstance(embed[adjs.row], torch.Tensor):
                f1 = torch.tensor(embed[adjs.row]).to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
                f2 = torch.tensor(embed[adjs.col]).to(self.device)
            else:
                f1 = embed[adjs.row].to(self.device)  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
                f2 = embed[adjs.col].to(self.device)
            selfemb = embed[node_idx] if isinstance(embed, torch.Tensor) else torch.tensor(embed[node_idx])
            selfemb = torch.unsqueeze(selfemb, 0).repeat([f1.shape[0], 1]).to(self.device)
            f12self = torch.cat([f1, f2, selfemb], dim=-1)
            h = f12self
        else:
            row = self.row.type(torch.LongTensor).to(self.device)#('cpu')
            col = self.col.type(torch.LongTensor).to(self.device)
            if not isinstance(embed[row], torch.Tensor):
                f1 = torch.Tensor(embed[row]).to(self.device)   # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
                f2 = torch.Tensor(embed[col]).to(self.device)
            else:
                f1 = embed[row]  # .to(self.device)  # <-- torch way to do tf.gather(embed, self.row)
                f2 = embed[col]

            h = torch.cat([f1, f2], dim=-1)

        h = h.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)


        self.values = torch.reshape(h, [-1])
        values = self.concrete_sample(self.values, beta=tmp, training=training)
        if node_idx is not None:
            sparsemask = torch.sparse.FloatTensor(
                indices=torch.transpose(torch.cat([torch.unsqueeze(torch.tensor(adjs.row),-1), torch.unsqueeze(torch.tensor(adjs.col),-1)], dim=-1), 0, 1).to(torch.int64).to(self.device),
                values=values,
                size=adjs.shape
            ).to(self.device)
        else:
            sparsemask = torch.sparse.FloatTensor(
                indices=torch.transpose(torch.cat([torch.unsqueeze(self.row, -1), torch.unsqueeze(self.col,-1)], dim=-1), 0, 1).to(torch.int64),
                values=values,
                size=[self.num_nodes,self.num_nodes]
            ).to(self.device)
        sym_mask = sparsemask.coalesce().to_dense().to(torch.float32)  #FIXME: again a reorder() is omitted, maybe coalesce
        self.mask = sym_mask

        # sym_mask = self.mask

        # sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2      # Maybe needs a .clone()
        sym_mask = (sym_mask + sym_mask.T) / 2
        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj
        x = torch.unsqueeze(x.detach().requires_grad_(True),0).to(torch.float32)        # Maybe needs a .clone()
        adj = torch.unsqueeze(self.masked_adj,0).to(torch.float32)
        x.to(self.device)
        if sub_nodes is not None:
            sub_num_nodes_l = [sub_nodes.cpu().numpy()]
        else:
            sub_num_nodes_l = None


        output, adj_att = self.model(x,adj,batch_num_nodes=sub_num_nodes_l)
        if node_idx is not None:
            # node mode
            res = self.softmax(output[0][node_idx, :])
        else:
            # graph mode
            res = self.softmax(output)
        return res, masked_adj

    def loss(self, pred, pred_label, node_idx=None):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        if node_idx is not None:
            # node mode
            pred_label_node = pred_label[node_idx]
            logit = pred[pred_label_node]
        else:
            # graph mode
            pred_reduce = pred[0]
            gt_label_node = self.label
            logit = pred_reduce[gt_label_node]
        pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.functional.relu(self.mask)
        if self.args.size_c > -0.001:
            size_loss = self.args.size_c * torch.sum(mask) #len(mask[mask > 0]) #torch.sum(mask)
        else:
            size_loss = self.coeffs["size"] * torch.sum(mask)

        # entropy
        mask = mask *0.99+0.005     # maybe a .clone()
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        if self.args.ent_c > -0.001:
            mask_ent_loss = self.args.ent_c * torch.mean(mask_ent)
        else:
            mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        # l2 norm
        l2norm = 0
        for name, parameter in self.elayers.named_parameters():
            if "weight" in name:
                l2norm = l2norm + torch.norm(parameter)
        l2norm = self.coeffs['weight_decay']*l2norm.clone()

        loss = pred_loss + size_loss + l2norm + mask_ent_loss

        return loss


""" explain_gnnexplainer.py

    Implementation of the GNNexplainer paper. 
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

# sub_label_nodes = pickle.load(open(
#     "synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p",
#     "rb"))['sub_label_nodes']

class ExplainerGnnExplainer(explain.Explainer):
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

    def explain_input(
        self, sub_feat, sub_adj, sub_nodes, sub_label, node_idx=0, graph_idx=None, graph_mode=False, unconstrained=False, model="exp"
    ):

        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        def shuffle_backward(l):
            l_out = [0] * len(l)
            for i, j in enumerate(l):
                l_out[j] = l[i]
            return l_out

        order = list(range(sub_adj.shape[1]))
        rand_order = shuffle_forward(order)
        
        # sub_feat[rand_order, :] = sub_feat[order,:]
        # sub_adj[rand_order, :] = sub_adj[order,:]
        # sub_adj[:, rand_order] = sub_adj[:,order]
        
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            node_idx_new = 0

        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            sub_num_nodes=sub_nodes
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            print("training..............")
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                single_subgraph_label = sub_label.squeeze()

                
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)

            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()
       
        # with open(os.path.join(label_dir, fname), 'wb') as outfile:
        #     np.save(outfile, np.asarray(masked_adj.copy()))
        #     print("Saved adjacency matrix to ", fname)
        print("masked adj: ", masked_adj.shape)
        # masked_adj[order, :] = masked_adj[rand_order, :]
        # masked_adj[:, order] = masked_adj[:, rand_order]
        return masked_adj


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp", noise_percent=0, gt_edges=None
    ):
        global flips
        global incorrect_preds

        global avg_add_edges
        global avg_removed_edges
        global global_noise_count
        global global_mask_dense_count
        global global_mask_density
        global avg_noise_diff
        global noise_diff_count
        global avg_adj_diff
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        sub_nodes = None
        if graph_mode:
            node_idx_new = node_idx   #set to 0, not used down the calls
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
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            if self.args.bmname == 'syn1' or self.args.bmname == 'syn2' or self.args.bmname == 'syn3' or self.args.bmname == 'syn4' or self.args.bmname == 'syn8':
                node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood_from_saved_data(
                    node_idx, self.args.bmname
                )


            sub_label = np.expand_dims(sub_label, axis=0)
        
        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        def shuffle_backward(l):
            l_out = [0] * len(l)
            for i, j in enumerate(l):
                l_out[j] = l[i]
            return l_out

        order = list(range(sub_adj.shape[1]))
        rand_order = shuffle_forward(order)
        
        # sub_feat[rand_order, :] = sub_feat[order,:]
        # sub_adj[rand_order, :] = sub_adj[order,:]
        # sub_adj[:, rand_order] = sub_adj[:,order]
        
        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        if self.args.noise_percent > 0 and self.args.noise:
            if gt_edges is None:
                h_nodes = []
                h_nodes.extend(sub_label_nodes[graph_idx,0,0,:].tolist())
                h_nodes.extend(sub_label_nodes[graph_idx,1,0,:].tolist())
                noise_orig_adj, added_edges, removed_edges = noise_utils.addNoiseToGraph(sub_adj[0], sub_feat[0], h_nodes, sub_nodes, self.args.noise_percent)
            else:
                noise_orig_adj, added_edges, removed_edges = noise_utils.addNoiseToGraphEdges(sub_adj[0], sub_feat[0], gt_edges, sub_nodes, self.args.noise_percent)

            adj_diff = np.sum(np.abs(sub_adj[0]-noise_orig_adj))
            sub_adj = noise_orig_adj
            sub_adj = np.expand_dims(sub_adj, axis=0)
            adj_diff = np.sum(np.abs(sub_adj[0]-noise_orig_adj))
            avg_adj_diff += adj_diff
            avg_add_edges += added_edges
            avg_removed_edges += removed_edges

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            if pred_label != label.item():
                incorrect_preds += 1
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            sub_num_nodes=sub_nodes
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            print("training..............")
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    # if epoch % 25 == 0:
                    #     explainer.log_mask(epoch)
                    #     explainer.log_masked_adj(
                    #         node_idx_new, epoch, label=single_subgraph_label
                    #     )
                    #     explainer.log_adj_grad(
                    #         node_idx_new, pred_label, epoch, label=single_subgraph_label
                    #     )

                    if epoch == 0:
                        print("model.att: ", self.model.att)
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if graph_mode:
                if torch.argmax(ypred).item() != pred_label:
                    flips += 1.0
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()
        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'graph_idx_'+str(graph_idx)+'.npy')
        if self.graph_mode:
            label_dir = os.path.join(self.args.logdir, ("label_" +str(pred_label)))
        else:
            label_dir = os.path.join(self.args.logdir, "node_explain")
        # with open(os.path.join(label_dir, fname), 'wb') as outfile:
        #     np.save(outfile, np.asarray(masked_adj.copy()))
        #     print("Saved adjacency matrix to ", fname)
        print("masked adj: ", masked_adj.shape)
        # masked_adj[order, :] = masked_adj[rand_order, :]
        # masked_adj[:, order] = masked_adj[:, rand_order]
        return masked_adj


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


        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        avg_map = 0.0
        map_d = {}
        count_d = {}
        for i, idx in enumerate(node_indices):
            # new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            # new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood(idx)

            new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood_from_saved_data(idx, self.args.bmname)
            pred = np.argmax(self.pred[graph_idx][nbrs], axis=1)
            labels = self.label[graph_idx][nbrs]

            r = self.evaluate_interpretation(args, masked_adjs[i], False, node_idx=idx, mode='keep-mask')


            # G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            # print(masked_adjs[i].shape, i, idx, new_idx)
            if self.args.bmname == 'syn3':
                map_score = accuracy_utils.getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                h_edges = accuracy_utils.gethedgessyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])

            elif self.args.bmname == 'syn4':
                # map_score = getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                map_score = accuracy_utils.getmAPsyn4(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                h_edges = accuracy_utils.gethedgessyn4(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            
            elif self.args.bmname == 'syn1':
                map_score = accuracy_utils.getmAPsyn1(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                h_edges = accuracy_utils.gethedgessyn1(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            
            elif self.args.bmname == 'syn2':
                map_score = accuracy_utils.getmAPsyn2(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                h_edges = accuracy_utils.gethedgessyn2(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
            else:
                map_score = accuracy_utils.getmAPNodes(masked_adjs[i], n_adj, labels, nbrs, new_idx)

            if labels[new_idx] not in map_d:
                map_d[labels[new_idx]] = 0.
                count_d[labels[new_idx]] = 0.
            map_d[labels[new_idx]] += map_score
            count_d[labels[new_idx]] += 1.0
            avg_map += map_score
            print("map score: ", map_score)

            continue

            # pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            #
            #
            # pred_all.append(pred)
            # real_all.append(real)
            # denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            # denoised_adj = nx.to_numpy_matrix(G)
            # graphs.append(G)
            # feats.append(denoised_feat)
            # adjs.append(denoised_adj)
            # io_utils.log_graph(
            #     self.writer,
            #     G,
            #     "graph/{}_{}_{}".format(self.args.dataset, model, i),
            #     identify_self=True,
            #     args=self.args
            # )
        avg_map_score = avg_map / len(node_indices)
        print("\n\navg map score: ", avg_map_score, "\n\n")
        for k in map_d.keys():
            print("label: ", k, "  map: ", map_d[k] / count_d[k], "  count: ", count_d[k])
        if args.fname != "":
            full_path = "./tuning/explain/" + self.args.fname
            file1 = open(full_path, "a")  # write mode
            file1.write(str(args.size_c) + " " + str(args.lap_c) + " " + str(args.ent_c) + "\n")
            file1.write(str(avg_map_score) + "\n")
            for k in map_d.keys():
                file1.write(str(k) + " " + str(map_d[k] / count_d[k]) + " " + str(count_d[k]) + " ")
            file1.write("\n\n")
            file1.close()

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
    def explain_graphs(self, args, graph_indices, test_graph_indices=None):


        global global_noise_count
        global noise_diff_count
        global avg_add_edges
        global avg_removed_edges
        """
        Explain graphs.
        """
        masked_adjs = []
        logging_graphs = False
        rule_acc_count = 0.
        noise_count = 0.
        mAP = 0.
        mAP_n = 0
        AUC_n = accuracy_utils.AUC()
        noise_AUC = accuracy_utils.AUC()

        # noise stats
        noise_iters = 1
        noise_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        if args.adversarial_path is not None:
            with open(args.adversarial_path, mode='rb') as infile:
                adversarial_file = pickle.load(infile, encoding='bytes')
            noise_handlers = [noise_utils.AdversarialNoiseHandler("GNNExplainer", self.model, self, noise_percent=0, adversarial_file=adversarial_file)]
        else:
            raise NotImplementedError


        if self.args.bmname == 'synthetic':
            graph_idx = 0
            h_nodes = sub_label_nodes.reshape(8000, 8).tolist()
            h_edges = [accuracy_utils.gethedges(nodes) for nodes in h_nodes]

            stats = accuracy_utils.Stats("GNNExplainer", self, gt_nodes=h_nodes, gt_edges=h_edges)
        elif self.args.bmname == 'Mutagenicity':
            #h_edges = accuracy_utils.gethedgesmutag()

            stats = accuracy_utils.Stats("GNNExplainer", self)
        else:
            stats = accuracy_utils.Stats("GNNExplainer", self)

        times = []

        idx = 0

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

            start = time.time()

            masked_adj = self.explain_input(sub_feat, sub_adj, sub_nodes, sub_label, graph_idx=graph_idx, graph_mode=True)
            end = time.time()
            times.append(end - start)
            print(np.mean(times), np.std(times))
            imp_nodes = explain.getImportantNodes(masked_adj, 8)
            stats.update(masked_adj, imp_nodes, graph_idx)

            if self.args.noise:
                print("SUMMARY")
                for nh in noise_handlers:
                    print(nh.summary())

            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)

            if pred_label == args.multigraph_class:
                if self.args.noise:
                    
                    masked_adj_gt = masked_adj

                    for n_iter in range(noise_iters):
                        for nh in noise_handlers:
                            
                            noise_feat, noise_adj, noise_label = nh.sample(sub_feat, sub_adj, sub_nodes, graph_idx=idx)
                            
                            # masked_adj_n = self.explain_input(sub_feat, sub_adj, sub_nodes, sub_label, graph_idx=graph_idx, graph_mode=True)
                            masked_adj_n = self.explain_input(noise_feat, noise_adj, sub_nodes, sub_label, graph_idx=graph_idx, graph_mode=True)
                            
                            masked_adj_n = masked_adj_n * noise_adj.cpu().detach().numpy()
                            nh.update(masked_adj_gt, masked_adj_n, sub_adj, noise_adj.cpu().detach().numpy(), None, graph_idx, noise_label)
                    idx += 1
            if self.args.bmname == 'synthetic' or self.args.bmname == 'Mutagenicity':
                print("flips: ", flips)
                thresh_nodes = 15

                
                if self.args.bmname == 'synthetic':

                    imp_nodes = explain.getImportantNodes(masked_adj, 8)
                    h_nodes = []
                    h_nodes.extend(sub_label_nodes[graph_idx,0,0,:].tolist())
                    h_nodes.extend(sub_label_nodes[graph_idx,1,0,:].tolist())
                    mAP_s = accuracy_utils.getmAP(masked_adj, h_nodes)
                    mAP += mAP_s
                    rule_acc_count += 1
                    h_edges = accuracy_utils.gethedges(h_nodes)


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

                masked_adjs.append(masked_adj)
                if logging_graphs:
                    G_orig = io_utils.denoise_graph(
                        self.adj[graph_idx].cpu().numpy(),
                        0,
                        feat=self.feat[graph_idx],
                        threshold=None,
                        max_component=False,
                    )

                    io_utils.log_graph(
                        self.writer,
                        G_orig,
                        "graph/graphidx_{}".format(graph_idx),
                        identify_self=False,
                        nodecolor="feat",
                        args=self.args
                    )

            else:
                masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)

                thresh_nodes = 15
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

                masked_adjs.append(masked_adj)
                if logging_graphs:
                    G_orig = io_utils.denoise_graph(
                        self.adj[graph_idx].cpu().numpy(),
                        0,
                        feat=self.feat[graph_idx],
                        threshold=None,
                        max_component=False,
                    )


                    io_utils.log_graph(
                        self.writer,
                        G_orig,
                        "graph/graphidx_{}".format(graph_idx),
                        identify_self=False,
                        nodecolor="feat",
                        args=self.args
                    )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")
        # if self.args.bmname == "Mutagenicity":
        #     print("ROC AUC mask-noise: {}".format(AUC_n.getAUC()))
        #     print("mAP mask-noise: {}".format(mAP_n / global_noise_count))
        #     # print(
        #     #     "ROC AUC score: {}".format(AUC.getAUC())
        #     # )
        #     # print(
        #     #     "noisy ROC AUC score: {}".format(noise_AUC.getAUC())
        #     # )
        #     print(
        #     "noise percent: {}".format(args.noise_percent)
        #     )
        #     print(
        #         "avg removed edges: {}".format(avg_removed_edges / global_noise_count)
        #     )
        #     print(
        #         "avg added edges: {}".format(avg_add_edges / global_noise_count)
        #     )
        #     print(
        #         "avg adj diff: {}".format(avg_adj_diff / global_noise_count)
        #     )
        #     # print(
        #     #     "avg noise diff: {}".format(avg_noise_diff / noise_diff_count)
        #     # )

        # print("Noise mAP: {}".format(mAP_n / noise_count))
        # print("Noise AUC: {}".format(AUC_n.getAUC()))
        if self.args.bmname == "synthetic":
            print("mAP: {}".format(mAP / rule_acc_count))

        # if self.args.bmname == 'Mutagenicity':
        #     pickle.dump(masked_adjs, open("../data/Mutagenicity/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        # else:
        #     pickle.dump(masked_adjs, open("../data/synthetic_data_3label_3sublabel/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        print("Flips: ", flips)
        print("Incorrect preds: ", incorrect_preds)

        print(stats)
        if self.args.noise:
            for nh in noise_handlers:
                print(nh)
            print("SUMMARY")
            for nh in noise_handlers:
                print(nh.summary())
        print("FIDELITY SUMMARY")
        print(stats.summary())
        return masked_adjs


    
class ExplainModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        sub_num_nodes = None
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.sub_num_nodes = sub_num_nodes
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.001,#syn1, 0.009
            # "size": 0.012,#mutag
            # "size": 0.009,#synthetic graph
            #"size": 0.007, #0.04 for synthetic,#0.007=>40%,#0.06,#0.005,
            "feat_size": 0.0,
            "ent": 0.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self, node_idx, unconstrained=False, mask_features=True, marginalize=False):
        x = self.x.cuda() if self.args.gpu else self.x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask #making it symmetric,
                # only true for undirected graphs?
            )
        else: #applicable for graph explanation
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize: # add noise or mask features
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x
                    # x = x * feat_mask

        if self.masked_adj.shape[1] == 5129:
            ypred = self.model.predict(x, probs=False, adj=self.masked_adj.squeeze(0))
            adj_att = None
        else:

            if self.sub_num_nodes is not None:
                sub_num_nodes_l = [self.sub_num_nodes.cpu().numpy()]
            else:
                sub_num_nodes_l = None
            ypred, adj_att = self.model(x, self.masked_adj, batch_num_nodes=sub_num_nodes_l)


        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        if adj.shape[1] == 5129:
            if len(adj.shape) == 3:
                adj_sq = adj.squeeze(0)
            ypred = self.model.predict(x, probs=False, adj=adj_sq)
        else:
            ypred, _ = self.model(x, adj)

        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[gt_label_node]
            # logit = pred[pred_label_node]

            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        if self.args.size_c > -0.001:
            size_loss = self.args.size_c * torch.sum(mask)
        else:
            size_loss = self.coeffs["size"] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy for making it discrete
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        if self.args.ent_c > -0.001:
            mask_ent_loss = self.args.ent_c * torch.mean(mask_ent)
        else:
            mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            if self.args.lap_c > -0.001:
                lap_loss = (self.args.lap_c
                            * (pred_label_t @ L @ pred_label_t)
                            / self.adj.numel()
                            )
            else:
                lap_loss = (self.coeffs["lap"]
                    * (pred_label_t @ L @ pred_label_t)
                    / self.adj.numel()
                )


        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss


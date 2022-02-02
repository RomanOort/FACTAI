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
import utils.noise_utils as noise_utils

import dnn_invariant.extract_rules as extract
import utils.graph_utils as graph_utils
import utils.accuracy_utils as accuracy_utils
import utils.neighbor_utils as neighbor_utils
import explainer.explain as explain
from explainer.explain import AverageMeter

from scipy.sparse import coo_matrix

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor



nbr_data = None
rule_dict_node = None

use_comb_mask = True
avg_add_edges = 0.
avg_removed_edges = 0.
global_noise_count = 0.
global_mask_dense_count = 0.
global_mask_density = 0.

ent_cf = -1.0
size_cf = -1.0
lap_cf = -1.0

sub_label_nodes = None
sub_label_array = None

# sub_label_nodes = pickle.load(open(
#     "/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p",
#     "rb"))['sub_label_nodes']
# sub_label_nodes = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20.p"))['sub_label_nodes']


def load_sublabel_nodes(args):
    global sub_label_nodes
    global sub_label_array

    if args.bmname == 'synthetic':
        synthetic_data = pickle.load(open(
            "/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p",
            "rb"))
    elif args.bmname == 'old_synthetic':
        synthetic_data = pickle.load(open(
            "/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_nofake.p", "rb"))
    sub_label_nodes = synthetic_data['sub_label_nodes']
    sub_label_array = synthetic_data['sub_label']



def get_boundary(node_idx, dataset="syn1"):

    if dataset == "syn1":
        offset_idx = node_idx
    elif dataset == "syn2":
        offset_idx = node_idx
        # if node_idx < 700:
        #     offset_idx = node_idx - 400
        # else:
        #     offset_idx = node_idx - 800
    elif dataset == "syn3":
        offset_idx = node_idx
    elif dataset == "syn4":
        offset_idx = node_idx
    elif dataset == "syn8":
        offset_idx = node_idx



    boundary_list = []


    # rule_ix = rule_dict_node['idx2rule'][offset_idx]
    if isinstance(rule_dict_node['idx2rule'][offset_idx], list):
        rule_list = rule_dict_node['idx2rule'][offset_idx]
    else:
        rule_list = [rule_dict_node['idx2rule'][offset_idx]]
    for rule_ix in rule_list:
        rule = rule_dict_node['rules'][rule_ix]
        rule_label = rule['label']
        rule_imp_nodes = []
        rule_h_nodes = []
        h_nodes = []
        for b_num in range(len(rule['boundary'])):
            boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
            boundary = boundary.cuda()
            boundary_label = rule['boundary'][b_num]['label']
            boundary_list.append((boundary, boundary_label))
    return boundary_list


class ExplainerTwoRCExplainer(explain.Explainer):
    def __init__(
        self,
        model,
        adj,
        feat,
        label,
        pred,
        emb,
        train_idx,
        args,
        writer=None,
        print_training=False,
        graph_mode=False,
        graph_idx=False,
        num_nodes = None,
        device='cpu'
    ):
        super().__init__(model, adj, feat, label, pred, train_idx, args, writer, print_training, graph_mode, graph_idx, num_nodes,device)

        self.coeffs = {
            "t0": 0.5,
            "t1": 4.99,
        }

        self.emb = emb

        # self.model = model
        self.model.eval()
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

    def get_nbr_data_emb(self, args, node_indices, graph_idx=0):
        torch_data = super().get_nbr_data(args, node_indices, graph_idx)


        emb_l = []
        for node_idx in node_indices:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_emb(node_idx, graph_idx)# get this

            emb_l.append(sub_emb)
            # TODO: port over rest of preprocessing, concat to torch_data, then use! 
            # Need embeddings for 
        torch_data['emb'] = emb_l

        self.nbr_data = torch_data

        return torch_data
    
    def extract_neighborhood_from_saved_data_emb(self, node_idx, dataset):
        new_idx, adj, feat, label, nbrs = super().extract_neighborhood_from_saved_data(node_idx, dataset)
        offset_idx = node_idx
        #print(self.nbr_data.keys())
        emb = self.nbr_data['emb'][offset_idx]
        return new_idx, adj, feat, label, nbrs, emb

    def get_nbr_data(self, args, node_indices, graph_idx=0):
        torch_data = super().get_nbr_data(args, node_indices, graph_idx)


        for node_idx in node_indices:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors= self.extract_neighborhood(node_idx, graph_idx)# get this

            # TODO: port over rest of preprocessing, concat to torch_data, then use! 
            # Need embeddings for 
        return torch_data


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
        # print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs
    def eval_nodes(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):


        if self.args.exp_path == "":
            print("no explainer file to load")
            exit()
        else:
            print("loading initial explainer ckpt from : ", self.args.exp_path)

        if args.draw_graphs:
            random.shuffle(node_indices)

            node_indices = node_indices[:5]

        nbr_data = self.get_nbr_data_emb(args, graph_node_indices, graph_idx)

        node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
            node_indices[0], self.args.bmname
        )

        explainer = ExplainModule(
            model=self.model,
            num_nodes=self.adj.shape[1],
            emb_dims=self.model.embedding_dim * self.model.num_layers * 3,
            device=self.device,
            args=self.args
        )

        try:
            state_dict = torch.load(self.args.exp_path)
        except:
            state_dict = torch.load(self.args.exp_path + 'rcexplainer.pth.tar')
        exp_state_dict = explainer.state_dict()
        for name, param in state_dict.items():
            if name in exp_state_dict and not ("model" in name):
                exp_state_dict[name].copy_(param)
        explainer.load_state_dict(exp_state_dict)

        self.model.eval()
        explainer.eval()

        num_classes = self.pred[0][0].shape[0]
        flips = np.zeros((num_classes))
        inv_flips = np.zeros((num_classes))
        topk_inv_flips = np.zeros((num_classes))
        pos_diff = np.zeros((num_classes))
        inv_diff = np.zeros((num_classes))
        topk_inv_diff = np.zeros((num_classes))
        total = np.zeros((num_classes))

        AUC = accuracy_utils.AUC()
        avg_mask_density = 0.
        avg_map = 0.0
        avg_acc = 0.0
        avg_node = 0.0
        stats = accuracy_utils.Stats("RCExplainer", self, graph_mode=False)

        loss_ep = 0.
        mask_density = 0.
        ep_variance = 0.
        incorrect_preds = 0.
        for i, node_idx in enumerate(node_indices):
            with torch.no_grad():

                # new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
                # new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood(idx)


                node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
                    node_idx, self.args.bmname
                )


                sub_label = np.expand_dims(sub_label, axis=0)
                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)


                adj = torch.tensor(sub_adj, dtype=torch.float)
                x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)

                if self.emb is not None:
                    sub_emb = np.expand_dims(sub_emb, axis=0)
                    emb   = torch.tensor(sub_emb, dtype=torch.float)
                else: 
                    emb = model.getEmbeddings(x, adj)
                    emb = emb.clone().detach()


                pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)

                t0 = self.coeffs['t0']
                t1 = self.coeffs['t1']
                tmp = float(t0 * np.power(t1 / t0, 1.0))
                gt_pred, gt_embedding = self.model(x.cuda(), adj.cuda(), batch_num_nodes=None,
                                                   new_node_idx=[node_idx_new])
                gt_pred = torch.nn.functional.softmax(gt_pred[0][node_idx_new, :], dim=0)

                pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer(
                    (x[0], emb[0], adj[0], tmp, label, None), node_idx=node_idx_new, training=False)

                # print("prefix: ", self.args.prefix)

                if graph_embedding is not None:
                    graph_embedding = graph_embedding.squeeze(0)

                if inv_embedding is not None:
                    inv_embedding = inv_embedding.squeeze(0)


                new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood_from_saved_data(node_idx, self.args.bmname)
                labels = self.label[graph_idx][nbrs]

                masked_adj = masked_adj.cpu().clone().detach().numpy()
                stats.update(masked_adj, None, node_idx=node_idx)

                topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=self.args.topk)

                # topk_adj, topk_x = noise_utils.filterGT(masked_adj, sub_adj[0], x, h_edges[graph_idx])

                # print("Adj: ", np.sum(topk_adj), np.sum(sub_adj[0]))

                topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0), batch_num_nodes=None,
                                          new_node_idx=[node_idx_new])
                pred_topk = torch.nn.functional.softmax(pred_topk[0][node_idx_new, :], dim=0)
                pred_label = pred_label[node_idx_new]
                topk_inv_diff[pred_label] += (gt_pred[pred_label] - pred_topk[pred_label]).item()
                # print("topk debug: ", pred_label, gt_pred, pred_topk)
                if torch.argmax(pred_topk) == pred_label:
                    topk_inv_flips[pred_label] += 1.0

                if pred_label != label[0, node_idx_new].item():
                    incorrect_preds += 1
                if pred is not None:
                    # print("debug pred: ", self.pred[0][graph_idx], pred, gt_pred)
                    if torch.argmax(pred[0]) != pred_label:
                        flips[pred_label] += 1.0
                        pos_diff[pred_label] += (gt_pred[pred_label] - pred[pred_label]).item()

                if inv_pred is not None:
                    # print("inv pred: ", inv_pred)
                    if torch.argmax(inv_pred[0]) == pred_label:
                        inv_flips[pred_label] += 1.0
                        inv_diff[pred_label] += (gt_pred[pred_label] - inv_pred[pred_label]).item()

                total[pred_label] += 1.0

                # G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
                # print(masked_adjs[i].shape, i, idx, new_idx)
                if self.args.bmname == 'syn3':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn3(masked_adj, n_adj, labels, nbrs, new_idx,
                                                                   self.adj[0])

                elif self.args.bmname == 'syn4':
                    # map_score = getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn4(masked_adj, n_adj, labels, nbrs, new_idx,
                                                                   self.adj[0])
                elif self.args.bmname == 'syn1':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn1(masked_adj, n_adj, labels, nbrs, new_idx,
                                                                   self.adj[0])
                elif self.args.bmname == 'syn2':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn2(masked_adj, n_adj, labels, nbrs, new_idx,
                                                                   self.adj[0])
                else:
                    map_score = accuracy_utils.getmAPNodes(masked_adj, n_adj, labels, nbrs, new_idx)

                AUC.addEdgesFromDict(masked_adj, h_edges)
                mask_density = np.sum(masked_adj) / np.sum(n_adj)
                assert (np.abs(np.sum(n_adj) - np.sum(sub_adj)) < 0.1)
                avg_mask_density += mask_density

                variance = np.sum(np.abs(masked_adj - 0.5) * sub_adj[0]) / np.sum(sub_adj[0])
                ep_variance += variance

                avg_map += map_score
                avg_acc += acc
                avg_node += acc_node



                if args.draw_graphs:

                    for n in range(len(neighbors)):
                        sub_feat[0,n] = 0.
                        sub_feat[0,n,labels[n]] = 1.0

                    gt_mask = sub_adj[0] - 0.9
                    for e in h_edges.keys():
                        gt_mask[e[0], e[1]] = 1.0
                        gt_mask[e[1], e[0]] = 1.0

                    # gt_mask
                    accuracy_utils.saveAndDrawGraph(gt_mask, sub_adj[0], sub_feat[0],
                                                    len(neighbors),
                                                    self.args,
                                                    label[0, node_idx_new].item(), pred_label, node_idx,
                                                    prob=pred[pred_label].item(), node_idx = node_idx_new,
                                                    plt_path=None, adj_mask_bool=True, prefix="gt_")

                    accuracy_utils.saveAndDrawGraph(masked_adj, sub_adj[0], sub_feat[0],
                                                    len(neighbors),
                                                    self.args,
                                                    label[0, node_idx_new].item(), pred_label, node_idx,
                                                    prob=pred[pred_label].item(), node_idx = node_idx_new,
                                                    plt_path=None, adj_mask_bool=True)
                    # topk_adj
                    accuracy_utils.saveAndDrawGraph(None, topk_adj, sub_feat[0], len(neighbors),
                                                    self.args,
                                                    label[0, node_idx_new].item(), pred_label, node_idx, prob=pred_topk[pred_label].item(),
                                                    node_idx=node_idx_new,
                                                    plt_path=None, adj_mask_bool=False)




        avg_map_score = avg_map / len(node_indices)
        avg_acc_score = avg_acc / len(node_indices)
        avg_acc_node = avg_node / len(node_indices)


        avg_mask_density = avg_mask_density / len(node_indices)
        total[total < 0.5] = 1.0

        print("\n\navg map score: ", avg_map_score, "\n\n")
        eval_dir = os.path.dirname(self.args.exp_path)
        eval_file = "eval_" + self.args.bmname + "_" + self.args.explainer_method + ".txt"

        eval_file = os.path.join(eval_dir, eval_file)

        myfile = open(eval_file, "a")

        auc_res = AUC.getAUC()
        print(
            "ROC AUC score: {}".format(auc_res)
        )

        myfile.write("\n ROC AUC score: {}".format(auc_res))

        print(
            "acc score: {}".format(avg_acc_score)
        )

        myfile.write("\n avg_acc_score: {}".format(avg_acc_score))

        print(
            "acc node score: {}".format(avg_acc_node)
        )

        myfile.write("\n avg_acc_node: {}".format(avg_acc_node))

        avg_map_score = avg_map / len(node_indices)
        myfile.write("\n mAP score: {}".format(avg_map_score))

        print(
            "Average mask density: {}".format(avg_mask_density)
        )
        myfile.write("\n Average mask density: {}".format(avg_mask_density))

        print(
            "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total,
                                                                          self.args.topk,
                                                                          topk_inv_diff / total)

        )

        myfile.write("\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total,
                                                                               topk_inv_diff / total))

        print("Variance: ", ep_variance / len(node_indices))
        myfile.write("\n Variance: {}".format(ep_variance / len(node_indices)))

        print("Flips: ", flips)
        print("inv Flips: ", inv_flips)
        print("topk inv Flips: ", topk_inv_flips)
        print("Incorrect preds: ", incorrect_preds)
        print("Total: ", total)

        print(stats.summary())


        myfile.write(
            "\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}, Total: {}".format(
                flips, inv_flips, self.args.topk, topk_inv_flips, incorrect_preds, total))

        myfile.close()


    def explain_nodes_gnn_stats(self, node_indices, graph_node_indices, args, graph_idx=0, model="exp"):
        if self.args.eval:
            self.eval_nodes(node_indices, graph_node_indices, args, graph_idx=graph_idx, model=model)
            exit()
        global nbr_data
        global rule_dict_node

        # if self.args.bmname == "syn1":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn1/torch_data.pth")
        # elif self.args.bmname == "syn2":
        #     # nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn2/torch_data.pth")
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn2/torch_data_all_binary.pth")

        # elif self.args.bmname == "syn3":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn3/torch_data_all.pth")
        # elif self.args.bmname == "syn4":
        #     nbr_data = torch.load("/home/mohit/Mohit/gcn_interpretation/data/syn4/torch_data_all.pth")


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
        log_name = self.args.prefix + "_logdir"
        log_path = os.path.join(self.args.ckptdir, log_name)
        if os.path.isdir(log_path):
            print("log dir already exists and will be overwritten")
            time.sleep(5)
        else:
            os.mkdir(log_path)

        log_file = self.args.prefix + "log_rcexplainer_" + self.args.bmname + ".txt"
        log_file_path = os.path.join(log_path, log_file)
        myfile = open(log_file_path, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}".format(self.args.bloss_version))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))

        myfile.write(
            "\n lr: {}, bound cf: {}, size cf: {}, ent cf {}, lap cf {}, inv cf: {}".format(self.args.lr, self.args.boundary_c,
                                                                                self.args.size_c, self.args.ent_c,
                                                                                self.args.lap_c, self.args.inverse_boundary_c))
        myfile.close()

        if self.args.bmname == "syn1":
            size = 700
            width = 4

        elif self.args.bmname == "syn2":
            size = 1400
            width = 10

        elif self.args.bmname == "syn3":
            size = 1020
            width = 10


        elif self.args.bmname == "syn4":
            size = 871
            width = 10

        elif self.args.bmname == "syn8":
            size = 660
            width = 5
        nbr_data = self.get_nbr_data_emb(args, graph_node_indices, graph_idx)


        train_data, val_data = neighbor_utils.process_for_boundary(nbr_data, size, size, size, 100, width)

        rule_dict_node = extract.extract_rules(self.args.bmname, train_data, val_data, self.args,
                                               self.model.state_dict())


        node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
            node_indices[0], self.args.bmname
        ) 

        explainer = ExplainModule(
            model = self.model, 
            num_nodes = self.adj.shape[1],
            emb_dims = self.model.embedding_dim * self.model.num_layers * 3,
            device = self.device,
            args = self.args
        )

        params_optim = []
        for name, param in explainer.named_parameters():
            if "model" in name:
                continue
            params_optim.append(param)

        scheduler, optimizer = train_utils.build_optimizer(self.args, params_optim)

        for epoch in range(self.args.num_epochs):
            num_classes = self.pred[0][0].shape[0]
            flips = np.zeros((num_classes))
            inv_flips = np.zeros((num_classes))
            topk_inv_flips = np.zeros((num_classes))
            pos_diff = np.zeros((num_classes))
            inv_diff = np.zeros((num_classes))
            topk_inv_diff = np.zeros((num_classes))
            total = np.zeros((num_classes))

            AUC = accuracy_utils.AUC()
            stats = accuracy_utils.Stats("RCExplainer", self, graph_mode=False)
            avg_mask_density = 0.
            avg_map = 0.0
            avg_acc = 0.0
            avg_node = 0.0
            loss_ep = 0.
            mask_density = 0.
            ep_variance = 0.
            incorrect_preds = 0.
            for i, node_idx in enumerate(node_indices):

                # new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
                # new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood(idx)
                boundary_l = get_boundary(node_idx, self.args.bmname)
                boundary_list = [bound[0] for bound in boundary_l]

                node_idx_new, sub_adj, sub_feat, sub_label, neighbors, sub_emb = self.extract_neighborhood_from_saved_data_emb(
                    node_idx, self.args.bmname
                ) 
                sub_label = np.expand_dims(sub_label, axis=0)
                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)

                adj   = torch.tensor(sub_adj, dtype=torch.float)
                x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)

                if self.emb is not None:
                    sub_emb = np.expand_dims(sub_emb, axis=0)
                    emb   = torch.tensor(sub_emb, dtype=torch.float)
                else: 
                    emb = model.getEmbeddings(x, adj)
                    emb = emb.clone().detach()

                pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)

                t0 = self.coeffs['t0']
                t1 = self.coeffs['t1']
                tmp = float(t0 * np.power(t1 / t0, epoch /self.args.num_epochs))
                gt_pred, gt_embedding = self.model(x.cuda(), adj.cuda(), batch_num_nodes=None, new_node_idx=[node_idx_new])
                gt_pred = torch.nn.functional.softmax(gt_pred[0][node_idx_new, :], dim=0)

                pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer(
                    (x[0], emb[0], adj[0], tmp, label, None), node_idx=node_idx_new, training=False)



                print("prefix: ", self.args.prefix)

                if graph_embedding is not None:
                    graph_embedding = graph_embedding.squeeze(0)

                if inv_embedding is not None:
                    inv_embedding = inv_embedding.squeeze(0)


                loss, bloss_s = explainer.loss(pred, pred_label, node_idx=node_idx_new, graph_embedding=graph_embedding,
                                               boundary_list=boundary_list, gt_embedding=gt_embedding.squeeze(0),
                                               inv_embedding=inv_embedding)




                loss_ep = loss_ep + loss

                new_idx, n_adj, feat, _, nbrs = self.extract_neighborhood_from_saved_data(node_idx, self.args.bmname)
                labels = self.label[graph_idx][nbrs]

                masked_adj = masked_adj.cpu().clone().detach().numpy()

                stats.update(masked_adj, None, node_idx=node_idx)

                topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=self.args.topk)

                # topk_adj, topk_x = noise_utils.filterGT(masked_adj, sub_adj[0], x, h_edges[graph_idx])

                print("Adj: ", np.sum(topk_adj), np.sum(sub_adj[0]))

                topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0), batch_num_nodes=None, new_node_idx=[node_idx_new])
                pred_topk = torch.nn.functional.softmax(pred_topk[0][node_idx_new, :], dim=0)
                pred_label = pred_label[node_idx_new]
                topk_inv_diff[pred_label] += (gt_pred[pred_label] - pred_topk[pred_label]).item()
                print("topk debug: ", pred_label, gt_pred, pred_topk)
                if torch.argmax(pred_topk) == pred_label:
                    topk_inv_flips[pred_label] += 1.0

                if pred_label != label[0, node_idx_new].item():
                    incorrect_preds += 1
                if pred is not None:
                    print("debug pred: ", self.pred[0][graph_idx], pred, gt_pred)
                    if torch.argmax(pred[0]) != pred_label:
                        flips[pred_label] += 1.0
                        pos_diff[pred_label] += (gt_pred[pred_label] - pred[pred_label]).item()

                if inv_pred is not None:
                    print("inv pred: ", inv_pred)
                    if torch.argmax(inv_pred[0]) == pred_label:
                        inv_flips[pred_label] += 1.0
                        inv_diff[pred_label] += (gt_pred[pred_label] - inv_pred[pred_label]).item()

                total[pred_label] += 1.0



                # G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
                # print(masked_adjs[i].shape, i, idx, new_idx)
                if self.args.bmname == 'syn3':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn3(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])

                elif self.args.bmname == 'syn4':
                    # map_score = getmAPsyn3(masked_adjs[i], n_adj, labels, nbrs, new_idx, self.adj[0])
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn4(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                elif self.args.bmname == 'syn1':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn1(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                elif self.args.bmname == 'syn2':
                    (map_score, acc, acc_node), h_edges = accuracy_utils.getmAPsyn2(masked_adj, n_adj, labels, nbrs, new_idx, self.adj[0])
                else:
                    (map_score, acc, acc_node) = accuracy_utils.getmAPNodes(masked_adj, n_adj, labels, nbrs, new_idx)
                AUC.addEdgesFromDict(masked_adj, h_edges)
                mask_density = np.sum(masked_adj) / np.sum(n_adj)
                assert(np.abs(np.sum(n_adj) - np.sum(sub_adj)) < 0.1)
                avg_mask_density += mask_density

                variance = np.sum(np.abs(masked_adj - 0.5) * sub_adj[0]) / np.sum(sub_adj[0])
                ep_variance += variance


                avg_map += map_score
                avg_acc += acc
                avg_node += acc_node
            optimizer.zero_grad()
            loss_ep.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            avg_map_score = avg_map / len(node_indices)
            avg_acc_score = avg_acc / len(node_indices)
            avg_acc_node = avg_node / len(node_indices)
            avg_mask_density = avg_mask_density / len(node_indices)
            total[total < 0.5] = 1.0

            print("\n\navg map score: ", avg_map_score, "\n\n")

            myfile = open(log_file_path, "a")

            auc_res = AUC.getAUC()
            print(
                "ROC AUC score: {}".format(auc_res)
            )
            myfile.write("\n\n epoch: {}".format(epoch))

            myfile.write("\n ROC AUC score: {}".format(auc_res))

            print(
                "acc score: {}".format(avg_acc_score)
            )

            myfile.write("\n avg_acc_score: {}".format(avg_acc_score))

            print(
                "acc node: {}".format(avg_acc_node)
            )

            myfile.write("\n avg_acc_node: {}".format(avg_acc_node))


            avg_map_score = avg_map / len(node_indices)
            myfile.write("\n mAP score: {}".format(avg_map_score))

            print(
                "Average mask density: {}".format(avg_mask_density)
            )
            myfile.write("\n Average mask density: {}".format(avg_mask_density))

            print(
                "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total, self.args.topk,
                                                                              topk_inv_diff / total)

            )

            myfile.write("\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total,
                                                                                   topk_inv_diff / total))

            print("Variance: ", ep_variance / len(node_indices))
            myfile.write("\n Variance: {}".format(ep_variance / len(node_indices)))

            print("Flips: ", flips)
            print("inv Flips: ", inv_flips)
            print("topk inv Flips: ", topk_inv_flips)
            print("Incorrect preds: ", incorrect_preds)
            print("Total: ", total)

            print(stats.summary())

            myfile.write(
                "\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}, Total: {}".format(
                    flips, inv_flips, self.args.topk, topk_inv_flips, incorrect_preds, total))

            if epoch % 10 == 0:
                explainer_sum = 0.0
                model_sum = 0.0
                for p in explainer.parameters():
                    explainer_sum += torch.sum(p).item()
                for p in self.model.parameters():
                    model_sum += torch.sum(p).item()

                # f_path = './ckpt/explainer3_synthetic_data_3label_3sublabel_pgeboundary' + '.pth.tar'
                myfile.write("\n explainer params sum: {}, model params sum: {}".format(explainer_sum, model_sum))

                f_path = self.args.prefix + "explainer_" + self.args.bmname + "_pgenoboundary.pth.tar"
                save_path = os.path.join(log_path, f_path)
                torch.save(explainer.state_dict(), save_path)
                myfile.write("\n ckpt saved at {}".format(save_path))
            if epoch % 100 == 0:
                # f_path = './ckpt/explainer3_synthetic_data_3label_3sublabel_pgeboundary' + '.pth.tar'
                f_path = self.args.prefix + "explainer_" + self.args.bmname + "_pgenoboundary_ep_" + str(
                    epoch) + ".pth.tar"
                save_path = os.path.join(log_path, f_path)
                torch.save(explainer.state_dict(), save_path)
                myfile.write("\n ckpt saved at {}".format(save_path))
            myfile.close()



        return None

    def eval_graphs_2(self, args, graph_indices, explainer):
        if self.args.apply_filter and self.args.bmname == 'Mutagenicity':
            # h_edges = accuracy_utils.gethedgesmutag()
            graph_indices, h_edges = accuracy_utils.filterMutag2(graph_indices, self.label, self.feat, self.adj,
                                                                 self.num_nodes)
            # random.shuffle(graph_indices)

        if args.draw_graphs:
            random.shuffle(graph_indices)

            graph_indices = graph_indices[:5]
            # graph_indices = [600, 1100, 1200, 1500, 2500, 2858, 2201, 2777, 228, 231]

        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        global global_noise_count
        global noise_diff_count
        global avg_add_edges
        global avg_removed_edges
        ep_variance = 0.

        incorrect_preds = 0.

        self.model.eval()
        num_classes = self.pred[0][0].shape[0]
        flips = np.zeros((num_classes))

        inv_flips = np.zeros((num_classes))
        topk_inv_flips = np.zeros((num_classes))
        pos_diff = np.zeros((num_classes))
        inv_diff = np.zeros((num_classes))
        topk_inv_diff = np.zeros((num_classes))
        total = np.zeros((num_classes))

        masked_adjs = []
        skipped_iters = 0.
        logging_graphs = False
        avg_top4_acc = 0.
        avg_top6_acc = 0.
        avg_top8_acc = 0.
        avg_noise_diff = 0.
        noise_diff_count = 0.
        avg_adj_diff = 0.
        acc_count = 0.

        rule_top8_acc = 0.

        avg_mask_density = 0.
        mAP = 0.
        noise_mAP = 0.
        AUC = accuracy_utils.AUC()
        noise_AUC = accuracy_utils.AUC()

        avg_pred_diff = 0.
        pred_removed_edges = 0.
        topk = self.args.topk

        noise_iters = 1
        noise_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        noise_handlers = [noise_utils.NoiseHandler("RCExplainer", self.model, self, noise_percent=x) for x in noise_range]


        graph_indices = list(graph_indices)
        np.random.shuffle(graph_indices)
        explainer.eval()
        explainer_sum = 0.0
        model_sum = 0.0
        mean_auc = 0.0

        for p in explainer.parameters():
            explainer_sum += torch.sum(p).item()
        for p in self.model.parameters():
            model_sum += torch.sum(p).item()
        print("sum of params of loaded explainer: {}".format(explainer_sum))
        print("sum of params of loaded model: {}".format(model_sum))

        # for graph_idx in graph_indices:
        # order = list(range(self.num_nodes[graph_idx].item()))
        # order = list(range(self.adj.shape[1]))
        # order = orders[graph_idx]
        # rand_order = shuffle_forward(order)
        # rand_order = rand_orders[graph_idx]
        # self.feat[graph_idx, rand_order, :] = self.feat[graph_idx, order, :]
        # self.adj[graph_idx, rand_order, :] = self.adj[graph_idx, order, :]
        # self.adj[graph_idx, :, rand_order] = self.adj[graph_idx, :, order]
        stats = accuracy_utils.Stats("PGExplainer_Boundary", self)

        for graph_idx in graph_indices:
            with torch.no_grad():
                print("doing for graph index: ", graph_idx)

                # extract features and

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
                neighbors = np.asarray(range(self.adj.shape[0]))  # 1,2,3....num_nodes

                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)

                adj = torch.tensor(sub_adj, dtype=torch.float)
                x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)

                # if self.emb is not None:
                #     sub_emb = self.emb[graph_idx, :]
                #     sub_emb = np.expand_dims(sub_emb, axis=0)
                #     emb = torch.tensor(sub_emb, dtype=torch.float)

                # else:
                emb = self.model.getEmbeddings(x, adj, batch_num_nodes=[sub_nodes.cpu().numpy()])
                emb = emb.clone().detach()


                pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
                if pred_label != label.item():
                    incorrect_preds += 1
                # print("Graph predicted label: ", pred_label)

                if self.args.bmname == 'Mutagenicity':
                    t0 = 5.0
                    t1 = 5.0
                else:
                    t0 = 0.5
                    t1 = 4.99

                tmp = float(t0 * np.power(t1 / t0, 1.0))
                # tmp = float(t0 * np.power(t1 / t0, 0.9))

                # tmp = float(t0 * np.power(t1 / t0, 0.0001))

                emb_noise2 = self.model.getEmbeddings(x.cuda(), adj.cuda(), [sub_nodes.cpu().numpy()])

                pred, masked_adj, _, _, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes),
                                                             training=False)
                # explainer.loss(pred, pred_label)


                pred_try, _ = self.model(x.cuda(), adj.cuda(), batch_num_nodes=[sub_nodes.cpu().numpy()])
                print("pred debug: ", self.pred[0][graph_idx], pred_try, pred, inv_pred)
                # if torch.argmax(pred[0]) != pred_label:
                flips[pred_label] += 1.0
                # if torch.argmax(inv_pred[0]) == pred_label:
                inv_flips[pred_label] += 1.0
                total[pred_label] += 1.0

                pred_try = nn.Softmax(dim=0)(pred_try[0])
                pred_t = torch.from_numpy(self.pred[0][graph_idx]).float().cuda()
                pred_t = nn.Softmax(dim=0)(pred_t)

                pos_diff[pred_label] += (pred_t[pred_label] - pred[0][pred_label]).item()
                inv_diff[pred_label] += (pred_t[pred_label] - inv_pred[0][pred_label]).item()

                # loss = loss + explainer.loss(pred, pred_label)
                masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()

                topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=topk)

                # topk_adj, topk_x = noise_utils.filterGT(masked_adj, sub_adj[0], x, h_edges[graph_idx])

                print("Adj: ", np.sum(topk_adj), np.sum(sub_adj[0]))

                topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0), batch_num_nodes=[sub_nodes.cpu().numpy()])
                # pred_topk, _ = self.model(topk_x, topk_adj_t.unsqueeze(0), batch_num_nodes=[sub_nodes.cpu().numpy()])

                pred_topk = nn.Softmax(dim=0)(pred_topk[0])

                topk_inv_diff[pred_label] += (pred_t[pred_label] - pred_topk[pred_label]).item()

                if self.args.post_processing:
                    masked_adj = accuracy_utils.getModifiedMask(masked_adj, sub_adj[0], sub_nodes.cpu().numpy())

                if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
                    if self.args.post_processing:
                        masked_adj = accuracy_utils.getModifiedMask(masked_adj, sub_adj[0], sub_nodes.cpu().numpy())
                    ht_edges = h_edges[graph_idx]
                    # ht_edges = {}
                    # for k, v in h_edges[graph_idx].items():
                    #     if v > 0.5:
                    #         ht_edges[k] = 1.0

                    # AUC.clearAUC()

                    AUC.addEdges2(masked_adj, ht_edges)

                    # mean_auc += AUC.getAUC()

                if torch.argmax(pred_topk) == pred_label:
                    topk_inv_flips[pred_label] += 1.0

                variance = np.sum(np.abs(masked_adj - 0.5) * sub_adj.squeeze()) / np.sum(sub_adj)
                ep_variance += variance

                masked_adj_sfmx = np.exp(masked_adj) / np.sum(np.exp(masked_adj))

                orig_adj = sub_adj[0]

                if self.args.bmname != 'synthetic' and self.args.bmname != 'old_synthetic':
                    h_nodes = noise_utils.getTopKNodes(masked_adj, self.num_nodes[graph_idx].cpu().item())
                    imp_nodes = explain.getImportantNodes(masked_adj, 8)

                else:
                    imp_nodes = explain.getImportantNodes(masked_adj, 8)
                    h_nodes = []
                    # h_nodes = explain.getHnodes(graph_idx, 0)
                    # h_nodes.extend(explain.getHnodes(graph_idx, 1))
                    h_nodes = accuracy_utils.getHNodes(graph_idx, sub_label_nodes, sub_label_array, self.args)

                    ht_edges = accuracy_utils.getHTEdges(h_nodes, sub_adj[0])

                    AUC.addEdges2(masked_adj, ht_edges)

                    # AUC.addEdges(masked_adj, h_nodes, sub_adj[0], dataset='synthetic')

                    mAP_s = accuracy_utils.getmAP(masked_adj, h_nodes)
                    mAP += mAP_s

                    top4_acc, top6_acc, top8_acc = accuracy_utils.getAcc(imp_nodes, h_nodes)

                    avg_top4_acc += top4_acc
                    avg_top6_acc += top6_acc
                    avg_top8_acc += top8_acc

                stats.update(masked_adj, imp_nodes, graph_idx)

                if args.draw_graphs:

                    gt_mask = sub_adj[0] - 0.9
                    for e in ht_edges.keys():
                        gt_mask[e[0], e[1]] = 1.0
                        gt_mask[e[1], e[0]] = 1.0

                    # gt_mask
                    accuracy_utils.saveAndDrawGraph(gt_mask, sub_adj[0], sub_feat[0],
                                                    self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx,
                                                    prob=pred_t[pred_label],
                                                    plt_path=None, adj_mask_bool=True, prefix="gt_")

                    accuracy_utils.saveAndDrawGraph(masked_adj, sub_adj[0], sub_feat[0],
                                                    self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx,
                                                    prob=pred_t[pred_label],
                                                    plt_path=None, adj_mask_bool=True)
                    # topk_adj
                    accuracy_utils.saveAndDrawGraph(None, topk_adj, sub_feat[0], self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx, prob=pred_topk[pred_label],
                                                    plt_path=None, adj_mask_bool=False)

                if self.args.inverse_noise:
                    adj_np = adj[0].cpu().numpy()
                    new_adj, added_edges, removed_edges = noise_utils.addNoiseToGraphInverse(masked_adj, adj_np, None,
                                                                                             h_nodes, sub_nodes,
                                                                                             self.args.noise_percent)
                    pred_removed_edges += removed_edges

                    new_adj_t = torch.from_numpy(new_adj).float().cuda()

                    pred_masked, _ = self.model(x.cuda(), new_adj_t.unsqueeze(0),
                                                batch_num_nodes=[sub_nodes.cpu().numpy()])
                    pred_masked_sfmx = torch.nn.functional.softmax(pred_masked[0]).detach().cpu().numpy()
                    pred_sfmx = torch.nn.functional.softmax(
                        torch.from_numpy(self.pred[0][graph_idx]).float()).cpu().numpy()
                    pred_diff = np.abs(pred_sfmx[pred_label] - pred_masked_sfmx[pred_label])
                    avg_pred_diff += pred_diff
                    # print("pred debug: ", self.pred[0][graph_idx], pred_try, pred, torch.nn.functional.softmax(pred_masked))

                acc_count += 1.0
                print("adj: ", np.sum(masked_adj), np.sum(adj.cpu().numpy()))
                mask_density = np.sum(masked_adj) / np.sum(adj.cpu().numpy())
                avg_mask_density += mask_density
                label = self.label[graph_idx]

                if self.args.noise:
                    for n_iter in range(noise_iters):
                        for nh in noise_handlers:
                            try:
                                noise_feat, noise_adj = nh.sample(sub_feat[0], sub_adj[0], sub_nodes)
                            except: 
                                continue

                            emb_noise = self.model.getEmbeddings(noise_feat.unsqueeze(0), noise_adj.unsqueeze(0), [sub_nodes.cpu().numpy()])
                            pred_n, masked_adj_n, _, _, _ = explainer((noise_feat, emb_noise[0], noise_adj, tmp, label, sub_nodes), training=False)
                            masked_adj_n = masked_adj_n.cpu().detach() * noise_adj

                            nh.update(masked_adj, masked_adj_n.cpu().detach().numpy(), sub_adj[0], noise_adj.cpu().detach().numpy(), None, graph_idx)
        if not args.noise:
            global_noise_count = 1.0
            noise_diff_count = 1.0
        else:
            global_noise_count = noise_diff_count = 1
        eval_dir = os.path.dirname(self.args.exp_path)
        eval_file = "eval_" + self.args.bmname + "_" + self.args.explainer_method + ".txt"

        eval_file = os.path.join(eval_dir, eval_file)

        myfile = open(eval_file, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}".format(self.args.bloss_version))
        myfile.write("\n ckpt dir: {}".format(self.args.ckptdir))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))
        myfile.write("\n explainer params sum: {}, model params sum: {}".format(explainer_sum, model_sum))

        myfile.write("\n use comb: {},  size cf: {}, ent cf {}".format(use_comb_mask, size_cf, ent_cf))

        if self.args.bmname == 'synthetic' or self.args.bmname == 'old_synthetic':
            print(
                "Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count,
                                                                                avg_top6_acc / acc_count,
                                                                                avg_top8_acc / acc_count)
            )
            myfile.write("\n Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count,
                                                                                            avg_top6_acc / acc_count,
                                                                                            avg_top8_acc / acc_count)
                         )

            print(
                "mAP score: {}".format(mAP / acc_count)
            )
            myfile.write("\n mAP score: {}".format(mAP / acc_count))

            print(
                "noisy mAP score: {}".format(noise_mAP / global_noise_count)
            )
            myfile.write("\n noisy mAP score: {}".format(noise_mAP / global_noise_count))

            # print(
            #     "mean_auc  score: {}".format(mean_auc/ acc_count)
            # )

            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))
            if global_noise_count > 2.:
                n_auc_score = noise_AUC.getAUC()
            else:
                n_auc_score = -1.0
            print(
                "Noise ROC AUC score: {}".format(n_auc_score)
            )
            myfile.write("\n Noise ROC AUC score: {}".format(n_auc_score))

        if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))

        total[total < 0.5] = 1.0

        print(
            "noise percent: {}, inverse noise: {}".format(self.args.noise_percent, self.args.inverse_noise)
        )
        myfile.write("\n noise percent: {}, inverse noise: {}".format(self.args.noise_percent, self.args.inverse_noise))

        print(
            "avg removed edges: {}".format(avg_removed_edges / global_noise_count)
        )
        myfile.write("\n avg removed edges: {}".format(avg_removed_edges / global_noise_count))

        print(
            "pred removed edges: {}".format(pred_removed_edges / acc_count)
        )
        myfile.write("\n pred removed edges: {}".format(pred_removed_edges / acc_count))

        print(
            "avg added edges: {}".format(avg_add_edges / global_noise_count)
        )
        myfile.write("\n avg added edges: {}".format(avg_add_edges / global_noise_count))
        print(
            "avg adj diff: {}".format(avg_adj_diff / global_noise_count)
        )
        print(
            "avg noise diff: {}".format(avg_noise_diff / noise_diff_count)
        )
        myfile.write("\n avg noise diff: {}".format(avg_noise_diff / noise_diff_count))

        print(
            "avg pred diff: {}".format(avg_pred_diff / acc_count)
        )
        myfile.write("\n avg pred diff: {}".format(avg_pred_diff / acc_count))

        print(
            "skipped iters: {}".format(skipped_iters)
        )
        myfile.write("\n skipped iters: {}".format(skipped_iters))

        print(
            "Average mask density: {}".format(avg_mask_density / acc_count)
        )
        myfile.write("\n Average mask density: {}".format(avg_mask_density / acc_count))

        print(
            "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total, topk,
                                                                          topk_inv_diff / total)

        )

        myfile.write("\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total,
                                                                               topk_inv_diff / total))

        print("Variance: ", ep_variance / acc_count)
        myfile.write("\n Variance: {}".format(ep_variance / acc_count))

        print("Flips: ", flips)
        print("inv Flips: ", inv_flips)
        print("topk inv Flips: ", topk_inv_flips)
        print("Incorrect preds: ", incorrect_preds)
        print("Total: ", total)

        myfile.write(
            "\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}, Total: {}".format(flips,
                                                                                                               inv_flips,
                                                                                                               self.args.topk,
                                                                                                               topk_inv_flips,
                                                                                                               incorrect_preds,
                                                                                                               total))

        # print(
        #     "Average rule accuracy: {}".format(rule_label_match / acc_count)
        # )
        # myfile.write("\n Average rule accuracy: {}".format(rule_label_match / acc_count))


        myfile.write("\n \n")

        print(stats)
        myfile.write("\n \n")
        myfile.write(str(stats))
        if self.args.noise:
            for nh in noise_handlers:
                print(nh)
                myfile.write("\n")
                myfile.write(str(nh))


            print("NOISE SUMMARY")
            myfile.write("\n\n NOISE SUMMARY \n")
            for nh in noise_handlers:
                print(nh.summary())
                myfile.write("\n")
                myfile.write(str(nh.summary()))

        print("SUMMARY")
        myfile.write("\n \n SUMMARY \n")
        myfile.write(str(stats.summary()))
        print(stats.summary())
        myfile.close()





    def eval_graphs(self, args, graph_indices, explainer):
        if self.args.apply_filter and self.args.bmname == 'Mutagenicity':
            # h_edges = accuracy_utils.gethedgesmutag()
            graph_indices, h_edges = accuracy_utils.filterMutag2(graph_indices, self.label, self.feat, self.adj, self.num_nodes)
            # random.shuffle(graph_indices)

        if args.draw_graphs:
            random.shuffle(graph_indices)

            graph_indices = graph_indices[:5]
            # graph_indices = [600, 1100, 1200, 1500, 2500, 2858, 2201, 2777, 228, 231]



        def shuffle_forward(l):
            order = list(range(len(l)))
            random.shuffle(order)
            return order

        global global_noise_count
        global noise_diff_count
        global avg_add_edges
        global avg_removed_edges
        ep_variance= 0.


        incorrect_preds = 0.

        self.model.eval()
        num_classes = self.pred[0][0].shape[0]
        flips = np.zeros((num_classes))

        inv_flips = np.zeros((num_classes))
        topk_inv_flips = np.zeros((num_classes))
        pos_diff = np.zeros((num_classes))
        inv_diff = np.zeros((num_classes))
        topk_inv_diff = np.zeros((num_classes))
        total = np.zeros((num_classes))

        masked_adjs = []
        skipped_iters = 0.
        logging_graphs = False
        avg_top4_acc = 0.
        avg_top6_acc = 0.
        avg_top8_acc = 0.
        avg_noise_diff = 0.
        noise_diff_count = 0.
        avg_adj_diff = 0.
        acc_count = 0.

        rule_top8_acc = 0.

        avg_mask_density = 0.
        mAP = 0.
        noise_mAP = 0.
        AUC = accuracy_utils.AUC()
        noise_AUC = accuracy_utils.AUC()

        avg_pred_diff = 0.
        pred_removed_edges = 0.
        topk = self.args.topk

        graph_indices = list(graph_indices)
        np.random.shuffle(graph_indices)
        explainer.eval()
        explainer_sum = 0.0
        model_sum = 0.0
        mean_auc = 0.0

        noise_iters = 1
        noise_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        noise_handlers = [noise_utils.NoiseHandler("RCExplainer", self.model, self, noise_percent=x) for x in noise_range]


        stats = accuracy_utils.Stats("PGExplainer_Boundary", self)

        for p in explainer.parameters():
            explainer_sum += torch.sum(p).item()
        for p in self.model.parameters():
            model_sum += torch.sum(p).item()
        print("sum of params of loaded explainer: {}".format(explainer_sum))
        print("sum of params of loaded model: {}".format(model_sum))

        # for graph_idx in graph_indices:
            # order = list(range(self.num_nodes[graph_idx].item()))
            # order = list(range(self.adj.shape[1]))
            # order = orders[graph_idx]
            # rand_order = shuffle_forward(order)
            # rand_order = rand_orders[graph_idx]
            # self.feat[graph_idx, rand_order, :] = self.feat[graph_idx, order, :]
            # self.adj[graph_idx, rand_order, :] = self.adj[graph_idx, order, :]
            # self.adj[graph_idx, :, rand_order] = self.adj[graph_idx, :, order]

        for graph_idx in graph_indices:
            with torch.no_grad():
                print("doing for graph index: ", graph_idx)

                # extract features and

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
                neighbors = np.asarray(range(self.adj.shape[0]))  # 1,2,3....num_nodes
                # sub_emb = self.emb[graph_idx, :]

                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)
                # sub_emb = np.expand_dims(sub_emb, axis=0)

                adj = torch.tensor(sub_adj, dtype=torch.float)
                x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)
                # emb = torch.tensor(sub_emb, dtype=torch.float)

                pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
                if pred_label != label.item():
                    incorrect_preds += 1
                # print("Graph predicted label: ", pred_label)

                if self.args.bmname == 'Mutagenicity':
                    t0 = 5.0
                    t1 = 5.0
                else:
                    t0 = 0.5
                    t1 = 4.99

                tmp = float(t0 * np.power(t1 / t0, 1.0))
                # tmp = float(t0 * np.power(t1 / t0, 0.9))

                # tmp = float(t0 * np.power(t1 / t0, 0.0001))

                emb = self.model.getEmbeddings(x.cuda(), adj.cuda(), [sub_nodes.cpu().numpy()])


                pred, masked_adj, _, _, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes),training=False)
                # explainer.loss(pred, pred_label)



                pred_try, _ = self.model(x.cuda(), adj.cuda(), batch_num_nodes=[sub_nodes.cpu().numpy()])
                # print("pred debug: ", self.pred[0][graph_idx], pred_try, pred, inv_pred)
                # if torch.argmax(pred[0]) != pred_label:
                #     flips[pred_label] += 1.0
                # if torch.argmax(inv_pred[0]) == pred_label:
                #     inv_flips[pred_label] += 1.0
                # total[pred_label] += 1.0

                pred_try = nn.Softmax(dim=0)(pred_try[0])
                pred_t = torch.from_numpy(self.pred[0][graph_idx]).float().cuda()
                pred_t = nn.Softmax(dim=0)(pred_t)


                # pos_diff[pred_label] += (pred_t[pred_label] - pred[0][pred_label]).item()
                # inv_diff[pred_label] += (pred_t[pred_label] - inv_pred[0][pred_label]).item()





                # loss = loss + explainer.loss(pred, pred_label)
                masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()


                topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=topk)

                # topk_adj, topk_x = noise_utils.filterGT(masked_adj, sub_adj[0], x, h_edges[graph_idx])

                print("Adj: ", np.sum(topk_adj), np.sum(sub_adj[0]))



                topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0), batch_num_nodes=[sub_nodes.cpu().numpy()])
                # pred_topk, _ = self.model(topk_x, topk_adj_t.unsqueeze(0), batch_num_nodes=[sub_nodes.cpu().numpy()])

                pred_topk = nn.Softmax(dim=0)(pred_topk[0])

                topk_inv_diff[pred_label] += (pred_t[pred_label] - pred_topk[pred_label]).item()
                thresh_nodes = 15
                imp_nodes = explain.getImportantNodes(masked_adj, 8)
                stats.update(masked_adj, imp_nodes, graph_idx)
                if self.args.post_processing:
                    masked_adj = accuracy_utils.getModifiedMask(masked_adj, sub_adj[0], sub_nodes.cpu().numpy())


                if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
                    if self.args.post_processing:
                        masked_adj = accuracy_utils.getModifiedMask(masked_adj, sub_adj[0], sub_nodes.cpu().numpy())
                    ht_edges = h_edges[graph_idx]
                    # ht_edges = {}
                    # for k, v in h_edges[graph_idx].items():
                    #     if v > 0.5:
                    #         ht_edges[k] = 1.0

                    # AUC.clearAUC()

                    AUC.addEdges2(masked_adj, ht_edges)

                    # mean_auc += AUC.getAUC()





                if torch.argmax(pred_topk) == pred_label:
                    topk_inv_flips[pred_label] += 1.0

                variance = np.sum(np.abs(masked_adj - 0.5)*sub_adj.squeeze()) / np.sum(sub_adj)
                ep_variance += variance





                masked_adj_sfmx = np.exp(masked_adj)/np.sum(np.exp(masked_adj))




                orig_adj = sub_adj[0]



                if self.args.bmname != 'synthetic' and self.args.bmname != 'old_synthetic':
                    h_nodes = noise_utils.getTopKNodes(masked_adj, self.num_nodes[graph_idx].cpu().item())

                else:
                    imp_nodes = explain.getImportantNodes(masked_adj, 8)
                    h_nodes = []
                    # h_nodes = explain.getHnodes(graph_idx, 0)
                    # h_nodes.extend(explain.getHnodes(graph_idx, 1))
                    h_nodes = accuracy_utils.getHNodes(graph_idx, sub_label_nodes, sub_label_array, self.args)


                    ht_edges = accuracy_utils.getHTEdges(h_nodes, sub_adj[0])

                    AUC.addEdges2(masked_adj, ht_edges)

                    # AUC.addEdges(masked_adj, h_nodes, sub_adj[0], dataset='synthetic')


                    mAP_s = accuracy_utils.getmAP(masked_adj, h_nodes)
                    mAP += mAP_s

                    top4_acc, top6_acc, top8_acc = accuracy_utils.getAcc(imp_nodes, h_nodes)

                    avg_top4_acc += top4_acc
                    avg_top6_acc += top6_acc
                    avg_top8_acc += top8_acc

                if args.draw_graphs:

                    gt_mask = sub_adj[0] - 0.9
                    for e in ht_edges.keys():
                        gt_mask[e[0], e[1]] = 1.0
                        gt_mask[e[1], e[0]] = 1.0

                    # gt_mask
                    accuracy_utils.saveAndDrawGraph(gt_mask, sub_adj[0], sub_feat[0],
                                                    self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx,
                                                    prob=pred_t[pred_label],
                                                    plt_path=None, adj_mask_bool=True, prefix="gt_")

                    accuracy_utils.saveAndDrawGraph(masked_adj, sub_adj[0], sub_feat[0],
                                                    self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx,
                                                    prob=pred_t[pred_label],
                                                    plt_path=None, adj_mask_bool=True)
                    # topk_adj
                    accuracy_utils.saveAndDrawGraph(None, topk_adj, sub_feat[0], self.num_nodes[graph_idx].item(),
                                                    self.args,
                                                    label.item(), pred_label, graph_idx, prob=pred_topk[pred_label],
                                                    plt_path=None, adj_mask_bool=False)

                if self.args.inverse_noise:

                    adj_np = adj[0].cpu().numpy()
                    new_adj, added_edges, removed_edges = noise_utils.addNoiseToGraphInverse(masked_adj, adj_np, None,
                                                                                                    h_nodes, sub_nodes,
                                                                                                    self.args.noise_percent)
                    pred_removed_edges += removed_edges

                    new_adj_t = torch.from_numpy(new_adj).float().cuda()

                    pred_masked, _ = self.model(x.cuda(), new_adj_t.unsqueeze(0), batch_num_nodes=[sub_nodes.cpu().numpy()])
                    pred_masked_sfmx = torch.nn.functional.softmax(pred_masked[0]).detach().cpu().numpy()
                    pred_sfmx = torch.nn.functional.softmax(torch.from_numpy(self.pred[0][graph_idx]).float()).cpu().numpy()
                    pred_diff = np.abs(pred_sfmx[pred_label] - pred_masked_sfmx[pred_label])
                    avg_pred_diff += pred_diff
                    # print("pred debug: ", self.pred[0][graph_idx], pred_try, pred, torch.nn.functional.softmax(pred_masked))

                acc_count += 1.0
                print("adj: ", np.sum(masked_adj), np.sum(adj.cpu().numpy()))
                mask_density = np.sum(masked_adj) / np.sum(adj.cpu().numpy())
                avg_mask_density += mask_density
                label = self.label[graph_idx]


                if self.args.noise:
                    for n_iter in range(noise_iters):
                        for nh in noise_handlers:
                            try:
                                noise_feat, noise_adj = nh.sample(sub_feat[0], sub_adj[0], sub_nodes)
                            except: 
                                continue

                            emb_noise = self.model.getEmbeddings(noise_feat.unsqueeze(0), noise_adj.unsqueeze(0), [sub_nodes.cpu().numpy()])
                            pred_n, masked_adj_n, _, _, _ = explainer((noise_feat, emb_noise[0], noise_adj, tmp, label, sub_nodes), training=False)
                            masked_adj_n = masked_adj_n.cpu().detach() * noise_adj

                            nh.update(masked_adj, masked_adj_n.cpu().detach().numpy(), sub_adj[0], noise_adj.cpu().detach().numpy(), None, graph_idx)
        if not args.noise:
            global_noise_count = 1.0
            noise_diff_count = 1.0
        eval_dir = os.path.dirname(self.args.exp_path)
        eval_file = "eval_" + self.args.bmname + "_" + self.args.explainer_method +".txt"

        eval_file = os.path.join(eval_dir, eval_file)

        myfile = open(eval_file, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}".format(self.args.bloss_version))
        myfile.write("\n ckpt dir: {}".format(self.args.ckptdir))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))
        myfile.write("\n explainer params sum: {}, model params sum: {}".format(explainer_sum, model_sum))


        myfile.write("\n use comb: {},  size cf: {}, ent cf {}".format(use_comb_mask, size_cf, ent_cf))

        if self.args.bmname == 'synthetic' or self.args.bmname == 'old_synthetic':
            print(
                "Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count,
                                                                                avg_top6_acc / acc_count,
                                                                                avg_top8_acc / acc_count)
            )
            myfile.write("\n Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count,
                                                                                            avg_top6_acc / acc_count,
                                                                                            avg_top8_acc / acc_count)
                         )


            print(
                "mAP score: {}".format(mAP / acc_count)
            )
            myfile.write("\n mAP score: {}".format(mAP / acc_count))

            print(
                "noisy mAP score: {}".format(noise_mAP / global_noise_count)
            )
            myfile.write("\n noisy mAP score: {}".format(noise_mAP / global_noise_count))

            # print(
            #     "mean_auc  score: {}".format(mean_auc/ acc_count)
            # )

            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))
            if global_noise_count > 2.:
                n_auc_score = noise_AUC.getAUC()
            else:
                n_auc_score = -1.0
            print(
                "Noise ROC AUC score: {}".format(n_auc_score)
            )
            myfile.write("\n Noise ROC AUC score: {}".format(n_auc_score))

        if self.args.bmname == 'Mutagenicity' and self.args.apply_filter:
            print(
                "ROC AUC score: {}".format(AUC.getAUC())
            )
            myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))


        total[total<0.5] = 1.0

        print(
            "noise percent: {}, inverse noise: {}".format(self.args.noise_percent, self.args.inverse_noise)
        )
        myfile.write("\n noise percent: {}, inverse noise: {}".format(self.args.noise_percent, self.args.inverse_noise))

        print(
            "avg removed edges: {}".format(avg_removed_edges / global_noise_count)
        )
        myfile.write("\n avg removed edges: {}".format(avg_removed_edges / global_noise_count))

        print(
            "pred removed edges: {}".format(pred_removed_edges / acc_count)
        )
        myfile.write("\n pred removed edges: {}".format(pred_removed_edges / acc_count))

        print(
            "avg added edges: {}".format(avg_add_edges / global_noise_count)
        )
        myfile.write("\n avg added edges: {}".format(avg_add_edges / global_noise_count))
        print(
            "avg adj diff: {}".format(avg_adj_diff / global_noise_count)
        )
        print(
            "avg noise diff: {}".format(avg_noise_diff / noise_diff_count)
        )
        myfile.write("\n avg noise diff: {}".format(avg_noise_diff / noise_diff_count))

        print(
            "avg pred diff: {}".format(avg_pred_diff / acc_count)
        )
        myfile.write("\n avg pred diff: {}".format(avg_pred_diff / acc_count))

        print(
            "skipped iters: {}".format(skipped_iters)
        )
        myfile.write("\n skipped iters: {}".format(skipped_iters))

        print(
            "Average mask density: {}".format(avg_mask_density / acc_count)
        )
        myfile.write("\n Average mask density: {}".format(avg_mask_density / acc_count))

        print(
            "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total, topk, topk_inv_diff/ total)

        )

        myfile.write("\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / total, inv_diff / total, topk_inv_diff/total))

        print("Variance: ", ep_variance/ acc_count)
        myfile.write("\n Variance: {}".format(ep_variance/ acc_count))

        print("Flips: ", flips)
        print("inv Flips: ", inv_flips)
        print("topk inv Flips: ", topk_inv_flips)
        print("Incorrect preds: ", incorrect_preds)
        print("Total: ", total)

        print(stats)
        myfile.write("\n \n")
        myfile.write(str(stats))
        if self.args.noise:
            for nh in noise_handlers:
                print(nh)
                myfile.write("\n")
                myfile.write(str(nh))

        myfile.write("\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}, Total: {}".format(flips, inv_flips, self.args.topk, topk_inv_flips, incorrect_preds, total))


        # print(
        #     "Average rule accuracy: {}".format(rule_label_match / acc_count)
        # )
        # myfile.write("\n Average rule accuracy: {}".format(rule_label_match / acc_count))

        myfile.close()




    # GRAPH EXPLAINER
    def explain_graphs(self, args, graph_indices, test_graph_indices=None):

        

        """
        Explain graphs.
        """

        graph_indices = list(graph_indices)




        # if self.emb is not None:
            # assert(self.model.embedding_dim * self.model.num_layers * 2 == self.emb[0, :].shape[1] * 2)

        explainers = []
        for i in range(2):
            explainer = ExplainModule(
                model = self.model, 
                num_nodes = self.adj.shape[1],
                emb_dims = self.model.embedding_dim * self.model.num_layers * 2, # TODO: fixme!
                device=self.device,
                args = self.args
            )
            explainers.append(explainer)

        if self.args.bmname == "synthetic" or self.args.bmname == "old_synthetic":
            load_sublabel_nodes(self.args)

        if self.args.eval or self.args.exp_path != "":
            if self.args.eval and self.args.exp_path == "":
                print("no explainer file to load")
                exit()
            else:
                print("loading initial explainer ckpt from : ", self.args.exp_path)


            try:
                state_dict = torch.load(self.args.exp_path)
            except:
                state_dict = torch.load(self.args.exp_path + 'rcexplainer.pth.tar')
            exp_state_dict = explainer.state_dict()
            for name, param in state_dict.items():
                if name in exp_state_dict and not ("model" in name):
                    exp_state_dict[name].copy_(param)
            explainer.load_state_dict(exp_state_dict)



            if self.args.eval:
                self.eval_graphs_2(args, graph_indices, explainer)
                exit()



        if self.args.bmname == "synthetic" or self.args.bmname == "old_synthetic":

            if max(graph_indices) > 3000:
                size = 7000
            else:
                size = 3000

            width = 12

        elif self.args.bmname == "Mutagenicity":
            if self.args.apply_filter:
                graph_indices = accuracy_utils.filterMutag(graph_indices, self.label)
                random.shuffle(graph_indices)

            size = 3000
            width = 14
        elif self.args.bmname == "PROTEINS":
            size = 1044
        elif self.args.bmname == "REDDIT-BINARY":
            size = 552
        elif self.args.bmname == "ER_MD":
            size = 444
        elif self.args.bmname == "MUTAG":
            size = 1864
        elif self.args.bmname == "COLLAB":
            size = 3000
        elif self.args.bmname == "NCI1":
            size = 2877
        elif self.args.bmname == "bbbp":
            size = 1427
        elif self.args.bmname == "Mutagenicity":
            size = 3000
        elif self.args.bmname == "BA_2Motifs":
            size = 700
        else:
            print(self.args.bmname + " not found!")
            assert (False)

        train_data = (self.adj[:size], self.feat[:size], self.label[:size], self.num_nodes[:size])
        val_data = (self.adj[size - 100:], self.feat[size - 100:], self.label[size - 100:], self.num_nodes[size - 100:])

        rule_dicts = []
        for i in range(2):

            rule_dict = extract.extract_rules(self.args.bmname, train_data, val_data, args, self.model.state_dict(), graph_indices=None, pool_size=args.pool_size)
            rule_dicts.append(rule_dict)

        schedulers = []
        optimizers = []

        for i in range(2):

            params_optim = []
            for name,param in explainers[i].named_parameters():
                if "model" in name:
                    continue
                params_optim.append(param)

            scheduler, optimizer = train_utils.build_optimizer(self.args, params_optim)
            schedulers.append(scheduler)
            optimizers.append(optimizer)

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
        log_name = self.args.prefix + "_logdir"
        log_path = os.path.join(self.args.ckptdir, log_name)
        if os.path.isdir(log_path):
            print("log dir already exists and will be overwritten")
            time.sleep(5)
        else:
            os.mkdir(log_path)

        training = True
        if self.args.gumbel:
            training = False

        log_file = self.args.prefix + "log_rcexplainer_" + self.args.bmname + ".txt"
        log_file_path = os.path.join(log_path, log_file)
        myfile = open(log_file_path, "a")

        myfile.write("\n \n \n {}".format(self.args.bmname))
        myfile.write("\n method: {}".format(self.args.explainer_method))
        myfile.write("\n bloss version: {}, node mask: {}, apply filter: {}".format(self.args.bloss_version, self.args.node_mask, self.args.apply_filter))
        myfile.write("\n exp_path: {}".format(self.args.exp_path))
        myfile.write("\n opt: {}".format(self.args.opt_scheduler))
        myfile.write("\n gumbel: {}, training: {}".format(self.args.gumbel, training))

        myfile.write("\n lr: {}, bound cf: {}, size cf: {}, ent cf {}, inv cf {}".format(self.args.lr, self.args.boundary_c, self.args.size_c, self.args.ent_c,  self.args.inverse_boundary_c))
        myfile.close()
        bloss_prev = None

        ep_count = 0.
        loss_ep = [0., 0.]

        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            myfile = open(log_file_path, "a")
            loss = 0
            logging_graphs=False

            masked_adjs = []
            rule_top4_acc = 0.
            rule_top6_acc = 0.
            rule_top8_acc = 0.

            rule_acc_count = 0.
            avg_mask_density = 0.
            mAP = 0.
            bloss_ep = 0.
            flips = 0.
            inv_flips = 0.
            pos_diff = 0.
            inv_diff = 0.
            topk_inv_diff = 0.
            topk_inv_flips = 0.

            incorrect_preds = 0.
            ep_variance = 0.

            AUC = accuracy_utils.AUC()
            stats = accuracy_utils.Stats("PGExplainer_Boundary", self)

            AUC_ind = AverageMeter()


            masked_adjs = []
            if self.args.bmname == 'synthetic' and self.args.bmname == 'old_synthetic':
                loss_ep = 0.

            np.random.shuffle(graph_indices)
            for i in range(2):
                explainers[i].train()

            for graph_idx in graph_indices:
                # print("doing for graph index: ", graph_idx)

                # extract features and

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
                
                sub_adj = np.expand_dims(sub_adj, axis=0)
                sub_feat = np.expand_dims(sub_feat, axis=0)

                adj   = torch.tensor(sub_adj, dtype=torch.float)
                x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
                label = torch.tensor(sub_label, dtype=torch.long)

                # if self.emb is not None:
                #     sub_emb = self.emb[graph_idx, :]
                #     sub_emb = np.expand_dims(sub_emb, axis=0)
                #     emb = torch.tensor(sub_emb, dtype=torch.float)

                # else:
                emb = self.model.getEmbeddings(x, adj, batch_num_nodes=[sub_nodes.cpu().numpy()])
                emb = emb.clone().detach()


                gt_pred, gt_embedding = self.model(x.cuda(), adj.cuda(), batch_num_nodes=[sub_nodes.cpu().numpy()])

                pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
                if pred_label != label.item():
                    incorrect_preds += 1


                if self.args.bmname != 'synthetic' and self.args.bmname != 'old_synthetic':
                    t0 = 5.0
                    t1 = 5.0
                else:
                    t0 = 0.5
                    t1 = 4.99


                tmp = float(t0 * np.power(t1 / t0, epoch /self.args.num_epochs))

                m_adj_tmps = []
                failed = False

                for i in range(2):
                    try:
                        rule_ix = rule_dicts[i]['idx2rule'][graph_idx]
                        rule = rule_dicts[i]['rules'][rule_ix]
                        rule_label = rule['label']
                    except KeyError:
                        failed = True
                        continue

                    boundary_list = []
                    for b_num in range(len(rule['boundary'])):

                        boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
                        if self.args.gpu:
                            boundary = boundary.cuda()
                        boundary_label = rule['boundary'][b_num]['label']
                        boundary_list.append(boundary)



                    pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainers[i]((x[0], emb[0], adj[0], tmp, label, sub_nodes), training=training)
                    m_adj_tmp = masked_adj.clone().cpu().detach().numpy() * sub_adj.squeeze()

                    m_adj_tmps.append(m_adj_tmp)
                    # if pred is not None:
                    #     print("debug pred: ", self.pred[0][graph_idx], pred, gt_pred)

                    # if inv_pred is not None:
                    #     print("inv pred: ", inv_pred)

                    # print("prefix: ", self.args.prefix)
                    loss, bloss_s = explainers[i].loss(pred, pred_label, graph_embedding=graph_embedding,
                                boundary_list=boundary_list, gt_embedding=gt_embedding, inv_embedding=inv_embedding)
                    # bloss_ep += bloss_s
                    if self.args.bmname != 'synthetic' and self.args.bmname != 'old_synthetic':
                        if ep_count < 200:
                            loss_ep[i] += loss
                            ep_count += 1.0
                        else:
                            ep_count = 0.
                            optimizers[i].zero_grad()
                            loss_ep[i].backward()
                            optimizers[i].step()
                            loss_ep = [0., 0.]
                    else:
                        loss_ep[i] += loss
                
                if failed:
                    continue
                AUC_sep = accuracy_utils.AUC()

                gt = m_adj_tmps[0]
                gt = coo_matrix(gt)
                topk = 8
                num_elem = len(np.nonzero(masked_adj)[0]) // 2 - 1
                topk = min(topk, num_elem)
                threshold = sorted(zip(gt.row, gt.col, gt.data), key = lambda x: x[2], reverse=True)[topk * 2][2]
                gt_edges = m_adj_tmps[0] >= threshold

                AUC_sep.addEdgesFromAdj(m_adj_tmps[1], gt_edges)
                try:
                    AUC_ind.update(AUC_sep.getAUC())
                except:
                    print("FAILED")
                pred_t = torch.from_numpy(self.pred[0][graph_idx]).float().cuda()
                pred_t = nn.Softmax(dim=0)(pred_t)

                if self.args.boundary_c > 0.0:

                    if torch.argmax(pred[0]) != pred_label:
                        flips += 1.0

                    pos_diff += (pred_t[pred_label] - pred[0][pred_label]).item()

                if self.args.inverse_boundary_c > 0.0:
                    if torch.argmax(inv_pred[0]) == pred_label:
                        inv_flips += 1.0
                    inv_diff += (pred_t[pred_label] - inv_pred[0][pred_label]).item()

                masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()

                thresh_nodes = 15
                imp_nodes = explain.getImportantNodes(masked_adj, 8)
                stats.update(masked_adj, imp_nodes, graph_idx)

                # masked_adj[orders[graph_idx], :] = masked_adj[rand_orders[graph_idx], :]
                # masked_adj[:, orders[graph_idx]] = masked_adj[:, rand_orders[graph_idx]]

                #why node_idx is set to 0?

                variance = np.sum(np.abs(masked_adj - 0.5)*sub_adj.squeeze()) / np.sum(sub_adj)
                ep_variance += variance

                if epoch%10 == 0:
                    topk_adj = noise_utils.filterTopK(masked_adj, sub_adj[0], k=self.args.topk)
                    topk_adj_t = torch.from_numpy(topk_adj).float().cuda()
                    pred_topk, _ = self.model(x.cuda(), topk_adj_t.unsqueeze(0),
                                              batch_num_nodes=[sub_nodes.cpu().numpy()])

                    pred_topk = nn.Softmax(dim=0)(pred_topk[0])

                    topk_inv_diff += (pred_t[pred_label] - pred_topk[pred_label]).item()

                    if torch.argmax(pred_topk) == pred_label:
                        topk_inv_flips += 1.0


                if self.args.bmname == 'synthetic' or self.args.bmname == 'old_synthetic':
                    imp_nodes = explain.getImportantNodes(masked_adj, 8)

                    # h_nodes = explain.getHnodes(graph_idx, 0)
                    # h_nodes.extend(explain.getHnodes(graph_idx, 1))
                    h_nodes = accuracy_utils.getHNodes(graph_idx, sub_label_nodes, sub_label_array, self.args)
                    ht_edges = accuracy_utils.getHTEdges(h_nodes, sub_adj[0])
                    AUC.addEdges2(masked_adj, ht_edges)


                    # AUC.addEdges(masked_adj, h_nodes, sub_adj[0], dataset='synthetic')

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
                # masked_adjs.append(masked_adj)
            if self.args.bmname == 'synthetic' and self.args.bmname == 'old_synthetic':
                optimizer.zero_grad()
                loss_ep.backward()
                optimizer.step()

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


            myfile.write("\n epoch: {}, loss: {}".format(epoch, loss.item()))

            # plot cmap for graphs' node features
            io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")
            # if self.args.bmname =='synthetic' or self.args.bmname =='old_synthetic':
            #     print(
            #         "Rule wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(rule_top4_acc / rule_acc_count,
            #                                                                     rule_top6_acc / rule_acc_count,
            #                                                                     rule_top8_acc / rule_acc_count)
            #     )
            #     myfile.write("\n Rule wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(rule_top4_acc / rule_acc_count,
            #                                                                     rule_top6_acc / rule_acc_count,
            #                                                                     rule_top8_acc / rule_acc_count))

            #     print(
            #         "mAP score: {}".format(mAP / rule_acc_count)
            #     )
            #     myfile.write("\n mAP score: {}".format(mAP / rule_acc_count))

            #     print(
            #         "ROC AUC score: {}".format(AUC.getAUC())
            #     )
            #     myfile.write("\n ROC AUC score: {}".format(AUC.getAUC()))


            # print(
            #     "Soft mask pred change: {}".format(softMaskPredChange.avg)
            # )
            # print(
            #     "Soft mask pred prob change: {}".format(softMaskPredProbChange.avg)
            # )

            # print(
            #     "Hard mask pred change [4, 6, 8]: {}".format(hardMaskPredChange.avg)
            # )
            # print(
            #     "Hard mask pred prob change [4, 6, 8]: {}".format(hardMaskPredProbChange.avg)
            # )

            # print(
            #     "Keep mask pred change [4, 6, 8]: {}".format(keepMaskPredChange.avg)
            # )
            # print(
            #     "Keep mask pred prob change [4, 6, 8]: {}".format(keepMaskPredProbChange.avg)
            # )
            # print(
            #     "Average mask density: {}".format(avg_mask_density / rule_acc_count)
            # )
            # myfile.write("\n Average mask density: {}".format(avg_mask_density / rule_acc_count))

            # print(
            #     "pos diff: {}, inv diff: {}".format(pos_diff / rule_acc_count, inv_diff / rule_acc_count)

            # )
            # print("Variance: ", ep_variance/rule_acc_count)
            # myfile.write("\n Variance: {}".format(ep_variance/rule_acc_count))


            # print("Flips: ", flips)
            # print("inv Flips: ", inv_flips)

            # print(
            #     "pos diff: {}, inv diff: {}, k: {}, topk inv diff: {}".format(pos_diff / rule_acc_count,
            #                                                                   inv_diff / rule_acc_count, self.args.topk,
            #                                                                   topk_inv_diff / rule_acc_count)

            # )

            # myfile.write(
            #     "\n pos diff: {}, inv diff: {}, topk inv diff: {}".format(pos_diff / rule_acc_count, inv_diff / rule_acc_count,
            #                                                               topk_inv_diff / rule_acc_count))

            # print("topk inv Flips: ", topk_inv_flips)
            # print("Incorrect preds: ", incorrect_preds)
            # myfile.write("\n flips: {}, Inv flips: {}, topk: {}, topk Inv flips: {}, Incorrect preds: {}".format(flips, inv_flips, self.args.topk,
            #                                                                                            topk_inv_flips,
            #                                                                                            incorrect_preds))

            # print("Total graphs optimized: ", len(graph_indices))

            # if avg_mask_density/rule_acc_count < 0.45:
            #     self.args.size_c = self.args.size_c*0.8
            # elif avg_mask_density/rule_acc_count > 0.55:
            #     self.args.size_c = self.args.size_c*1.2

            print(stats)
            print("\n \n AUC SUMMARY\n \n")
            print(AUC_ind.avg, AUC_ind.count)

            myfile.write("\n New size_cf : {}".format(self.args.size_c))

            if epoch %10 == 0:
                explainer_sum = 0.0
                model_sum = 0.0
                for p in explainer.parameters():
                    explainer_sum += torch.sum(p).item()
                for p in self.model.parameters():
                    model_sum += torch.sum(p).item()

                # f_path = './ckpt/explainer3_synthetic_data_3label_3sublabel_pgeboundary' + '.pth.tar'
                myfile.write("\n explainer params sum: {}, model params sum: {}".format(explainer_sum, model_sum))

                f_path = self.args.prefix + "explainer_" + self.args.bmname + "_pgeboundary.pth.tar"
                save_path = os.path.join(log_path, f_path)
                torch.save(explainer.state_dict(), save_path)
                myfile.write("\n ckpt saved at {}".format(save_path))
            if epoch % 100 == 0:
                # f_path = './ckpt/explainer3_synthetic_data_3label_3sublabel_pgeboundary' + '.pth.tar'
                f_path = self.args.prefix + "explainer_" + self.args.bmname + "_pgeboundary_ep_" + str(epoch) + ".pth.tar"
                save_path = os.path.join(log_path, f_path)
                torch.save(explainer.state_dict(), save_path)
                myfile.write("\n ckpt saved at {}".format(save_path))
            myfile.close()
        exit()
        # if self.args.bmname == 'Mutagenicity':
        #     pickle.dump(masked_adjs, open("../data/Mutagenicity/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        # else:
        #     pickle.dump(masked_adjs, open("../data/synthetic_data_3label_3sublabel/masked_adjs_explainer_{}.p".format(self.args.multigraph_class),"wb"))
        # print("Flips: ", flips)
        # print("Incorrect preds: ", incorrect_preds)
        # torch.save(explainer.state_dict(), 'synthetic_data_3label_3sublabel_pgexplainer' + '.pth.tar')
        myfile.close()
        if test_graph_indices is not None:
            print("EVALUATING")
            self.eval_graphs_2(args, test_graph_indices, explainer)
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
            "sample_bias": 0
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
            bias = self.coeffs['sample_bias']
            random_noise = bias + torch.FloatTensor(log_alpha.shape).uniform_(bias, 1.0-bias)
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
            adjs = coo_matrix(adj.cpu())
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

        # if not self.args.gumbel:
        #     values = self.concrete_sample(self.values, beta=tmp, training=training)
        # else:
        #     values = self.concrete_sample(self.values, beta=tmp, training=False)

            # values = torch.nn.functional.relu(self.values)
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

        if self.args.gumbel:
            sym_mask = sym_mask*adj
            self.mask = sym_mask
            # max_tensor = (torch.zeros_like(sym_mask).cuda() + 0.75)*adj
            # sym_mask = torch.max(sym_mask, max_tensor)

            sym_mask = sym_mask.unsqueeze(2)
            inv_sym_mask = 1.0 - sym_mask
            eps = 1e-5
            sym_mask = torch.log(torch.cat([sym_mask, inv_sym_mask],dim=2)+eps)
            sym_mask = torch.nn.functional.gumbel_softmax(sym_mask, hard=True, tau=0.5, dim=2)
            sym_mask = sym_mask[:,:,0]
            # sym_mask = torch.nn.functional.gumbel_softmax(sym_mask, hard=True, tau=0.5, dim=1)
            triu_np = np.ones((sym_mask.shape[0],sym_mask.shape[1]))
            triu_np = np.triu(triu_np, 1)
            triu_mask = torch.from_numpy(triu_np).float().cuda()
            # sym_mask = torch.triu(sym_mask, diagonal=1)
            sym_mask = sym_mask*triu_mask
            sym_mask = (sym_mask + sym_mask.T)


        else:
            self.mask = sym_mask

        # sym_mask = self.mask

        # sym_mask = (sym_mask.clone() + sym_mask.clone().T) / 2      # Maybe needs a .clone()
        sym_mask = (sym_mask + sym_mask.T) / 2

        masked_adj = torch.mul(adj, sym_mask)
        self.masked_adj = masked_adj

        if self.args.gumbel:
            inverse_mask = (1.0 - sym_mask)
            inverse_masked_adj = adj * inverse_mask
            self.inverse_masked_adj = inverse_masked_adj
            print("mask debug: ", torch.sum(self.mask), torch.sum(sym_mask), torch.sum(adj), torch.sum(inverse_mask))


        else:
            inverse_mask = (1.0 - sym_mask)
            orig_adj = adj + 0.
            inverse_masked_adj = torch.mul(adj, inverse_mask)
            self.inverse_masked_adj = inverse_masked_adj





        if self.args.node_mask:
            inverse_node_mask = torch.max(inverse_masked_adj, dim=1)[0]
            # sum_adj = torch.sum(adj, dim=1)
            # ones = torch.ones_like(sum_adj).float().cuda()
            # inverse_node_mask = torch.sum(inverse_masked_adj, dim=1)/torch.max(sum_adj, ones)

            self.inverse_node_mask = inverse_node_mask


        x = torch.unsqueeze(x.detach().requires_grad_(True),0).to(torch.float32)        # Maybe needs a .clone()
        adj = torch.unsqueeze(self.masked_adj,0).to(torch.float32 )
        x.to(self.device)
        if sub_nodes is not None:
            sub_num_nodes_l = [sub_nodes.cpu().numpy()]
        else:
            sub_num_nodes_l = None

        inv_embed = None
        inv_res = None
        res = None
        g_embed = None
        if self.args.boundary_c > 0.0:
            if node_idx is not None:
                # node mode
                output, g_embed = self.model(x, adj, batch_num_nodes=sub_num_nodes_l, new_node_idx=[node_idx])

                res = self.softmax(output[0][node_idx, :])
            else:
                # graph mode
                output, g_embed = self.model(x, adj, batch_num_nodes=sub_num_nodes_l)
                res = self.softmax(output)

        if self.args.inverse_boundary_c > 0.0:
            if self.args.node_mask:
                one_hot_rand = np.eye(x.shape[2])[np.random.choice(x.shape[2], x.shape[1])]
                one_hot_rand[sub_num_nodes_l[0]:,:] = 0.
                one_hot_rand_t = torch.from_numpy(one_hot_rand).float().cuda()
                one_hot_rand_t = one_hot_rand_t.unsqueeze(0)


                inverse_node_mask = inverse_node_mask.unsqueeze(1).expand(x.shape[1],x.shape[2])
                inverse_node_mask = inverse_node_mask.unsqueeze(0)
                x = x*inverse_node_mask
                # x = x*inverse_node_mask + (1.0 - inverse_node_mask)*one_hot_rand_t

                # inv_adj = orig_adj.unsqueeze(0).float()

            inv_adj = torch.unsqueeze(self.inverse_masked_adj, 0).to(torch.float32)

            if node_idx is not None:
                # node mode
                inv_output, inv_embed = self.model(x, inv_adj, batch_num_nodes=sub_num_nodes_l, new_node_idx=[node_idx])

                inv_res = self.softmax(inv_output[0][node_idx, :])
            else:
                # graph mode
                inv_output, inv_embed = self.model(x, inv_adj, batch_num_nodes=sub_num_nodes_l)

                inv_res = self.softmax(inv_output)


        return res, masked_adj, g_embed, inv_embed, inv_res

    def loss(self, pred, pred_label, node_idx=None, graph_embedding=None, boundary_list=None, gt_embedding=None, inv_embedding=None):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        pred_loss = torch.zeros(1).cuda()
        if pred is not None:
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


        if self.args.bloss_version == 'proj':
            boundary_loss = 0.
            for boundary in boundary_list:
                boundary_loss += torch.norm(torch.sum((graph_embedding - gt_embedding) * boundary[:20]))
            boundary_loss = (boundary_loss / len(boundary_list))  # 0.3
            net_boundary_loss = boundary_loss
        elif self.args.bloss_version == 'sigmoid':
            if self.args.boundary_c < 0.0:
                boundary_loss = torch.zeros(1).cuda()
            else:
                boundary_loss = 0.
                sigma = 1.0
                for boundary in boundary_list:
                    gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                    ft_proj = torch.sum(graph_embedding * boundary[:20]) + boundary[20]
                    boundary_loss += torch.torch.sigmoid(-1.0 * sigma * (gt_proj * ft_proj))
                boundary_loss = self.args.boundary_c * (boundary_loss / len(boundary_list))

            if self.args.inverse_boundary_c < 0.0:
                net_boundary_loss = boundary_loss
            else:
                sigma = 1.0
                inv_losses = []
                for boundary in boundary_list:

                    gt_proj = torch.sum(gt_embedding * boundary[:20]) + boundary[20]
                    inv_proj = torch.sum(inv_embedding * boundary[:20]) + boundary[20]
                    # print("inv: ", gt_proj, inv_proj)
                    inv_loss = torch.torch.sigmoid(sigma * (gt_proj * inv_proj))
                    inv_losses.append(inv_loss)

                inv_losses_t = torch.stack(inv_losses)
                # print("debug: ", inv_losses, torch.min(inv_losses_t))

                inverse_boundary_loss = self.args.inverse_boundary_c * torch.min(inv_losses_t)

                net_boundary_loss = boundary_loss + inverse_boundary_loss
                # print("Boundary loss: {}, Inverse boundary loss: {}".format(boundary_loss.item(),
                #                                                             inverse_boundary_loss.item()))



        else:
            assert (False)



        # boundary_loss = 0.
        # for boundary in boundary_list:
        #     boundary_loss += torch.norm(torch.sum((graph_embedding - gt_embedding) * boundary[:20]))
        # boundary_loss = self.args.boundary_c*(boundary_loss / len(boundary_list))

        # size
        mask = self.mask
        # if self.mask_act == "sigmoid":
        #     mask = torch.sigmoid(self.mask)
        # elif self.mask_act == "ReLU":
        #     mask = nn.functional.relu(self.mask)
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

        # # l2 norm
        # l2norm = 0
        # for name, parameter in self.elayers.named_parameters():
        #     if "weight" in name:
        #         l2norm = l2norm + torch.norm(parameter)
        # l2norm = self.coeffs['weight_decay']*l2norm.clone()

        # loss = pred_loss + size_loss + mask_ent_loss
        loss = net_boundary_loss + size_loss + mask_ent_loss

        # print("net boundary loss: ", net_boundary_loss.item(), "pred_loss: ", pred_loss.item(), "size_loss: ", size_loss.item(), "mask ent loss: ", mask_ent_loss.item())
        # print("total loss: ", loss.item())
        return loss, boundary_loss.item()


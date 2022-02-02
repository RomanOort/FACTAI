import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, ndcg_score
import matplotlib.pyplot as plt
import os
import statistics
import torch

from explainer import explain


class AdjacencyError(Exception):
    pass


def evaluate_interpretation(model, masked_adj, graph_mode, adj, feat, label, sub_nodes, graph_idx=None, node_idx=None,
                            mode=None, topk=[4, 6, 8], sparsity=[0, 0.5]):
    batch_num_nodes = [sub_nodes.cpu().numpy()] if sub_nodes is not None else None
    logits, _ = model(feat, adj, batch_num_nodes=batch_num_nodes)

    if not graph_mode:
        logits = logits[0][node_idx_new]
    else:
        logits = logits[0]

    pred_label = np.argmax(logits.cpu().detach().numpy())

    # get prediction changes from new adj

    def get_graph_pred_changes(m_adj, m_x):
        if graph_mode:
            logits_masked, _ = model(m_x, m_adj, batch_num_nodes=batch_num_nodes)
        else:
            logits_masked, _ = model(m_x, m_adj, batch_num_nodes=batch_num_nodes, new_node_idx=[node_idx_new])

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

    if mode == 'edge-fidelity':
        '''
        Note: we evaluate edge fidelity
        '''
        pred_changes = []
        pred_prob_changes = []
        sparsities = []

        flat_masked_adj = masked_adj.flatten()
        nnz = np.count_nonzero(adj.clone().detach().cpu().numpy())
        flat_masked_adj = np.sort(flat_masked_adj)[::-1]
        adj = adj.cpu()

        for sp in sparsity:
            threshold = flat_masked_adj[int(nnz * (1 - sp))]

            fid_masked_adj = masked_adj > threshold

            mask = torch.ones(masked_adj.shape)
            mask -= fid_masked_adj

            fid_masked_adj = torch.mul(adj, mask)
            sparsities.append(torch.sum(fid_masked_adj) / torch.sum(adj))

            pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, feat)
            pred_changes.append(pred_change)
            pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

        return pred_changes, pred_prob_changes, sparsities


class Stats(object):
    def __init__(self, name, explainer, model, gt_nodes=None, gt_edges=None, graph_mode=True):
        self.name = name
        self.explainer = explainer
        self.model = model

        self.count = 0
        self.graph_mode = graph_mode

        self.hardtopk = [2, 8, 12, 16]
        self.keeptopk = [2, 8, 12, 16]
        self.edgesparsity = [0, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
        self.nodesparsity = [0, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
        self.gt_nodes = gt_nodes
        self.gt_edges = gt_edges

        if gt_nodes is not None:
            self.top4_acc = 0.
            self.top6_acc = 0.
            self.top8_acc = 0.
        if gt_edges is not None:
            self.mAP = 0.
            self.AUC = AUC()

        self.avg_mask_density = explain.AverageMeter()

        self.softMaskPredChange = explain.AverageMeter()
        self.softMaskPredProbChange = explain.AverageMeter()
        self.hardMaskPredChange = explain.AverageMeter(size=len(self.hardtopk))
        self.hardMaskPredProbChange = explain.AverageMeter(size=len(self.hardtopk))
        self.keepMaskPredChange = explain.AverageMeter(size=len(self.keeptopk))
        self.keepMaskPredProbChange = explain.AverageMeter(size=len(self.keeptopk))
        self.edgeFidelityPredChange = explain.AverageMeter(size=len(self.edgesparsity))
        self.edgeFidelityPredProbChange = explain.AverageMeter(size=len(self.edgesparsity))
        self.edgeFidelitySparsity = explain.AverageMeter(size=len(self.edgesparsity))
        self.nodeFidelityPredChange = explain.AverageMeter(size=len(self.nodesparsity))
        self.nodeFidelityPredProbChange = explain.AverageMeter(size=len(self.nodesparsity))
        self.nodeFidelitySparsity = explain.AverageMeter(size=len(self.nodesparsity))
        self.nodeFidelityMaxPredChange = explain.AverageMeter(size=len(self.nodesparsity))
        self.nodeFidelityMaxPredProbChange = explain.AverageMeter(size=len(self.nodesparsity))
        self.nodeFidelityMaxSparsity = explain.AverageMeter(size=len(self.nodesparsity))

    def update(self, masked_adj, imp_nodes, adj, feat, label, sub_nodes):
        # pred_change, pred_prob_change = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='soft-mask')
        # self.softMaskPredChange.update(pred_change)
        # self.softMaskPredProbChange.update(pred_prob_change)
        # pred_change, pred_prob_change = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='hard-mask', topk=self.hardtopk)
        # self.hardMaskPredChange.update(pred_change)
        # self.hardMaskPredProbChange.update(pred_prob_change)
        # pred_change, pred_prob_change = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='keep-mask', topk=self.keeptopk)
        # self.keepMaskPredChange.update(pred_change)
        # self.keepMaskPredProbChange.update(pred_prob_change)
        pred_change, pred_prob_change, sparsity = evaluate_interpretation(self.model, masked_adj, self.graph_mode, adj,
                                                                          feat, label, sub_nodes, mode='edge-fidelity',
                                                                          sparsity=self.edgesparsity)
        self.edgeFidelityPredChange.update(pred_change)
        self.edgeFidelityPredProbChange.update(pred_prob_change)
        self.edgeFidelitySparsity.update(sparsity)
        # pred_change, pred_prob_change, sparsity = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='node-fidelity-min-feat-mask', sparsity=self.nodesparsity)
        # self.nodeFidelityPredChange.update(pred_change)
        # self.nodeFidelityPredProbChange.update(pred_prob_change)
        # self.nodeFidelitySparsity.update(sparsity)
        # pred_change, pred_prob_change, sparsity = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='node-fidelity-max', sparsity=self.nodesparsity)
        # self.nodeFidelityMaxPredChange.update(pred_change)
        # self.nodeFidelityMaxPredProbChange.update(pred_prob_change)
        # self.nodeFidelityMaxSparsity.update(sparsity)
        # pred_change, pred_prob_change, sparsity = self.explainer.evaluate_interpretation(masked_adj, self.graph_mode, graph_idx=graph_idx, node_idx=node_idx, mode='edge-fidelity-k', sparsity=self.edgesparsity)

        if self.gt_edges is not None:
            mAP_s = getmAPEdges(masked_adj, self.gt_edges[graph_idx])
            self.mAP += mAP_s
            self.AUC.addEdges(masked_adj, self.gt_edges[graph_idx])
        if self.gt_nodes is not None:
            top4_acc_s, top6_acc_s, top8_acc_s = getAcc(imp_nodes, self.gt_nodes[graph_idx])
            self.top4_acc += top4_acc_s
            self.top6_acc += top6_acc_s
            self.top8_acc += top8_acc_s

        self.count += 1

    def __str__(self):
        retval = ""
        retval += "Reporting statistics\n"
        retval += "Samples: {}\n".format(self.count)
        if self.gt_edges is not None:
            retval += "mAP: {}\n".format(self.mAP / self.count)
            retval += "AUC: {}\n".format(self.AUC.getAUC())
        if self.gt_nodes is not None:
            retval += "top 4 acc: {}\t top 6 acc: {}\t top 8 acc: {}\n".format(self.top4_acc, self.top6_acc,
                                                                               self.top8_acc)
        # retval += "Soft mask prediction change: {}\n Soft mask confidence change: {}\n".format(self.softMaskPredChange.avg, self.softMaskPredProbChange.avg)
        # retval += "Hard mask prediction change: {}\n Hard mask confidence change: {} for topk {}\n".format(self.hardMaskPredChange.avg, self.hardMaskPredProbChange.avg, self.hardtopk)
        # retval += "Keep mask prediction change: {}\n Keep mask confidence change: {} for topk {}\n".format(self.keepMaskPredChange.avg, self.keepMaskPredProbChange.avg, self.keeptopk)
        retval += "Edge fidelity prediction change: {}\n Edge fidelity confidence change: {} for sparsity {}\n".format(
            self.edgeFidelityPredChange.avg, self.edgeFidelityPredProbChange.avg, self.edgeFidelitySparsity.avg)
        # retval += "Node fidelity prediction change: {}\n Node fidelity confidence change: {} for sparsity {}\n".format(self.nodeFidelityPredChange.avg, self.nodeFidelityPredProbChange.avg, self.nodeFidelitySparsity.avg)
        # retval += "Node max fidelity prediction change: {}\n Node max fidelity confidence change: {} for sparsity {}\n".format(self.nodeFidelityMaxPredChange.avg, self.nodeFidelityMaxPredProbChange.avg, self.nodeFidelityMaxSparsity.avg)
        retval += "Mask density: {}\n".format(self.avg_mask_density.avg)

        summary_sp = ""
        summary_fid = ""
        for i, sp in enumerate(self.edgeFidelitySparsity.avg):
            fid = self.edgeFidelityPredProbChange.avg[i]
            summary_sp += str(sp) + ','
            summary_fid += str(fid) + ','
        retval += "Sparsity, {}\nFidelity, {}\n".format(summary_sp, summary_fid)

        return retval

    def summary(self):
        retval = ""
        summary_sp = ""
        summary_fid = ""
        for i, sp in enumerate(self.edgeFidelitySparsity.avg):
            fid = self.edgeFidelityPredProbChange.avg[i]
            summary_sp += "{:1.4f}".format(sp) + ','
            summary_fid += "{:1.4f}".format(fid) + ','
        retval += "Sparsity, {}\nFidelity, {}\n".format(summary_sp, summary_fid)
        return retval

    def get_sparsity_fidelity(self):
        sparsity = []
        fidelity = []

        for i, sp in enumerate(self.edgeFidelitySparsity.avg):
            fid = self.edgeFidelityPredProbChange.avg[i]
            sparsity.append(sp)
            fidelity.append(fid)

        return sparsity, fidelity


def getHNodes(graph_idx, sub_label_nodes, sub_label_array, args):
    h_nodes = []
    if args.bmname == 'synthetic':
        h_nodes.extend(sub_label_nodes[graph_idx, 0, 0, :].tolist())
        h_nodes.extend(sub_label_nodes[graph_idx, 1, 0, :].tolist())
    else:
        for ix in range(sub_label_array.shape[1]):
            if sub_label_array[graph_idx, ix] == -1:
                continue
            h_nodes.extend(sub_label_nodes[graph_idx, ix, 0, :].tolist())
    return h_nodes


def getHTEdges(h_nodes_all, orig_adj):
    assert (len(h_nodes_all) == 8)
    G = nx.from_numpy_array(orig_adj)

    h_edges = {}
    for ix in range(2):
        h_nodes = h_nodes_all[ix * 4:((ix + 1) * 4)]
        edges = G.edges(h_nodes)
        for e in edges:
            if e[0] in h_nodes and e[1] in h_nodes:
                h_edges[e] = 1
    return h_edges


class AUC(object):
    def __init__(self):
        self.reals = []
        self.preds = []
        self.FN = 0

    def addEdges2(self, masked_adj, h_edges, h_nodes_all=None, dataset=None):

        if dataset == 'synthetic':
            h_edges = {}
            h_nodes = h_nodes_all[:4]
            for i_n in range(len(h_nodes)):
                h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1
            h_nodes = h_nodes_all[4:]
            for i_n in range(len(h_nodes)):
                h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1

        adj = coo_matrix(masked_adj)

        for r, c in list(zip(adj.row, adj.col)):
            if (r, c) in h_edges.keys() or (c, r) in h_edges.keys():
                self.reals.append(1)
            else:
                self.reals.append(0)
            self.preds.append(masked_adj[r][c])

    def addEdgesFromAdj(self, masked_adj_n, gt_edges, dataset=None):
        adj = coo_matrix(masked_adj_n)
        gt_sparse = coo_matrix(gt_edges)

        # keep track of number of false negatives
        fn = set(zip(gt_sparse.row, gt_sparse.col)) - set(zip(adj.row, adj.col)) - set(zip(adj.col, adj.row))
        self.FN += len(fn)

        for r, c in list(zip(adj.row, adj.col)):
            if gt_edges[r, c] != 0 or gt_edges[c, r] != 0:
                self.reals.append(1)

            else:
                self.reals.append(0)

            self.preds.append(masked_adj_n[r][c])

    def addEdgesFromAdj_includeFN(self, masked_adj_n, gt_edges, dataset=None):
        adj = coo_matrix(masked_adj_n)
        gt_sparse = coo_matrix(gt_edges)

        # keep track of number of false negatives
        fn = set(zip(gt_sparse.row, gt_sparse.col)) - set(zip(adj.row, adj.col)) - set(zip(adj.col, adj.row))
        self.FN += len(fn)

        nonzero = set(zip(gt_sparse.row, gt_sparse.col)).union(set(zip(adj.row, adj.col)))

        for r, c in nonzero:
            if gt_edges[r, c] != 0 or gt_edges[c, r] != 0:
                self.reals.append(1)

            else:
                self.reals.append(0)

            self.preds.append(masked_adj_n[r][c])

    def addEdgesFromDict(self, masked_adj, h_edges, dataset=None):
        adj = coo_matrix(masked_adj)

        for r, c in list(zip(adj.row, adj.col)):
            if (r, c) in h_edges.keys() or (c, r) in h_edges.keys():
                self.reals.append(1)
            else:
                self.reals.append(0)
            self.preds.append(masked_adj[r][c])

    def getAUC(self):
        if len(self.reals) == 0 or len(self.preds) == 0:
            return 0

        print("AUC computation:")
        print(" Number of false negatives found:", self.FN)
        print(" Total number of edges in data:  ", len(self.preds))
        return roc_auc_score(self.reals, self.preds)

    def clearAUC(self):
        self.reals = []
        self.preds = []


def getmAPAdj(adj_n, adj, edge_thresh=7.9):
    nodes = adj_n.shape[0]
    argsort_adj = np.dstack(np.unravel_index(np.argsort(adj_n.ravel()), (nodes, nodes)))[0]
    edges_covered = {}
    rank = 1.0
    sum_precision = 0.
    pos_found = 0.
    for i in range(nodes * nodes - 1, -1, -1):
        x = argsort_adj[i][0]
        y = argsort_adj[i][1]

        if (x, y) in edges_covered or (y, x) in edges_covered:
            continue
        if adj[x, y] != 0 or adj[y, x] != 0:
            pos_found += 1.0
            sum_precision += (pos_found / rank)

        edges_covered[(x, y)] = 1
        rank += 1.0
        if pos_found > edge_thresh:  # 4 edges found
            break

    return sum_precision / pos_found


def getNDCGAdj(adj_true, adj_score, **kwargs):
    nodes = adj_true.shape[0]
    argsort_adj = np.dstack(np.unravel_index(np.argsort(adj_score.ravel()), (nodes, nodes)))[0]
    edges_covered = {}

    y_true = []
    y_score = []

    for i in range(nodes * nodes - 1, -1, -1):
        x = argsort_adj[i][0]
        y = argsort_adj[i][1]

        if (x, y) in edges_covered or (y, x) in edges_covered:
            continue
        if adj_true[x, y] != 0 or adj_true[y, x] != 0:
            if adj_true[x, y] != 0:
                y_true.append(adj_true[x, y])
            elif adj_true[y, x] != 0:
                y_true.append(adj_true[y, x])

            if adj_score[x, y] != 0:
                y_score.append(adj_score[x, y])
            elif adj_score[y, x] != 0:
                y_score.append(adj_score[y, x])
            else:
                y_score.append(0)

        edges_covered[(x, y)] = 1

    y_true = [sorted(y_true).index(x) for x in y_true]

    y_true = [y_true]
    y_score = [y_score]

    return ndcg_score(y_true, y_score)


def getmAPEdges(masked_adj, h_edges, edge_thresh=7.9):
    nodes = masked_adj.shape[0]
    argsort_adj = np.dstack(np.unravel_index(np.argsort(masked_adj.ravel()), (nodes, nodes)))[0]
    edges_covered = {}
    nodes_covered = {}
    rank = 1.0
    sum_precision = 0.
    pos_found = 0.
    valid_acc = True
    valid_node_acc = True
    sum_edges = 0.
    sum_nodes = 0.
    h_nodes = {}
    for (x, y) in h_edges:
        h_nodes[x] = 1
        h_nodes[y] = 1

    for i in range(nodes * nodes - 1, -1, -1):
        x = argsort_adj[i][0]
        y = argsort_adj[i][1]
        if valid_node_acc:
            if x not in nodes_covered:
                if x in h_nodes:
                    sum_nodes += 1.0
                nodes_covered[x] = 1
            if y not in nodes_covered:
                if y in h_nodes:
                    sum_nodes += 1.0
                nodes_covered[y] = 1

            if len(nodes_covered) > (len(h_nodes) - 0.1):
                valid_node_acc = False

        if (x, y) in edges_covered or (y, x) in edges_covered:
            continue
        if (x, y) in h_edges or (y, x) in h_edges:
            pos_found += 1.0
            sum_precision += (pos_found / rank)
            if valid_acc:
                sum_edges += 1.0

        edges_covered[(x, y)] = 1
        if len(edges_covered) >= edge_thresh:
            valid_acc = False
        rank += 1.0
        if pos_found > edge_thresh:  # 4 edges found
            break
    return sum_precision / pos_found, sum_edges / (edge_thresh + 0.1), sum_nodes / float(len(h_nodes))


def getmAPsyn4(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()
    old_idx = nbrs[new_idx]
    start_idx = old_idx - ((old_idx + 5) % 6)  # for size 6
    # start_idx = old_idx - ((old_idx + 2) % 9) #for cycle size 9

    for ix in range(start_idx, start_idx + 5, 1):
        # for ix in range(start_idx, start_idx + 8, 1):

        assert (full_adj[ix, ix + 1] > 0.5)
        # if ix not in nbrs_list:
        #     continue
        # if ix+1 not in nbrs_list:
        #     continue
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

        # nix = ix+2
        # if nix > start_idx + 5:
        #     nix = start_idx
        # assert (full_adj[ix, nix] > 0.5)
        # new_u = nbrs_list.index(ix)
        # new_v = nbrs_list.index(nix)
        # assert (orig_adj[new_u, new_v] > 0.5)
        # h_edges[(new_u, new_v)] = 1

    assert (ix + 1 == start_idx + 5)
    assert (full_adj[ix + 1, start_idx] > 0.5)
    if ix + 1 in nbrs_list and start_idx in nbrs_list:
        new_u = nbrs_list.index(start_idx)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    # assert (full_adj[start_idx + 5, start_idx + 1] > 0.5)
    # new_u = nbrs_list.index(start_idx + 5)
    # new_v = nbrs_list.index(start_idx + 1)
    # assert (orig_adj[new_u, new_v] > 0.5)
    # h_edges[(new_u, new_v)] = 1
    e_thresh = len(h_edges) - 0.1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=e_thresh), h_edges

    # return getmAPEdges(masked_adj, h_edges, edge_thresh = 5.9)


def getmAPsyn3(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    start_idx = old_idx - ((old_idx + 6) % 9)

    grid_G = nx.grid_graph([3, 3])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start_idx)
    for u, v in grid_G.edges():
        assert (full_adj[u, v] > 0.5)
        if u in nbrs_list:
            new_u = nbrs_list.index(u)
        else:
            continue
        if v in nbrs_list:
            new_v = nbrs_list.index(v)
        else:
            continue
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1
    e_thresh = len(h_edges) - 0.1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=e_thresh), h_edges
    # return getmAPEdges(masked_adj, h_edges, edge_thresh = 11.9), h_edges


def getmAPsyn2(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    assert (((old_idx >= 400) and (old_idx < 700)) or ((old_idx >= 1100) and (old_idx < 1400)))

    start_idx = old_idx - ((old_idx) % 5)
    for ix in range(start_idx, start_idx + 3, 1):
        assert (full_adj[ix, ix + 1] > 0.5)
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 3, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 3)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 4, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 1, start_idx + 4] > 0.5)
    new_u = nbrs_list.index(start_idx + 1)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1
    e_thresh = len(h_edges) - 0.1

    return getmAPEdges(masked_adj, h_edges, edge_thresh=e_thresh), h_edges

    # return getmAPEdges(masked_adj, h_edges, edge_thresh = 5.9), h_edges


def getmAPsyn1(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    assert (old_idx >= 400)

    start_idx = old_idx - ((old_idx) % 5)
    for ix in range(start_idx, start_idx + 3, 1):
        assert (full_adj[ix, ix + 1] > 0.5)
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 3, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 3)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 4, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 1, start_idx + 4] > 0.5)
    new_u = nbrs_list.index(start_idx + 1)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1
    e_thresh = len(h_edges) - 0.1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=e_thresh), h_edges


def gethedges(h_nodes_all):
    h_edges = {}
    h_nodes = h_nodes_all[:4]
    for i_n in range(len(h_nodes)):
        h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1
    h_nodes = h_nodes_all[4:]
    for i_n in range(len(h_nodes)):
        h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1
    return h_edges


def gethedgesmutag():
    hedges = torch.load('Mutagenicity_gt_edge_labels_new.p')
    for idx in hedges.keys():
        newdata = {}
        for data in hedges[idx]:
            r, c = data[0]
            v = data[1]
            newdata[(r, c)] = v
        hedges[idx] = newdata
    return hedges
    # return torch.load('Mutagenicity_our_gt.p')


def getmAPsyn4(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()
    old_idx = nbrs[new_idx]
    start_idx = old_idx - ((old_idx + 5) % 6)  # for size 6
    # start_idx = old_idx - ((old_idx + 2) % 9) #for cycle size 9

    for ix in range(start_idx, start_idx + 5, 1):
        # for ix in range(start_idx, start_idx + 8, 1):

        assert (full_adj[ix, ix + 1] > 0.5)
        # if ix not in nbrs_list:
        #     continue
        # if ix+1 not in nbrs_list:
        #     continue
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

        # nix = ix+2
        # if nix > start_idx + 5:
        #     nix = start_idx
        # assert (full_adj[ix, nix] > 0.5)
        # new_u = nbrs_list.index(ix)
        # new_v = nbrs_list.index(nix)
        # assert (orig_adj[new_u, new_v] > 0.5)
        # h_edges[(new_u, new_v)] = 1

    assert (ix + 1 == start_idx + 5)
    assert (full_adj[ix + 1, start_idx] > 0.5)
    if ix + 1 in nbrs_list and start_idx in nbrs_list:
        new_u = nbrs_list.index(start_idx)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    # assert (full_adj[start_idx + 5, start_idx + 1] > 0.5)
    # new_u = nbrs_list.index(start_idx + 5)
    # new_v = nbrs_list.index(start_idx + 1)
    # assert (orig_adj[new_u, new_v] > 0.5)
    # h_edges[(new_u, new_v)] = 1
    e_thresh = len(h_edges) - 0.1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=e_thresh), h_edges

    # return getmAPEdges(masked_adj, h_edges, edge_thresh = 5.9)


def getmAPsyn3(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    start_idx = old_idx - ((old_idx + 6) % 9)

    grid_G = nx.grid_graph([3, 3])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start_idx)
    for u, v in grid_G.edges():
        assert (full_adj[u, v] > 0.5)
        if u in nbrs_list:
            new_u = nbrs_list.index(u)
        else:
            continue
        if v in nbrs_list:
            new_v = nbrs_list.index(v)
        else:
            continue
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=11.9), h_edges


def getmAPsyn2(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    assert (((old_idx >= 400) and (old_idx < 700)) or ((old_idx >= 1100) and (old_idx < 1400)))

    start_idx = old_idx - ((old_idx) % 5)
    for ix in range(start_idx, start_idx + 3, 1):
        assert (full_adj[ix, ix + 1] > 0.5)
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 3, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 3)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 4, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 1, start_idx + 4] > 0.5)
    new_u = nbrs_list.index(start_idx + 1)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    return getmAPEdges(masked_adj, h_edges, edge_thresh=5.9), h_edges


def getmAPsyn1(masked_adj, orig_adj, pred, nbrs, new_idx, full_adj):
    h_edges = {}
    nbrs_list = nbrs.tolist()

    old_idx = nbrs[new_idx]
    assert (old_idx >= 400)

    start_idx = old_idx - ((old_idx) % 5)
    for ix in range(start_idx, start_idx + 3, 1):
        assert (full_adj[ix, ix + 1] > 0.5)
        new_u = nbrs_list.index(ix)
        new_v = nbrs_list.index(ix + 1)
        assert (orig_adj[new_u, new_v] > 0.5)
        h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 3, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 3)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 4, start_idx] > 0.5)
    new_u = nbrs_list.index(start_idx)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    assert (full_adj[start_idx + 1, start_idx + 4] > 0.5)
    new_u = nbrs_list.index(start_idx + 1)
    new_v = nbrs_list.index(start_idx + 4)
    assert (orig_adj[new_u, new_v] > 0.5)
    h_edges[(new_u, new_v)] = 1

    return getmAPEdges(masked_adj, h_edges, edge_thresh=5.9), h_edges


def getmAPNodes(masked_adj, orig_adj, pred, nbrs, new_idx):
    h_edges = {}

    old_idx = nbrs[new_idx]
    # if old_idx + 1 in nbrs and :

    ix = new_idx - 1
    t_ix = old_idx - 1
    ix_l = []
    lbl = pred[new_idx]

    while (True):
        if ix > 0:
            if nbrs[ix] == t_ix and pred[ix] > 0 and pred[ix] != 4 and (pred[ix] == lbl or pred[ix] == lbl - 1):
                ix_l.append(ix)
                lbl = pred[ix]
                ix = ix - 1
                t_ix = t_ix - 1
            else:
                break
        else:
            break

    ix = new_idx + 1
    t_ix = old_idx + 1
    lbl = pred[new_idx]

    while (True):
        if ix < len(nbrs):
            if nbrs[ix] == t_ix and pred[ix] > 0 and pred[ix] != 4 and (pred[ix] == lbl or pred[ix] == lbl + 1):
                ix_l.append(ix)
                lbl = pred[ix]
                ix = ix + 1
                t_ix = t_ix + 1
            else:
                break
        else:
            break

    ix_l.append(new_idx)
    if len(ix_l) != 5:
        print("new_idx: ", new_idx)
        print("pred: ", pred)
        print("nbrs: ", nbrs)
        print("ix_l: ", ix_l)
        # TODO: remove this (allow syn8 to train without proper mAP calculation)
        return 0
        assert (False)
    ix_l.sort()
    # print("ix_l: ", ix_l)

    assert (pred[ix_l].tolist() == [1, 1, 2, 2, 3] or pred[ix_l].tolist() == [5, 5, 6, 6, 7])

    assert orig_adj[(ix_l[0], ix_l[4])] > 0.5
    assert orig_adj[(ix_l[0], ix_l[1])] > 0.5
    assert orig_adj[(ix_l[0], ix_l[3])] > 0.5
    assert orig_adj[(ix_l[1], ix_l[2])] > 0.5
    assert orig_adj[(ix_l[1], ix_l[4])] > 0.5
    assert orig_adj[(ix_l[2], ix_l[3])] > 0.5

    h_edges[(ix_l[0], ix_l[4])] = 1
    h_edges[(ix_l[0], ix_l[1])] = 1
    h_edges[(ix_l[0], ix_l[3])] = 1
    h_edges[(ix_l[1], ix_l[2])] = 1
    h_edges[(ix_l[1], ix_l[4])] = 1
    h_edges[(ix_l[2], ix_l[3])] = 1
    return getmAPEdges(masked_adj, h_edges, edge_thresh=5.9)


def getmAP(masked_adj, h_nodes_all):
    h_edges = {}
    h_nodes = h_nodes_all[:4]
    for i_n in range(len(h_nodes)):
        h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1
    h_nodes = h_nodes_all[4:]
    for i_n in range(len(h_nodes)):
        h_edges[(h_nodes[i_n], h_nodes[((i_n + 1) % 4)])] = 1
    return getmAPEdges(masked_adj, h_edges)


def getAcc(imp_nodes, h_nodes):
    top4_acc = 0.
    top6_acc = 0.
    top8_acc = 0.
    hnode_count = 0.
    if len(h_nodes) == 0:
        return top4_acc, top6_acc, top8_acc
    for n in h_nodes:
        hnode_count += 1.0
        if n in imp_nodes[:4]:
            top4_acc += 1.0
            top6_acc += 1.0
            top8_acc += 1.0
        elif n in imp_nodes[4:6]:
            top6_acc += 1.0
            top8_acc += 1.0
        elif n in imp_nodes[6:8]:
            top8_acc += 1.0

    return top4_acc / hnode_count, top6_acc / hnode_count, top8_acc / hnode_count


def getNodeLabels(args):
    if args.bmname == 'Mutagenicity':
        return ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
    elif args.bmname == 'old_synthetic':
        return ['A', 'B', 'C', 'D', 'E', 'F']
    elif args.bmname == 'synthetic':
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    else:
        assert (False)


def gethedgesmutag():
    return torch.load('./ckpt/fisher_Mutagenicity/Mutagenicity_gt_edge_labels.p')


def saveAndDrawGraph(adj_mask, orig_adj, feat, num_nodes, args, glabel, pred, graph_id, prob, plt_path=None,
                     adj_mask_bool=True, prefix=""):
    # print(np.sum(adj_mask), adj_mask)
    # print(orig_adj)
    prob = "%.2f" % prob
    fig_size = (4, 3)
    dpi = 300

    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_labels = getNodeLabels(args)

    adj = orig_adj[:num_nodes, :num_nodes]
    feats = feat[:num_nodes, :]
    G = nx.from_numpy_array(adj)
    pos_layout = nx.kamada_kawai_layout(G, weight=None)

    labels_dict = {}
    for n in range(num_nodes):
        node_l = np.argmax(feats[n])
        labels_dict[n] = node_labels[node_l]

    fig, ax_l = plt.subplots(1, 1, figsize=(7, 7))
    color_list = [(0.9, 0.9, 0.9), (0.9, 0.7, 0.7), (0.9, 0.4, 0.4), (0.9, 0.1, 0.1)]
    edge_colors = []
    edge_lbl_dict = {}
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in G.edges.data('weight', default=1)]
    for (u, v, w) in G.edges.data('weight', default=1):
        if adj_mask_bool == True:
            edge_lbl_dict[(u, v)] = str("%.2f" % adj_mask[u, v])
        if w > 0.:
            if adj_mask_bool == False:
                edge_colors.append(0.99)
            elif adj_mask[u, v] > 0.75:
                edge_colors.append(0.99)
            elif adj_mask[u, v] > 0.5:
                edge_colors.append(0.70)
            elif adj_mask[u, v] > 0.25:
                edge_colors.append(0.33)
            else:
                edge_colors.append(0.1)

        else:
            edge_colors.append(0.0)

    nx.draw_networkx(G, pos=pos_layout, font_size=12,
                     node_size=150, labels=labels_dict, cmap=cmap, edge_color=edge_colors,
                     edge_cmap=plt.get_cmap("rainbow"),
                     edge_vmin=0.0,
                     edge_vmax=1.0,
                     vmax=8, vmin=0, alpha=0.8)

    if adj_mask_bool:
        nx.draw_networkx_edge_labels(G, pos=pos_layout, font_size=6,
                                     node_size=150, labels=labels_dict, edge_labels=edge_lbl_dict, cmap=cmap,
                                     edge_color=edge_colors,
                                     edge_cmap=plt.get_cmap("rainbow"),
                                     edge_vmin=0.,
                                     edge_vmax=1.0,
                                     vmax=8, vmin=0, alpha=0.8)

    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    # nx.draw_networkx(G, labels=labels_dict, edge_color = edge_colors, ax=ax_l[0])
    # nx.draw_networkx(G,  node_color = node_colors, ax=ax_l[1])
    if plt_path is None:
        pge_dir = os.path.dirname(args.exp_path)
        vis_dir_name = "visual"
        vis_dir = os.path.join(pge_dir, vis_dir_name)
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)
        log_file = "vis_log.txt"
        log_file_path = os.path.join(vis_dir, log_file)
        vislog = open(log_file_path, "a")
        vis_file = "{}_{}_lb{}_pred{}_{}.pdf".format(args.bmname, graph_id, glabel, pred, prob)
        if adj_mask_bool == False:
            vis_file = "topk" + str(args.topk) + "_" + vis_file
        vis_file = prefix + vis_file
        vislog.write("\n \n \n {}".format(args.exp_path))
        vislog.write("\n {}".format(vis_file))
        if adj_mask_bool:
            vislog.write("\n adj sum {}, mask sum {}".format(np.sum(orig_adj), np.sum(adj_mask)))
        else:
            vislog.write("\n topk sum {}".format(np.sum(orig_adj)))

        vislog.close()
        plt_path = os.path.join(vis_dir, vis_file)

    plt.savefig(plt_path)


def getModifiedMask(masked_adj, adj, num_nodes):
    threshold = 0.49

    masked_adj_discrete = np.zeros_like(masked_adj)
    masked_adj_discrete[masked_adj > threshold] = 1.

    inverse_adj_discrete = (1.0 - masked_adj_discrete) * adj
    masked_adj_discrete = masked_adj_discrete[:num_nodes, :num_nodes]
    inverse_adj_discrete = inverse_adj_discrete[:num_nodes, :num_nodes]

    G = nx.from_numpy_array(inverse_adj_discrete)
    G_masked = nx.from_numpy_array(masked_adj_discrete)

    mod_adj = (np.zeros_like(masked_adj) + 0.1) * adj
    G_orig = nx.from_numpy_array(adj)

    count = 0
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        if count == 0:
            count += 1
            continue

        l_c = list(c)
        # print(l_c)
        edges = G_orig.edges(l_c)
        mask_edges = G_masked.edges(l_c)
        avg_weight = 0.

        for em in mask_edges:
            # print("**em ", em)
            avg_weight += masked_adj[em[0], em[1]]
        avg_weight = avg_weight / max(1.0, len(mask_edges))

        # print(edges)
        for e in edges:
            if len(l_c) == 1:
                # print("**",e)
                mod_adj[e[0], e[1]] = avg_weight
                mod_adj[e[1], e[0]] = avg_weight

            elif e[0] in l_c and e[1] in l_c:
                # print("**",e)
                mod_adj[e[0], e[1]] = avg_weight
                mod_adj[e[1], e[0]] = avg_weight

    return mod_adj


def filterMutag(graph_indices, labels):
    h_edges = gethedgesmutag()
    new_g_indices = []
    for graph_idx in graph_indices:
        count = 0
        # check edge labels (should be >0 positive labels)
        for i in h_edges[graph_idx].items():
            if i[1] > 0:
                count += 1
        # should have label 0 (mutagenic)
        if labels[graph_idx].item() == 0 and count > 0:
            new_g_indices.append(graph_idx)
    return new_g_indices


def filterMutag2(graph_indices, labels, feats, adjs, num_nodes):
    # h_edges = gethedgesmutag()
    new_g_indices = []
    h_edges = {}
    for graph_idx in graph_indices:
        if labels[graph_idx].item() == 1:
            continue
        adj_np = adjs[graph_idx].cpu().numpy()
        feat_np = feats[graph_idx].cpu().numpy()
        num_nodes_np = num_nodes[graph_idx].item()
        n_cand = []
        for ix in range(num_nodes_np):
            if np.argmax(feat_np[ix]) == 4:
                n_cand.append(ix)

        if len(n_cand) == 0:
            continue
        all_edges = {}
        G = nx.from_numpy_array(adj_np)
        select = False
        for n_c in n_cand:
            edges_n = G.edges([n_c])
            count_1 = 0.
            count_3 = 0.
            edges_l = []
            for e in edges_n:
                assert e[0] == n_c
                label_e1 = np.argmax(feat_np[e[1]])
                if label_e1 == 1:
                    count_1 += 1.0
                    edges_l.append(e)
                elif label_e1 == 3:
                    count_3 += 1.0
                    edges_l.append(e)
            if count_1 > 1 or count_3 > 1:
                select = True
                for ed in edges_l:
                    all_edges[ed] = 1.0

        if select:
            new_g_indices.append(graph_idx)
            h_edges[graph_idx] = all_edges
    return new_g_indices, h_edges

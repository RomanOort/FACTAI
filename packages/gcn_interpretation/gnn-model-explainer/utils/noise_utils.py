import numpy as np
from explainer.explain_pgexplainer import *
import torch
from scipy.sparse import coo_matrix
import networkx as nx

from explainer.explain import AverageMeter
import utils.accuracy_utils as accuracy_utils

class PredictionError(Exception):
    pass

    

class NoiseHandler(object):
    def __init__(self, name, model, explainer, noise_percent=0.1, mode='auc'):
        self.sample_count = [0, 0]
        self.update_count = [0, 0]
        self.noise_percent = noise_percent
        self.explainer = explainer
        self.model = model
        
        # graph changes
        self.adj_diff = AverageMeter()
        self.feat_diff = AverageMeter()
        self.noise_diff = AverageMeter()

        # model changes
        self.pred_change = AverageMeter()
        self.prob_change = AverageMeter()

        self.topk = [4] # TODO: belangrijk!
        self.node_topk = [1, 2, 4, 8]
    
        # AUC mode
        self.mAP = AverageMeter()
        self.AUC = accuracy_utils.AUC()
        # self.AUC_ind = AverageMeter(size=2)
        self.AUC_ind = 0
        # self.AUC_ind.avg = 0
        self.nDCG = AverageMeter()

        self.AUC_roll = []

        # accuracy mode
        self.edge_accuracy = AverageMeter(size=len(self.topk))
        self.node_accuracy = AverageMeter(size=len(self.node_topk))

        # explainer changes
        self.name = name
        self.mode = mode
        self.stats = accuracy_utils.Stats("Noise_" + name, explainer, self.model)


    def sample(self, feat, adj, num_nodes, gt_nodes=None, gt_edges=None, mode='graph-noise'):
        if mode == 'graph-noise':
            new_feat, _ = addNoiseToGraphFeat(feat, num_nodes, gt_nodes=gt_nodes, noise_percent=self.noise_percent)
            new_adj, _ = addNoiseToGraphAdj(adj, num_nodes, gt_edges=gt_edges, noise_percent=self.noise_percent)
        elif mode == 'edge-noise':
            new_feat = np.copy(feat)
            new_adj, _ = addNoiseToGraphAdj(adj, num_nodes, gt_edges=gt_edges, noise_percent=self.noise_percent)
        elif mode == 'node-noise':
            new_feat, _ = addNoiseToGraphFeat(feat, num_nodes, gt_nodes=gt_nodes, noise_percent=self.noise_percent)
            new_adj = np.copy(adj)
        else:
            raise NotImplementedError

        adj_diff = np.sum(np.abs(adj - new_adj))
        feat_diff = np.sum(np.abs(feat - new_feat))

        new_feat = torch.tensor(new_feat, dtype=torch.float)
        new_adj = torch.tensor(new_adj, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)

        pred_noise, _ = self.model(new_feat.unsqueeze(0), new_adj.unsqueeze(0), batch_num_nodes=[num_nodes.cpu().numpy()])
        pred, _ = self.model(feat.unsqueeze(0), adj.unsqueeze(0), batch_num_nodes=[num_nodes.cpu().numpy()])
        pred_noise = pred_noise.detach().cpu()
        pred = pred.detach().cpu()
        noise_label = torch.argmax(pred_noise[0]).item()
        orig_label = torch.argmax(pred[0]).item()

        self.pred_change.update(noise_label != orig_label)
        self.prob_change.update(float(torch.abs(pred_noise[0][orig_label] - pred[0][orig_label])))

        self.sample_count[noise_label] += 1
        if noise_label == orig_label:
            self.adj_diff.update(adj_diff)
            self.feat_diff.update(feat_diff)

        return new_feat, new_adj, noise_label

    def update(self, masked_adj, masked_adj_noise, orig_adj, orig_adj_noise,
               imp_nodes, graph_idx, noise_label, x=None, sub_nodes=None):
        if self.mode == 'auc':
            self.update_auc(masked_adj, masked_adj_noise, orig_adj, orig_adj_noise, imp_nodes, graph_idx, noise_label, x=x, sub_nodes=sub_nodes)
        elif self.mode == 'acc':
            self.update_acc(masked_adj, masked_adj_noise, orig_adj, orig_adj_noise, imp_nodes, graph_idx, noise_label)

    # update as accuracy
    def update_acc(self, masked_adj, masked_adj_noise, orig_adj, orig_adj_noise, imp_nodes, graph_idx, noise_label):
        noise_diff = compare_adjs(masked_adj, masked_adj_noise, orig_adj, orig_adj_noise)
        
        gt = masked_adj
        gt_coo = coo_matrix(gt)

        sp = masked_adj_noise
        sp_coo = coo_matrix(sp)

        edge_accs = []

        for topk in self.topk:
            num_elem = len(np.nonzero(masked_adj)[0]) // 2 - 1
            # selecting all edges doesn't make sense

            topk = min(2 * topk, num_elem) // 2
            # topk edges

            threshold = sorted(zip(gt_coo.row, gt_coo.col, gt_coo.data), key = lambda x: x[2], reverse=True)[topk * 2][2]
            gt_edges = masked_adj >= threshold

            threshold = sorted(zip(sp_coo.row, sp_coo.col, sp_coo.data), key = lambda x: x[2], reverse=True)[topk * 2][2]
            sp_edges = masked_adj_noise >= threshold

            edge_accs.append(np.sum(np.multiply(gt_edges, sp_edges)) / np.sum(gt_edges))
        self.edge_accuracy.update(edge_accs)

        def getImportantNodes(masked_adj, max_topk=8):
            topk_nodes = []
            topk_count = 0

            nodes = masked_adj.shape[0]
            argsort_adj = np.dstack(np.unravel_index(np.argsort(masked_adj.ravel()), (nodes, nodes)))[0]
            for i in range(nodes*nodes-1,-1,-1):
                x = argsort_adj[i][0]
                y = argsort_adj[i][1]
                if x not in topk_nodes:
                    topk_nodes.append(x)
                    topk_count += 1
                if topk_count >= max_topk:
                    break
                if y not in topk_nodes:
                    topk_nodes.append(y)
                    topk_count += 1
                if topk_count >= max_topk:
                    break
            if len(topk_nodes) < max_topk:
                for j in range(max_topk - len(topk_nodes)):
                    topk_nodes.append(-1)
            return topk_nodes

        node_accs = []
            
        for topk in self.node_topk:
            gt_nodes = getImportantNodes(masked_adj, max_topk=topk)
            sp_nodes = getImportantNodes(masked_adj_noise, max_topk=topk)

            cnt_matching = 0
            cnt_total = 0
            for node in gt_nodes:
                if node != -1:
                    if node in sp_nodes:
                        cnt_matching += 1
                    cnt_total += 1

            node_accs.append(cnt_matching / cnt_total)
        self.node_accuracy.update(node_accs)

    # noise as AUC
    def update_auc(self, masked_adj, masked_adj_noise, orig_adj,
                   orig_adj_noise, imp_nodes, graph_idx, noise_label, x=None, sub_nodes=None):
        noise_diff = compare_adjs(masked_adj, masked_adj_noise, orig_adj, orig_adj_noise)
        
        gt = masked_adj
        gt = coo_matrix(gt)

        for topk in self.topk:
            num_elem = len(np.nonzero(masked_adj)[0]) // 2 - 1
            topk = min(topk, num_elem)

            threshold = sorted(zip(gt.row, gt.col, gt.data), key = lambda x: x[2], reverse=True)[topk * 2][2]
            # print("THRESHOLD", threshold)
            gt_edges = masked_adj >= threshold

            # AUC_ind = accuracy_utils.AUC()
            # AUC_ind.addEdgesFromAdj(masked_adj_noise, gt_edges)
            # self.AUC_ind.selective_update(AUC_ind.getAUC(), noise_label)

            mAP = accuracy_utils.getmAPAdj(masked_adj_noise, gt_edges, edge_thresh=topk)
            self.mAP.update(mAP)
            self.AUC.addEdgesFromAdj(masked_adj_noise, gt_edges)
        self.nDCG.update(accuracy_utils.getNDCGAdj(masked_adj, masked_adj_noise))

        self.noise_diff.update(noise_diff)
        # self.stats.update(masked_adj_noise, imp_nodes, torch.tensor(orig_adj_noise), x,noise_label,sub_nodes)
        self.update_count[noise_label] += 1

    def __str__(self):
        # assert(self.sample_count == self.update_count)
        retval = ""
        retval += "Evaluted {} samples with noise {}\n".format(self.sample_count, self.noise_percent * 100)
        retval += "Average adj diff: {}\n".format(self.adj_diff.avg)
        retval += "Average feat diff: {}\n".format(self.feat_diff.avg)
        retval += "Average noise diff: {}\n".format(self.feat_diff.avg)

        if self.mode == 'auc':
            retval += "Average mAP: {}\n".format(self.mAP.avg)
            retval += "ROC AUC: {}\n".format(self.AUC.getAUC())
            # retval += "AUC_ind: {}\n".format(self.AUC_ind.avg)
            retval += "nDCG: {}\n".format(self.nDCG.avg)
        elif self.mode == 'acc':
            retval += "Edge accuracy: {}\n".format(self.edge_accuracy.avg)
            retval += "Node accuracy: {}\n".format(self.node_accuracy.avg)
        retval += self.stats.__str__()
        return retval

    def summary(self):
        if self.mode == 'auc':
            return self.summary_auc()
        elif self.mode == 'acc':
            return self.summary_acc()

    def summary_auc(self):
        return "{}, {}, {}".format(self.noise_percent, 0, self.sample_count)
    def summary_acc(self):
        ret = ""

        for acc in self.edge_accuracy.avg:
            ret += str(acc) + ','
        for acc in self.node_accuracy.avg:
            ret += str(acc) + ','
        return "{}, {}".format(self.noise_percent, ret)

def getTopKNodes(masked_adj, total_nodes):
    # print(total_nodes)
    # print(masked_adj)

    knodes = int(total_nodes/6)
    node_sum = np.sum(masked_adj,axis=1)
    nodes_sort = node_sum.argsort()[::-1]
    # h_nodes = getImportantNodes(masked_adj, knodes)
    # print(node_sum)
    # #
    # print(nodes_sort)
    # # print(h_nodes_2)
    h_nodes = nodes_sort[:knodes]

    return h_nodes


def addNoiseToGraphFeat(feat, num_nodes, gt_nodes=None, noise_percent=0.1):
    feat_n = np.copy(feat)

    rand_nodes = int(num_nodes * noise_percent)
    changed_nodes = 0

    for i in range(num_nodes):
        if gt_nodes is not None and i in gt_nodes:
            continue
        if np.random.randint(num_nodes) < rand_nodes:
            feat_n[i, :] = 0
            rand_bit = np.random.randint(feat_n.shape[1])
            feat_n[i, rand_bit] = 1
            changed_nodes += 1

    return feat_n, changed_nodes
            
def addNoiseToGraphAdj(adj, num_nodes, gt_edges=None, noise_percent=0.1, p=0.5):
    adj_n = np.copy(adj)
    num_edges = np.sum(adj) // 2

    rand_edges = int(num_edges * noise_percent)
    added_edges = 0
    removed_edges = 0
    
    added_edge_list = []
    edge_weights = [0, 1]
    while len(added_edge_list) < rand_edges:
        v1 = np.random.randint(num_nodes)
        v2 = np.random.randint(num_nodes)
        if gt_edges is not None and (v1, v2) in gt_edges:
            continue
        new_edge_weight = np.random.choice(edge_weights, p=[p, 1-p])
        if new_edge_weight:
            added_edges += 1
        else:
            removed_edges += 1
        adj_n[v1, v2] = new_edge_weight
        adj_n[v2, v1] = new_edge_weight
        added_edge_list.append((v1, v2))

    return adj_n, (added_edges, removed_edges)

        

def addNoiseToGraph2(sub_adj, sub_feat, h_nodes, sub_nodes, noise_percent=10.0):



    # max_edges = max(int(np.sum(sub_adj)/10.0), 4)
    # rand_edges = np.random.randint(2, int(max_edges) + 10)
    new_feat = np.zeros_like(sub_feat)
    new_feat = new_feat + sub_feat

    rand_nodes = int((sub_nodes*noise_percent)/100.0)
    changed_nodes = 0.0
    for i in range(sub_nodes):
        if i in h_nodes:
            continue
        if np.random.randint(sub_nodes) < rand_nodes:
            new_feat[i,:] = 0.0
            rand_bit = np.random.randint(new_feat.shape[1])
            new_feat[i,rand_bit] = 1.0
            changed_nodes += 1.0
    return new_feat, changed_nodes

def filterTopK(sub_adj, adj, k=8):
    res_adj = np.zeros_like(adj)
    res_adj = res_adj + adj
    nodes = sub_adj.shape[0]
    covered_edges = {}
    argsort_adj = np.dstack(np.unravel_index(np.argsort(sub_adj.ravel()), (nodes, nodes)))[0]
    for i in range(nodes * nodes - 1, -1, -1):
        if len(covered_edges) >= k:
            break
        x = argsort_adj[i][0]
        y = argsort_adj[i][1]
        if (x,y) in covered_edges or (y,x) in covered_edges:
            continue
        # assert(adj[x,y] > 0.5)
        covered_edges[(x,y)] = 1.0
        res_adj[x,y] = 0.0
        res_adj[y,x] = 0.0

    return res_adj


def filterGT(sub_adj, adj, x, ht_edges):
    res_x = torch.zeros_like(x)
    res_x = res_x + x
    res_adj = np.zeros_like(adj)
    res_adj = res_adj + adj
    nodes = sub_adj.shape[0]

    # for e in ht_edges.keys():
    #     res_adj[e[0], e[1]] = 0.0
    #     res_adj[e[1],e[0]] = 0.0
    #
    #     res_x[0,e[1],:] = 0.
    #     res_x[0, e[0], :] = 0.
    #
    #     rand_bit = np.random.randint(x.shape[2])
    #     res_x[0, e[1], rand_bit] = 1.0
    #     rand_bit = np.random.randint(x.shape[2])
    #     res_x[0, e[0], rand_bit] = 1.0

    nodes_covered = []
    for e in ht_edges.keys():
        if e[0] not in nodes_covered:
            nodes_covered.append(e[0])
        if e[1] not in nodes_covered:
            nodes_covered.append(e[1])

    G_orig = nx.from_numpy_array(adj)
    edges = G_orig.edges(nodes_covered)
    for e in edges:
        res_adj[e[0], e[1]] = 0.0
        res_adj[e[1],e[0]] = 0.0


    return res_adj, res_x.cuda()



def addNoiseToGraphInverse(sub_adj, orig_adj, sub_feat, h_nodes, sub_nodes, noise_percent=10.0):

    positive_edges = 8.0

    # max_edges = max(int(np.sum(sub_adj)/10.0), 4)
    # rand_edges = np.random.randint(2, int(max_edges) + 10)



    rand_edges = int((positive_edges*noise_percent)/100.0)



    # rem_max_edges = max_edges + 6
    # rem_max_edges = min(rem_max_edges, int(np.sum(sub_adj)/2))

    # miss_edges = np.random.randint(2, rem_max_edges)
    miss_edges = int((positive_edges*noise_percent)/100.0)


    removed_edges = 0.0
    added_edges = 0.0

    nodes = sub_adj.shape[0]
    argsort_adj = np.dstack(np.unravel_index(np.argsort(sub_adj.ravel()), (nodes, nodes)))[0]

    for i in range(nodes * nodes - 1, -1, -1):
        x = argsort_adj[i][0]
        y = argsort_adj[i][1]
        orig_adj[x,y] = 0.
        orig_adj[y,x] = 0.
        removed_edges += 1.0
        if removed_edges > miss_edges:
            break

    return orig_adj, added_edges, removed_edges





def addNoiseToGraph(sub_adj, sub_feat, h_nodes, sub_nodes, noise_percent=10.0):



    # max_edges = max(int(np.sum(sub_adj)/10.0), 4)
    # rand_edges = np.random.randint(2, int(max_edges) + 10)
    new_adj = np.zeros_like(sub_adj)
    new_adj = new_adj + sub_adj


    rand_edges = int(((np.sum(sub_adj)/2.0)*noise_percent)/100.0)



    # rem_max_edges = max_edges + 6
    # rem_max_edges = min(rem_max_edges, int(np.sum(sub_adj)/2))

    # miss_edges = np.random.randint(2, rem_max_edges)
    miss_edges = int(((np.sum(sub_adj)/2.0)*noise_percent)/100.0)


    removed_edges = 0.0
    added_edges = 0.0


    indices = np.nonzero(sub_adj)
    indices_2 = np.nonzero(new_adj)
    assert (len(indices[0]) == len(indices[1]))


    for i in range(len(indices[0])):
        assert sub_adj[indices[0][i], indices[1][i]] > 0.5
        if new_adj[indices[0][i], indices[1][i]] < 0.5:
            continue
        # if indices[0][i] in h_nodes or indices[1][i] in h_nodes:
        #     continue

        if indices[0][i] in h_nodes and indices[1][i] in h_nodes:
            continue
        if np.random.randint(0, int(np.sum(sub_adj)/2.0)) < miss_edges:

            new_adj[indices[0][i], indices[1][i]] = 0.0
            new_adj[indices[1][i], indices[0][i]] = 0.0
            removed_edges += 1.0


    for re in range(rand_edges):
        sn = np.random.randint(sub_nodes)
        nodeFound = False
        if sn in h_nodes:
            nodeFound = True
        # while sn in h_nodes:
        #     sn = np.random.randint(sub_nodes)
        iter_count = 0
        skip = False
        en =  np.random.randint(sub_nodes)
        while(en == sn) or (nodeFound == True and en in h_nodes):
        # while(en == sn) or (en in h_nodes):
            en = np.random.randint(sub_nodes)
            iter_count += 1
            if iter_count > 5:
                skip = True
                break
        if skip:
            continue
        new_adj[sn, en] = 1.0
        new_adj[en, sn] = 1.0
        added_edges += 1.0
    print("edge stats: ", miss_edges, rand_edges, removed_edges, added_edges)


    return new_adj, added_edges, removed_edges

def compare_adjs(masked_adj, masked_adjs_noise, orig_adj, noise_orig_adjs, h_nodes=None, use_comb_mask=True):
    diff = 0.
    hnodes_mask = np.zeros_like(masked_adj)
    if h_nodes is not None:
        for h in h_nodes:
            for h2 in h_nodes:
                hnodes_mask[h,h2] = 1.0

    for i,m in enumerate(masked_adjs_noise):
        orig_noise_adj = noise_orig_adjs[i]


        if use_comb_mask:
            comb_mask = orig_adj*orig_noise_adj
            diff += np.sum(np.square(masked_adj - m)*comb_mask)
        else:
            if h_nodes is not None:
                diff += np.sum(np.square(masked_adj - m)*hnodes_mask)
            else:
                diff += np.sum(np.square(masked_adj - m))

    diff = diff/len(masked_adjs_noise)
    # print("diff: ", diff)
    return diff

# use edges as gt
def addNoiseToGraphEdges(sub_adj, sub_feat, h_edges, sub_nodes, noise_percent=10.0):



    # max_edges = max(int(np.sum(sub_adj)/10.0), 4)
    # rand_edges = np.random.randint(2, int(max_edges) + 10)
    new_adj = np.zeros_like(sub_adj)
    new_adj = new_adj + sub_adj


    rand_edges = int(((np.sum(sub_adj)/2.0)*noise_percent)/100.0)



    # rem_max_edges = max_edges + 6
    # rem_max_edges = min(rem_max_edges, int(np.sum(sub_adj)/2))

    # miss_edges = np.random.randint(2, rem_max_edges)
    miss_edges = int(((np.sum(sub_adj)/2.0)*noise_percent)/100.0)


    removed_edges = 0.0
    added_edges = 0.0


    indices = np.nonzero(sub_adj)
    indices_2 = np.nonzero(new_adj)
    assert (len(indices[0]) == len(indices[1]))


    for i in range(len(indices[0])):
        assert sub_adj[indices[0][i], indices[1][i]] > 0.5
        if new_adj[indices[0][i], indices[1][i]] < 0.5:
            continue
        # if indices[0][i] in h_nodes or indices[1][i] in h_nodes:
        #     continue

        if h_edges[indices[0][i], indices[1][i]] > 0:
            continue
        if np.random.randint(0, int(np.sum(sub_adj)/2.0)) < miss_edges:

            new_adj[indices[0][i], indices[1][i]] = 0.0
            new_adj[indices[1][i], indices[0][i]] = 0.0
            removed_edges += 1.0

    
    node_set = set()
    gt = coo_matrix(h_edges)
    for (r, c) in list(zip(gt.row, gt.col)):
        node_set.add(r)
        node_set.add(c)

    for re in range(rand_edges):
        sn = np.random.randint(sub_nodes)
        nodeFound = False
        if sn in node_set:
            nodeFound = True
        # while sn in h_nodes:
        #     sn = np.random.randint(sub_nodes)
        iter_count = 0
        skip = False
        en =  np.random.randint(sub_nodes)
        while(en == sn) or (nodeFound == True and en in node_set):
        # while(en == sn) or (en in h_nodes):
            en = np.random.randint(sub_nodes)
            iter_count += 1
            if iter_count > 5:
                skip = True
                break
        if skip:
            continue
        new_adj[sn, en] = 1.0
        new_adj[en, sn] = 1.0
        added_edges += 1.0
    print("edge stats: ", miss_edges, rand_edges, removed_edges, added_edges)


    return new_adj, added_edges, removed_edges

def get_topk_mask(adj, k):
    adj = torch.tensor(adj)
    flat_adj = adj.flatten()
    topk, indices = torch.topk(flat_adj, k)
    mask = torch.zeros(flat_adj.shape)
    mask[indices] = 1
    mask = np.reshape(mask, adj.shape)    
    return mask

class AdversarialNoiseHandler(NoiseHandler):
    def __init__(self, name, model, explainer, noise_percent=0.1, mode='auc', adversarial_file=None):
        super(AdversarialNoiseHandler, self).__init__(name, model, explainer, noise_percent, mode)
        self.adversarial_adj = adversarial_file[b'adj']
        self.adversarial_feat = adversarial_file[b'feat']
        self.adversarial_pred = adversarial_file[b'pred'][0]
        self.adversarial_label = adversarial_file[b'label']
        self.adversarial_num_nodes = adversarial_file[b'num_nodes']
        self.adversarial_file = adversarial_file
        self.feat_diffs = []
        self.adj_diffs = []

    def sample(self, feat, adj, num_nodes, gt_nodes=None, gt_edges=None, graph_idx=None):
        new_feat = self.adversarial_feat[graph_idx]
        new_adj = self.adversarial_adj[graph_idx]

        self.feat_diffs.append(np.sum(np.abs(feat - new_feat)))
        self.adj_diffs.append((np.sum(np.abs(adj - new_adj))/np.sum(adj)))

        assert(num_nodes == self.adversarial_num_nodes[graph_idx])

        adj_diff = np.sum(np.abs(adj - new_adj))
        feat_diff = np.sum(np.abs(feat - new_feat))

        new_feat = torch.tensor(new_feat, dtype=torch.float)
        new_adj = torch.tensor(new_adj, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)

        pred_noise, _ = self.model(new_feat.unsqueeze(0), new_adj.unsqueeze(0), batch_num_nodes=[num_nodes.cpu().numpy()])
        pred, _ = self.model(feat.unsqueeze(0), adj.unsqueeze(0), batch_num_nodes=[num_nodes.cpu().numpy()])

        pred_noise = pred_noise.detach().cpu()
        pred = pred.detach().cpu()
        noise_label = torch.argmax(pred_noise[0]).item()
        orig_label = torch.argmax(pred[0]).item()

        self.pred_change.update(noise_label != orig_label)
        self.prob_change.update(float(torch.abs(pred_noise[0][orig_label] - pred[0][orig_label])))

        self.sample_count[noise_label] += 1
        if noise_label == orig_label:
            self.adj_diff.update(adj_diff)
            self.feat_diff.update(feat_diff)

        return new_feat, new_adj, noise_label

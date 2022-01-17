import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import tensorboardX.utils

import torch
import torch.nn as nn

import sklearn.metrics as metrics

from scipy.sparse import coo_matrix

import utils.graph_utils as graph_utils
import pickle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


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

misexplain = [0, 0]
totalexplain = [0, 0]

class Explainer:
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
        device='cpu',
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.num_nodes = num_nodes
        self.n_hops = args.num_gc_layers
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.device = device

    def extract_neighborhood_from_saved_data(self, node_idx, dataset):
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

        adj = self.nbr_data['adj'][offset_idx]
        feat = self.nbr_data['feat'][offset_idx]
        label = self.nbr_data['label'][offset_idx]
        nbrs = self.nbr_data['neighbors'][offset_idx]
        old_idx = self.nbr_data['old_idx'][offset_idx]
        new_idx = self.nbr_data['new_idx'][offset_idx]
        assert(node_idx == old_idx)
        return new_idx, adj, feat, label, nbrs

    def get_nbr_data(self, args, node_indices, graph_idx=0):
        ajs_l = []
        feats_l = []
        labels_l = []
        neighbors_l = []
        old_idx_l = []
        new_idx_l = []
        pred_l = []
        for node_idx in node_indices:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            ajs_l.append(sub_adj)
            feats_l.append(sub_feat)
            labels_l.append(sub_label)
            neighbors_l.append(neighbors)
            old_idx_l.append(node_idx)
            new_idx_l.append(node_idx_new)

            sub_label = np.expand_dims(sub_label, axis=0)

            sub_adj = np.expand_dims(sub_adj, axis=0)
            sub_feat = np.expand_dims(sub_feat, axis=0)

            adj   = torch.tensor(sub_adj, dtype=torch.float)
            x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
            label = torch.tensor(sub_label, dtype=torch.long)

            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            pred_l.append(pred_label)

        torch_data = {}
        torch_data['adj'] = ajs_l
        torch_data['feat'] = feats_l
        torch_data['label'] = labels_l
        torch_data['neighbors'] = neighbors_l
        torch_data['old_idx'] = old_idx_l
        torch_data['new_idx'] = new_idx_l
        torch_data['pred'] = pred_l

        self.nbr_data = torch_data
        return torch_data

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )


    '''
    topk parameter are weights for directed graphs. for undirected graphs, double the topk vector
    '''
    def evaluate_interpretation(model, masked_adj, graph_mode, graph_idx=None, node_idx=None, mode=None, topk=[4, 6, 8], sparsity=[0, 0.5]):
        assert(graph_idx is not None or node_idx is not None)

        batch_num_nodes = [sub_nodes.cpu().numpy()] if sub_nodes is not None else None

        logits, _ = model(x, adj, batch_num_nodes=batch_num_nodes)

        if not graph_mode:
            logits = logits[0][node_idx_new]
        else:
            logits = logits[0]

        pred_label = np.argmax(logits.cpu().detach().numpy())

        # get prediction changes from new adj
        
        def get_graph_pred_changes2(m_adj, nodes_rem, zeroing=True):
            nodes_rem = nodes_rem.tolist()
            num_nodes = len(nodes_rem)
            if zeroing:
                sub_feat = x.detach().clone()
                for n in nodes_rem:
                    sub_feat[0,n,:] = 0.
                sub_adj = adj
            else:
                G = nx.from_numpy_array(m_adj.cpu().numpy()[0,:,:])
                H = G.subgraph(nodes_rem)
            
                assert(num_nodes > 0)
                sub_adj = nx.adjacency_matrix(H).todense()
                #for e in H.edges:
                    #print("edge: ", e, m_adj[0, e[0], e[1]], sub_adj[e[0],e[1]])
                #print(m_adj.cpu().numpy()[0,:num_nodes, :num_nodes])
                #print(np.sum(sub_adj), torch.sum(m_adj).item())
                sub_adj = torch.from_numpy(sub_adj).float().unsqueeze(0)
                sub_feat = torch.zeros((1, num_nodes, x.shape[2]))
                for ix, n in enumerate(nodes_rem):
                    sub_feat[0,ix,:] = x[0,n,:]
            
            logits_masked, _ = self.model(sub_feat, sub_adj, batch_num_nodes=[num_nodes])
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

        def get_graph_pred_changes(m_adj, m_x):
            if graph_mode:
                logits_masked, _ = self.model(m_x, m_adj, batch_num_nodes=batch_num_nodes)
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
        '''
        soft-mask: inverse masked adj
        hard-mask: mask out topk in masked adj
        keep-mask: mask in topk in masked adj
        '''
        if mode == 'soft-mask':
            nz = masked_adj > 0
            nz = torch.tensor(nz, dtype=torch.float32)
            inv_masked_adj = nz - masked_adj
            inv_masked_adj = torch.tensor(inv_masked_adj, dtype=torch.float)
            inv_masked_adj = inv_masked_adj.unsqueeze(0)
            pred_change, pred_prob_change = get_graph_pred_changes(inv_masked_adj, x)
            return pred_change, pred_prob_change.cpu().detach().numpy()
        elif mode == 'hard-mask':
            pred_changes = []
            pred_prob_changes = []
            masked_adj = torch.tensor(masked_adj, dtype=torch.float)
            for k in topk:
                flat_masked_adj = masked_adj.flatten()
                topk, indices = torch.topk(flat_masked_adj, k)
                mask = torch.ones(flat_masked_adj.shape)
                mask[indices] = 0
                flat_masked_adj = adj.flatten() * mask
                hard_masked_adj = np.reshape(flat_masked_adj, masked_adj.shape)
                hard_masked_adj = torch.unsqueeze(hard_masked_adj, 0)

                pred_change, pred_prob_change = get_graph_pred_changes(hard_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())
            return pred_changes, pred_prob_changes
        elif mode == 'keep-mask':
            pred_changes = []
            pred_prob_changes = []
            masked_adj = torch.tensor(masked_adj, dtype=torch.float)

            for k in topk:
                flat_masked_adj = masked_adj.flatten()
                topk, indices = torch.topk(flat_masked_adj, k)
                mask = torch.zeros(flat_masked_adj.shape)
                mask[indices] = 1
                flat_masked_adj = adj.flatten() * mask
                hard_masked_adj = np.reshape(flat_masked_adj, masked_adj.shape)
                hard_masked_adj = torch.unsqueeze(hard_masked_adj, 0)

                pred_change, pred_prob_change = get_graph_pred_changes(hard_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())
            return pred_changes, pred_prob_changes

        elif mode == 'edge-fidelity-k':
            '''            Note: we evaluate edge fidelity            '''            
            nodes = adj.shape[1]            
            argsort_adj = np.dstack(np.unravel_index(np.argsort(masked_adj.ravel()), (nodes, nodes)))[0]            
            pred_changes = []            
            pred_prob_changes = []            
            sparsities = []            
            #flat_masked_adj = masked_adj.flatten()            
            nnz = np.count_nonzero(adj.cpu().numpy())            
            #flat_masked_adj = np.sort(flat_masked_adj)[::-1]            
            for sp in sparsity:                
                k_edges = round(0.5*nnz*(1-sp))                
                #threshold = flat_masked_adj[int(nnz * (1-sp))]                
                fid_masked_adj = np.ones((1,nodes,nodes))                
                count_k = 0               
                #fid_masked_adj = masked_adj > threshold                
                # #mask = torch.ones(masked_adj.shape)                
                # #mask -= fid_masked_adj                
                # #fid_masked_adj = torch.mul(adj, mask)                
                edges_covered = {}                 
                for i in range(nodes * nodes - 1, -1, -1):                    
                    if count_k == k_edges:                        
                        break                    
                    x = argsort_adj[i][0]                    
                    y = argsort_adj[i][1]                    
                    if (x,y) in edges_covered or (y,x) in edges_covered:                        
                        continue                    
                    fid_masked_adj[0,x,y] = 0.0                    
                    fid_masked_adj[0,y,x] = 0.0                    
                    edges_covered[(x,y)] = 1                    
                    count_k += 1                
                fid_masked_adj = torch.from_numpy(fid_masked_adj).float()               
                fid_masked_adj = torch.mul(adj, fid_masked_adj)                
                sparsities.append(torch.sum(fid_masked_adj) / torch.sum(adj))  
                print(x)

                x = torch.tensor(x)
                print(fid_masked_adj.shape)
                print(x.shape)
                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)                
                pred_changes.append(pred_change)                
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())            
                    
            return pred_changes, pred_prob_changes, sparsities
        elif mode == 'edge-fidelity':
            '''
            Note: we evaluate edge fidelity
            '''
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            flat_masked_adj = masked_adj.flatten()
            nnz = np.count_nonzero(adj)
            flat_masked_adj = np.sort(flat_masked_adj)[::-1]

            for sp in sparsity:
                threshold = flat_masked_adj[int(nnz * (1 - sp))]

                fid_masked_adj = masked_adj > threshold
                # fid_masked_adj = torch.full(masked_adj.shape, float(sp))
                # fid_masked_adj = torch.bernoulli(fid_masked_adj)

                mask = torch.ones(masked_adj.shape)
                mask -= fid_masked_adj

                fid_masked_adj = torch.mul(adj, mask)
                sparsities.append(torch.sum(fid_masked_adj) / torch.sum(adj))
                # fid_masked_adj = torch.zeros(fid_masked_adj.shape)
                # fid_x = torch.zeros(x.shape)
                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

                # if pred_changes[0] == False:
                #     log_graph(x, fid_masked_adj, "edge-fid-rand-" + str(sp) + "-" + str(pred_label) + "-" + str(pred_change), str(pred_prob_change))
                # if pred_changes[0] == False:
                #     misexplain[pred_label] += 1
                # totalexplain[pred_label] += 1
                # print(misexplain, totalexplain)
            # if pred_changes[0] == False:
            #     exit()
            return pred_changes, pred_prob_changes, sparsities
        elif mode == 'node-fidelity-avg':
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            flat_masked_adj = masked_adj.flatten()
            nnz = np.count_nonzero(flat_masked_adj)
            flat_masked_adj = np.sort(flat_masked_adj)[::-1]

            for sp in sparsity:
                node_edge_weights = np.sum(masked_adj, axis=1)
                num_edges = np.sum(adj.clone().detach().numpy(), axis=1)
                num_nodes = np.sum(num_edges > 0)
                threshold = int(np.floor(num_nodes * (1-sp)))

                mask_threshold = sorted(node_edge_weights, reverse=True)[num_nodes - threshold]

                mask = torch.ones(masked_adj.shape)
                idx = node_edge_weights >= mask_threshold
                sparsities.append(np.sum(idx) / num_nodes)

                mask[idx, :] = 0
                mask[:, idx] = 0
                fid_masked_adj = torch.multiply(adj, mask)

                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

            return pred_changes, pred_prob_changes, sparsities

        elif mode == 'node-fidelity-max':
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            flat_masked_adj = masked_adj.flatten()
            nnz = np.count_nonzero(flat_masked_adj)
            flat_masked_adj = np.sort(flat_masked_adj)[::-1]

            for sp in sparsity:
                node_edge_weights = np.max(masked_adj, axis=1)
                num_edges = np.sum(adj.clone().detach().numpy(), axis=1)
                num_nodes = np.sum(num_edges > 0)
                threshold = min(int(np.floor(num_nodes * (1 - sp))), num_nodes-1)

                mask_threshold = sorted(node_edge_weights, reverse=True)[threshold]

                mask = torch.ones(masked_adj.shape)
                idx = node_edge_weights > mask_threshold
                sparsities.append(1 - np.sum(idx) / num_nodes)

                mask[idx, :] = 0
                mask[:, idx] = 0
                fid_masked_adj = torch.multiply(adj, mask)

                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

            return pred_changes, pred_prob_changes, sparsities
        elif mode == 'node-fidelity':
            '''
            Node fidelity based on connectivity
            '''
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            coo_adj = coo_matrix(masked_adj)
            tuples = zip(coo_adj.row, coo_adj.col, coo_adj.data)
            sorted_coo_adj = sorted(tuples, key=lambda x: x[2], reverse=True)

            for sp in sparsity:
                node_edge_weights = np.sum(masked_adj, axis=1)
                num_edges = np.sum(adj.clone().detach().numpy(), axis=1)
                num_nodes = np.sum(num_edges > 0)
                threshold = min(int(num_nodes * (1 - sp)), num_nodes - 1)

                def get_node_mask(threshold, gentle_mask=False):
                    masked_nodes = []
                    i = 0
                    
                    edges = list(zip(coo_adj.row, coo_adj.col))
                    refcount = {}
                    for u, v in edges:
                        if u not in refcount.keys():
                            refcount[u] = 0
                        if v not in refcount.keys():
                            refcount[v] = 0
                        refcount[u] += 1
                        refcount[v] += 1

                    while len(masked_nodes) < threshold:
                        u = sorted_coo_adj[i][0]
                        v = sorted_coo_adj[i][1]
                        i += 1
                        
                        if (u, v) in edges:
                            edges.remove((u, v))
                            refcount[u] -= 1
                            refcount[v] -= 1
                            if refcount[u] == 0:
                                masked_nodes.append(u)
                            if refcount[v] == 0:
                                masked_nodes.append(v)
                        
                    return masked_nodes

                masked_nodes = get_node_mask(threshold)
                
                mask = torch.ones(masked_adj.shape)
                for node in masked_nodes:
                    mask[node, :] = 0
                    mask[:, node] = 0

                sparsities.append(1 - len(masked_nodes) / num_nodes)

                fid_masked_adj = torch.mul(adj, mask)

                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

            return pred_changes, pred_prob_changes, sparsities

        elif mode == 'node-fidelity-min':
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            flat_masked_adj = masked_adj.flatten()
            nnz = np.count_nonzero(flat_masked_adj)
            flat_masked_adj = np.sort(flat_masked_adj)[::-1]

            for sp in sparsity:
                masked_adj_nz = np.ma.masked_equal(masked_adj, 0, copy=True)
                node_edge_weights = np.min(masked_adj_nz, axis=1)
                num_edges = np.sum(adj.clone().detach().numpy(), axis=1)
                num_nodes = np.sum(num_edges > 0)
                threshold = min(int(np.floor(num_nodes * (1 - sp))), num_nodes-1)

                mask_threshold = sorted(node_edge_weights)[threshold]

                mask = torch.ones(masked_adj.shape)
                idx = node_edge_weights < mask_threshold

                sparsities.append(1 - np.sum(idx) / num_nodes)

                mask[idx, :] = 0
                mask[:, idx] = 0
                fid_masked_adj = torch.multiply(adj, mask)

                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())

            return pred_changes, pred_prob_changes, sparsities
        elif mode == 'node-fidelity-min-feat-mask':
            pred_changes = []
            pred_prob_changes = []
            sparsities = []

            flat_masked_adj = masked_adj.flatten()
            nnz = np.count_nonzero(flat_masked_adj)
            flat_masked_adj = np.sort(flat_masked_adj)[::-1]

            for sp in sparsity:
                # masked_adj_nz = np.ma.masked_equal(masked_adj, 0, copy=True)
                node_edge_weights = np.max(masked_adj, axis=1)
                num_edges = np.sum(adj.clone().detach().numpy(), axis=1)
                num_nodes = np.sum(num_edges > 0)
                threshold = min(int(np.floor(num_nodes * (1 - sp))), num_nodes-1)

                mask_threshold = sorted(node_edge_weights, reverse=True)[threshold]

                mask = torch.ones(masked_adj.shape)
                idx = node_edge_weights > mask_threshold
                sparsities.append(1 - np.sum(idx) / num_nodes)

                node_mask = ~idx

                fid_x = x #* torch.tensor(node_mask).unsqueeze(1)
                # print(torch.nonzero(torch.tensor(node_mask)))
                # print(fid_x)
                mask[idx, :] = 0
                mask[:, idx] = 0
                fid_masked_adj = torch.multiply(adj, mask)
                # print(coo_matrix(fid_masked_adj[0]))
                # print("Get embedding for " + str(1 - np.sum(idx) / num_nodes))
                # print(idx)
                pred_change, pred_prob_change = get_graph_pred_changes(fid_masked_adj, fid_x)
                pred_changes.append(pred_change)
                pred_prob_changes.append(pred_prob_change.cpu().detach().numpy())
            return pred_changes, pred_prob_changes, sparsities
        else:
            raise NotImplementedError

class AverageMeter(object):
    def __init__(self, size=1):
        self.size = size
        self.reset()
        
    def reset(self):
        self.val = np.zeros(self.size)
        self.avg = np.zeros(self.size)
        self.sum = np.zeros(self.size)
        self.count = np.zeros(self.size)

    def update(self, val, n=1):
        self.val = np.array(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if np.sum(self.count) != 0 else 0

    def selective_update(self, val, index):
            
        self.val[index] = val
        self.sum[index] += val
        self.count[index] += 1
        self.avg[index] = self.sum[index] / self.count[index] if self.count[index] != 0 else 0

def log_graph(feat, adj, name, title, feat_lut=None):

    feat = feat[0].detach().cpu().numpy()
    adj = adj[0].detach().cpu().numpy()
    rel = np.sum(np.sum(feat, axis=1) > 0)
    plt.title(title)
    G = nx.from_numpy_matrix(adj[:rel, :rel])
    node_color = np.sum(feat)
    nx.draw(
        G,
        pos=nx.spring_layout(G),
    )
    plt.savefig(name+".png")
    plt.clf()
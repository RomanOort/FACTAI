import torch
import torch.nn as nn
import numpy as np

import configs
import models
from utils import train_utils
from utils import accuracy_utils

def get_mutagenicity_args():
    args = configs.arg_parse()
    args.graph_mode = True
    args.num_gc_layers = 3
    args.explainer_method = 'rcexplainer'
    args.gpu = True
    args.lr = 0.001
    args.size_c = 0.001
    args.ent_c = 8.0
    args.boundary_c = 3.0
    args.inverse_boundary_c = 12.0
    args.bloss_version = 'sigmoid'
    args.num_epochs = 1
    return args

def get_mutagenicity_model(input_dim, num_classes, device, args):
    return models.GcnEncoderGraph(
        input_dim=input_dim,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        pred_hidden_dims=[],
        bn=False,
        args=args,
        device=device
    )

def get_dataset_from_ckpt(ckpt):
    cg_dict = ckpt["cg"]
    adj = cg_dict["adj"]
    feat = cg_dict["feat"]
    label = cg_dict["label"]
    pred = cg_dict["pred"]
    num_nodes = cg_dict["num_nodes"]

    return adj, feat, label, pred, num_nodes

def train_explainer(explainer, model, rule_dict, adjs, feats, labels, preds, num_nodes, args, graph_indices):
    params_optim = []
    for name, param in explainer.named_parameters():
        if "model" in name:
            continue
        params_optim.append(param)

    scheduler, optimizer = train_utils.build_optimizer(args, params_optim)

    ep_count = 0
    loss_ep = 0 

    for epoch in range(args.start_epoch, args.num_epochs):
        loss_epoch = 0

        stats = accuracy_utils.Stats("RCExplainer", explainer, model)

        np.random.shuffle(graph_indices)
        explainer.train()
        for graph_idx in graph_indices:
            # preprocess inputs
            sub_adj = adjs[graph_idx]
            sub_nodes = num_nodes[graph_idx]
            sub_feat = feats[graph_idx]
            sub_label = labels[graph_idx]

            sub_adj = np.expand_dims(sub_adj, axis=0)
            sub_feat = np.expand_dims(sub_feat, axis=0)

            adj   = torch.tensor(sub_adj, dtype=torch.float).cuda()
            x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).cuda()
            label = torch.tensor(sub_label, dtype=torch.long).cuda()

            pred_label = np.argmax(preds[0][graph_idx], axis=0)
            # extract model embeddings from layer
            emb = model.getEmbeddings(x, adj, batch_num_nodes=[sub_nodes.cpu().numpy()])
            emb = emb.clone().detach()

            gt_pred, gt_embedding = model(x.cuda(), adj.cuda(), batch_num_nodes=[sub_nodes.cpu().numpy()])

            # get boundaries for sample
            rule_ix = rule_dict['idx2rule'][graph_idx]
            rule = rule_dict['rules'][rule_ix]
            rule_label = rule['label']

            boundary_list = []
            for b_num in range(len(rule['boundary'])):

                boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
                if args.gpu:
                    boundary = boundary.cuda()
                boundary_label = rule['boundary'][b_num]['label']
                boundary_list.append(boundary)

            # explain prediction
            t0 = 0.5
            t1 = 4.99

            tmp = float(t0 * np.power(t1 / t0, epoch /args.num_epochs))
            pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes), training=True)
            loss, bloss_s = explainer.loss(pred, pred_label, graph_embedding=graph_embedding,
                               boundary_list=boundary_list, gt_embedding=gt_embedding, inv_embedding=inv_embedding)

            if ep_count < 200:
                loss_ep += loss
                ep_count += 1.0
            else:
                ep_count = 0.
                optimizer.zero_grad()
                loss_ep.backward()
                optimizer.step()
                loss_epoch += loss_ep.detach()
                loss_ep = 0.

            # evaluate explanation
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
            imp_nodes = None
            stats.update(masked_adj, imp_nodes, adj, x, label, sub_nodes)

        if scheduler is not None:
            scheduler.step()

        print("Epoch: {} \t Loss: {}".format(epoch, loss_epoch))
    torch.save(explainer.state_dict(), './rcexplainer_mutagenicity.pth')
    print(stats.summary())

    return explainer


def evaluate_explainer(explainer, model, rule_dict, adjs, feats, labels, preds, num_nodes, args, graph_indices):
    masked_adjs = []
    ep_count = 0
    loss_ep = 0 

    loss_epoch = 0

    stats = accuracy_utils.Stats("RCExplainer", explainer, model)

    explainer.train()
    for graph_idx in graph_indices:
        # preprocess inputs
        sub_adj = adjs[graph_idx]
        sub_nodes = num_nodes[graph_idx]
        sub_feat = feats[graph_idx]
        sub_label = labels[graph_idx]

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float).cuda()
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).cuda()
        label = torch.tensor(sub_label, dtype=torch.long).cuda()

        pred_label = np.argmax(preds[0][graph_idx], axis=0)
        # extract model embeddings from layer
        emb = model.getEmbeddings(x, adj, batch_num_nodes=[sub_nodes.cpu().numpy()])
        emb = emb.clone().detach()

        gt_pred, gt_embedding = model(x.cuda(), adj.cuda(), batch_num_nodes=[sub_nodes.cpu().numpy()])

        # get boundaries for sample
        rule_ix = rule_dict['idx2rule'][graph_idx]
        rule = rule_dict['rules'][rule_ix]
        rule_label = rule['label']

        boundary_list = []
        for b_num in range(len(rule['boundary'])):

            boundary = torch.from_numpy(rule['boundary'][b_num]['basis'])
            if args.gpu:
                boundary = boundary.cuda()
            boundary_label = rule['boundary'][b_num]['label']
            boundary_list.append(boundary)

        # explain prediction
        t0 = 0.5
        t1 = 4.99

        tmp = float(t0 * np.power(t1 / t0, 1.0))
        pred, masked_adj, graph_embedding, inv_embedding, inv_pred = explainer((x[0], emb[0], adj[0], tmp, label, sub_nodes), training=False)
        loss, bloss_s = explainer.loss(pred, pred_label, graph_embedding=graph_embedding,
                            boundary_list=boundary_list, gt_embedding=gt_embedding, inv_embedding=inv_embedding)

        if ep_count < 200:
            loss_ep += loss
            ep_count += 1.0
        else:
            ep_count = 0.
            loss_epoch += loss_ep.detach()
            loss_ep = 0.

        # evaluate explanation
        masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        masked_adjs.append(masked_adj)
        imp_nodes = None


        stats.update(masked_adj, imp_nodes, adj, x, label, sub_nodes)

    print("Loss: {}".format(loss_epoch))
    # torch.save(explainer.state_dict(), './rcexplainer_mutagenicity.pth')
    print(stats.summary())
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
                    boundary_loss += torch.nn.functional.sigmoid(-1.0 * sigma * (gt_proj * ft_proj))
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
                    inv_loss = torch.nn.functional.sigmoid(sigma * (gt_proj * inv_proj))
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


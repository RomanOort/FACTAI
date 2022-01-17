import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from dnn_invariant.algorithms.mine_gcn_invariant import Struct_BB

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        add_self=False,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        gpu=True,
        att=False,
    ):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim)
                )
            if att:
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim).cuda()
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim).cuda()
                )
        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj, adj_v = None):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        # deg = torch.sum(adj, -1, keepdim=True)
        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            # import pdb
            # pdb.set_trace()
            att = x_att @ x_att.permute(0, 2, 1)
            # att = self.softmax(att)
            adj = adj * att

        if adj_v is not None:
            x_exp = x.unsqueeze(2).expand(x.shape[0], x.shape[1], x.shape[1],x.shape[2]) #bs*n*n*d
            x_exp = x_exp * adj_v
            x_exp = x_exp * adj.unsqueeze(3).expand(adj.shape[0], adj.shape[1], adj.shape[2], x_exp.shape[3])
            y = torch.sum(x_exp,dim=2)
        else:
            y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y, adj



class GcnEncoderGraph(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=False,
        bn=True,
        dropout=0.0,
        add_self=True,
        args=None,
    ):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1




        self.bias = True
        self.gpu = True
        self.att = False
        # if args.method == "att":
        #     self.att = True
        # else:
        #     self.att = False
        # if args is not None:
        #     self.bias = args.bias

        self.add_embedding = False

        # self.adj_linear1 = nn.Linear(input_dim, input_dim)
        # self.adj_linear2 = nn.Linear(input_dim, input_dim)
        #
        # init.xavier_uniform_(self.adj_linear1.weight)
        # init.constant_(self.adj_linear1.bias, 0.0)
        # init.xavier_uniform_(self.adj_linear2.weight)
        # init.constant_(self.adj_linear2.bias, 0.0)
        # self.sigmoid = nn.Sigmoid()
        #
        # if args.add_embedding:
        #     self.add_embedding = args.add_embedding
        #     self.embed = nn.Parameter(torch.rand(1, 32))

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.att:
                    init.xavier_uniform_(
                        m.att_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.add_self:
                    init.xavier_uniform_(
                        m.self_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers,
        add_self,
        normalize=False,
        dropout=0.0,
    ):
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    add_self=add_self,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                    att=self.att,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        return conv_first, conv_block, conv_last

    def build_pred_layers(
        self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(
        self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):

        """ Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """

        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)

        x_tensor = x  # added by mohit and commented below lines

        # x_all.append(x)
        # adj_att_all.append(adj_att)
        # # x_tensor: [batch_size x num_nodes x embedding]
        # x_tensor = torch.cat(x_all, dim=2)

        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def _getBBOfOneLayer(self, input, layer_ptr, layer_start=-1):
        # prepare the buffer
        basis_list = []
        bias_list = []

        # set true grad flag for input
        input.requires_grad_(True)

        out_of_layer = self._getOutputOfOneLayer(input)

        out_of_layer = out_of_layer.reshape(out_of_layer.size(0), -1)

        self.zero_grad()
        self.eval()
        for idx in range(out_of_layer.size(1)):
            unit_mask = torch.zeros(out_of_layer.size())
            unit_mask[:, idx] = 1
            unit_mask = unit_mask.cuda()

            # compute basis of this unit
            out_of_layer.backward(unit_mask, retain_graph=True)
            basis = input.grad.clone().detach().reshape(input.size(0), -1)
            basis_list.append(basis)

            # do substraction to get bias
            basis_mul_x = torch.mul(input.clone().detach(), input.grad.clone().detach())
            # print("basis shape: ", basis_mul_x.shape)
            basis_mul_x = torch.sum(basis_mul_x, dim=1).cuda()
            bias = out_of_layer[:, idx].clone().detach() - basis_mul_x
            bias_list.append(bias)

            # clean up
            self.zero_grad()
            input.grad.data.zero_()

        # set false grad flag for input
        input.requires_grad_(False)

        # reshape basis to tensor shape
        stacked_basis = torch.stack(basis_list, dim=2)
        array_basis = stacked_basis.detach().squeeze().cpu().numpy()

        # reshape bias to tensor shape
        stacked_bias = torch.stack(bias_list, dim=1)
        array_bias = stacked_bias.detach().squeeze().cpu().numpy()

        return Struct_BB(array_basis, array_bias)

    def  _getBBOfLastLayer(self, input, layer_start=-1):
        # get the bb of the logits
        # layer_ptr = self._layers_list.__len__() - 1
        bb_of_logits = self._getBBOfOneLayer(input, -1, layer_start)

        # identify the idx of the pivot logit
        logits = bb_of_logits.computeHashVal(input.reshape(input.size(0), -1).cpu().numpy())
        assert(logits.shape[0] == 1)
        logits = logits.squeeze()

        logits_order = np.argsort(logits)
        pivot_id1 = logits_order[-1]
        pivot_id2 = logits_order[-2]

        #pivot_id = np.argmax(logits)

        # subtract between the logits to get BB_of_last_layer
        bb_of_logits.subPivotOverOthers(pivot_id1, pivot_id2)

        return bb_of_logits

    def _getOutputOfOneLayer(self, boundary_pt):
        ypred = self.pred_model(boundary_pt)
        return ypred


    def _getOutputOfOneLayer_Group(self, adj, instances, batch_num_nodes=None, layer_start=-1, xyz=-1):
        feats_exp = self.forward(instances, adj, batch_num_nodes=batch_num_nodes, extract_feats=True)
        feats_exp = feats_exp*self.embedding_mask
        feats = torch.sum(feats_exp,dim=1)/torch.sum(self.embedding_mask,1)
        # feats, _ = torch.max(feats_exp, dim=1)

        return feats, feats_exp

    def forward(self, x, adj, batch_num_nodes=None, adj_v=None, extract_feats=False, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]






        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        if adj_v is not None:
            adj_v = self.act(self.adj_linear1(adj_v))
            x = self.act(self.adj_linear2(x))
            adj_v = torch.transpose(adj_v,1,2)

            # bs*n*n*d, bs*n*n*d , bs*n*n*d

            # bs*n*n*d
            t_mask = torch.transpose(self.embedding_mask, 1,2)
            adj_mask = torch.bmm(self.embedding_mask, t_mask)
            diag_mask = (1. - torch.eye(adj_mask.shape[1]).float().cuda()).unsqueeze(0)
            diag_mask = diag_mask.expand(adj_mask.shape[0], diag_mask.shape[1], diag_mask.shape[2])
            adj_mask = adj_mask*diag_mask

            adj_v = adj_v*adj_mask.unsqueeze(3)

        # #embed  added by mbajaj
        # if self.add_embedding:
        #     x = x.view(-1,x.shape[2])
        #
        #     x = torch.matmul(x, self.embed)
        #     x = x.view(-1,max_num_nodes,32)

        # conv

        x, adj_att = self.conv_first(x, adj, adj_v)


        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]

        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)

        if extract_feats:
            return x



        # adj_att_all.append(adj_att)
        # # x = self.act(x)
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        # if self.num_aggs == 2:
        #     out = torch.sum(x, dim=1)
        #     out_all.append(out)
        # if self.concat:
        #     output = torch.cat(out_all, dim=1)
        # else:
        #     output = out
        #
        # # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        # adj_att_tensor = torch.stack(adj_att_all, dim=3)
        #
        # self.embedding_tensor = output
        adj_att_tensor = None
        output = x * self.embedding_mask
        output = torch.sum(output, dim=1) / torch.sum(self.embedding_mask, 1)
        # output, _ = torch.max(output, dim=1)

        ypred = self.pred_model(output)
        # print(output.size())
        return ypred, adj_att_tensor

    def loss(self, pred, label, type="softmax"):
        # softmax + CE
        if type == "softmax":
            return F.cross_entropy(pred, label, size_average=True)
        elif type == "margin":
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnEncoderNode(GcnEncoderGraph):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=True,
        bn=True,
        dropout=0.0,
        args=None,
    ):
        super(GcnEncoderNode, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims,
            concat,
            bn,
            dropout,
            args=args,
        )
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = nn.CrossEntropyLoss()

    def forward(self, x, adj, batch_num_nodes=None, new_node_idx = None, extract_feats=False, **kwargs):

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        self.embedding_mask = embedding_mask

        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )
        if extract_feats:
            return self.embedding_tensor
        if new_node_idx is not None:

            np_mask = np.zeros((self.embedding_tensor.shape[0], self.embedding_tensor.shape[1], self.embedding_tensor.shape[2]))

            for bix in range(self.embedding_tensor.shape[0]):
                np_mask[bix, new_node_idx[bix], :] = 1.0

            mask_t = torch.from_numpy(np_mask).float().cuda()
            self.embedding_tensor = self.embedding_tensor * mask_t
            self.embedding_tensor = torch.sum(self.embedding_tensor, dim=1)

        pred = self.pred_model(self.embedding_tensor)

        return pred, adj_att

    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)

    def _getBBOfOneLayer(self, input, layer_ptr, layer_start=-1):
        # prepare the buffer
        basis_list = []
        bias_list = []

        # set true grad flag for input
        input.requires_grad_(True)

        out_of_layer = self._getOutputOfOneLayer(input)

        out_of_layer = out_of_layer.reshape(out_of_layer.size(0), -1)

        self.zero_grad()
        self.eval()
        for idx in range(out_of_layer.size(1)):
            unit_mask = torch.zeros(out_of_layer.size())
            unit_mask[:, idx] = 1
            unit_mask = unit_mask.cuda()

            # compute basis of this unit
            out_of_layer.backward(unit_mask, retain_graph=True)
            basis = input.grad.clone().detach().reshape(input.size(0), -1)
            basis_list.append(basis)

            # do substraction to get bias
            basis_mul_x = torch.mul(input.clone().detach(), input.grad.clone().detach())
            # print("basis shape: ", basis_mul_x.shape)
            basis_mul_x = torch.sum(basis_mul_x, dim=1).cuda()
            bias = out_of_layer[:, idx].clone().detach() - basis_mul_x
            bias_list.append(bias)

            # clean up
            self.zero_grad()
            input.grad.data.zero_()

        # set false grad flag for input
        input.requires_grad_(False)

        # reshape basis to tensor shape
        stacked_basis = torch.stack(basis_list, dim=2)
        array_basis = stacked_basis.detach().squeeze().cpu().numpy()

        # reshape bias to tensor shape
        stacked_bias = torch.stack(bias_list, dim=1)
        array_bias = stacked_bias.detach().squeeze().cpu().numpy()

        return Struct_BB(array_basis, array_bias)

    def  _getBBOfLastLayer(self, input, layer_start=-1):
        # get the bb of the logits
        # layer_ptr = self._layers_list.__len__() - 1
        bb_of_logits = self._getBBOfOneLayer(input, -1, layer_start)

        # identify the idx of the pivot logit
        logits = bb_of_logits.computeHashVal(input.reshape(input.size(0), -1).cpu().numpy())
        assert(logits.shape[0] == 1)
        logits = logits.squeeze()

        logits_order = np.argsort(logits)
        pivot_id1 = logits_order[-1]
        pivot_id2 = logits_order[-2]

        #pivot_id = np.argmax(logits)

        # subtract between the logits to get BB_of_last_layer
        bb_of_logits.subPivotOverOthers(pivot_id1, pivot_id2)

        return bb_of_logits

    def _getOutputOfOneLayer(self, boundary_pt):
        ypred = self.pred_model(boundary_pt)
        return ypred


    def _getOutputOfOneLayer_Group(self, adj, instances, batch_num_nodes=None, new_node_idx = None, layer_start=-1, xyz=-1):
        feats_exp = self.forward(instances, adj, batch_num_nodes=batch_num_nodes, extract_feats=True)
        feats_exp = feats_exp * self.embedding_mask
        np_mask = np.zeros((feats_exp.shape[0], feats_exp.shape[1], feats_exp.shape[2]))

        for bix in range(feats_exp.shape[0]):
            np_mask[bix,new_node_idx[bix],:] = 1.0

        mask_t = torch.from_numpy(np_mask).float().cuda()
        feats = feats_exp*mask_t
        feats = torch.sum(feats,dim=1)

        # feats, _ = torch.max(feats_exp, dim=1)

        return feats, feats_exp


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(
        self,
        max_num_nodes,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        assign_hidden_dim,
        assign_ratio=0.25,
        assign_num_layers=-1,
        num_pooling=1,
        pred_hidden_dims=[50],
        concat=True,
        bn=True,
        dropout=0.0,
        linkpred=True,
        assign_input_dim=-1,
        args=None,
    ):
        """
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        """

        super(SoftPoolingGcnEncoder, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=pred_hidden_dims,
            concat=concat,
            args=args,
        )
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                self.pred_input_dim,
                hidden_dim,
                embedding_dim,
                num_layers,
                add_self,
                normalize=True,
                dropout=dropout,
            )
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                assign_input_dim,
                assign_hidden_dim,
                assign_dim,
                assign_num_layers,
                add_self,
                normalize=True,
            )
            assign_pred_input_dim = (
                assign_hidden_dim * (num_layers - 1) + assign_dim
                if concat
                else assign_dim
            )
            self.assign_pred = self.build_pred_layers(
                assign_pred_input_dim, [], assign_dim, num_aggs=1
            )

            # next pooling layer
            assign_input_dim = embedding_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)
            self.assign_pred_modules.append(self.assign_pred)

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim * (num_pooling + 1),
            pred_hidden_dims,
            label_dim,
            num_aggs=self.num_aggs,
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if "assign_x" in kwargs:
            x_a = kwargs["assign_x"]
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(
                x_a,
                adj,
                self.assign_conv_first_modules[i],
                self.assign_conv_block_modules[i],
                self.assign_conv_last_modules[i],
                embedding_mask,
            )
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(
                self.assign_pred(self.assign_tensor)
            )
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(
                torch.transpose(self.assign_tensor, 1, 2), embedding_tensor
            )
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(
                x,
                adj,
                self.conv_first_after_pool[i],
                self.conv_block_after_pool[i],
                self.conv_last_after_pool[i],
            )

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        """ 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        """
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(
                1 - pred_adj + eps
            )
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print("Warning: calculating link pred loss without masking")
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1 - adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss


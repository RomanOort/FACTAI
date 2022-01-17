""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import copy

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models
import pandas as pd
import datasets


from torch_geometric.utils import to_dense_adj

only_save_cg = False





'''
Use with deterministic dataloader generator, or results won't be replicable
'''
def extract_cg_pyg(args, model, device, train_loader, val_loader):

    print("Loading ckpt .....")
    ckpt_dict = io_utils.load_ckpt(args)
    model.load_state_dict(ckpt_dict['model_state'])
    model.eval()
    torch.no_grad()

    all_preds = {}

    # dummy optimizer for ckpt
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    train_acc, val_acc = 0, 0
    all_feats = {}
    all_labels = {}
    all_adjs = {}
    all_num_nodes = {}
    w_lbl = 0
    for batch_idx, data in enumerate(train_loader):

        feats = data.x
        if (data.edge_index.shape[1] == 0):
            continue
        adj = to_dense_adj(data.edge_index)
        num_nodes = min(len(data.x), adj.shape[1])
        label = data.y
        if args.bmname == 'tox21':
            label = label[:,0]


        adj = Variable(adj, requires_grad=False).float().to(device)
        h0 = Variable(feats, requires_grad=False).float().to(device)
        label = Variable(label.long(), requires_grad=False).to(device)
        batch_num_nodes = [num_nodes]
        label = label.reshape((-1))
     
        if len(h0.shape) == 2:
            h0 = h0.unsqueeze(0) 
        if h0.shape[1] > num_nodes:
            h0 = h0[:,:num_nodes,:]

        try:
            if label.item() !=0 and label.item() != 1:
                print(label)
                w_lbl += 1
                #continue
        except:
            x = 0
        all_feats[batch_idx] = h0.cpu().numpy()
        all_labels[batch_idx] = label.cpu().numpy()
        all_adjs[batch_idx] = adj.cpu().numpy()
        all_num_nodes[batch_idx] = torch.tensor(num_nodes)
        # if batch_idx == 0:
        #     all_feats = feats
        #     all_labels = label
        #     all_adjs = adj
        #     all_num_nodes = torch.tensor(num_nodes)
        # else:  
            # all_feats = torch.cat((all_feats, feats), dim=0)
            # all_labels = torch.cat((all_labels, label), dim=0)
            # all_adjs = torch.cat((all_adjs, adj), dim=0)
            # all_num_nodes = torch.cat((all_num_nodes, num_nodes), dim=0)
 
        ypred, att_adj = model(h0, adj, batch_num_nodes)
        all_preds[batch_idx] = ypred.cpu().detach().numpy()

        train_acc += torch.argmax(ypred) == label

    print("wrong labels: ", w_lbl)
    val_preds = {}
    val_feats = {}
    val_labels = {}
    val_adjs = {}
    val_num_nodes = {}
    for batch_idx, data in enumerate(val_loader):


        feats = data.x
        if (data.edge_index.shape[1] == 0):
            continue
        adj = to_dense_adj(data.edge_index)
        num_nodes = min(len(data.x), adj.shape[1])
        label = data.y
        if args.bmname == 'tox21':
            label = label[:,0]
            
        adj = Variable(adj, requires_grad=False).float().to(device)
        h0 = Variable(feats, requires_grad=False).float().to(device)
        label = Variable(label.long(), requires_grad=False).to(device)
        batch_num_nodes = [num_nodes]
        label = label.reshape((-1))
     
        if len(h0.shape) == 2:
            h0 = h0.unsqueeze(0) 
        if h0.shape[1] > num_nodes:
            h0 = h0[:,:num_nodes,:]

        val_feats[batch_idx] = h0.cpu().numpy()
        val_labels[batch_idx] = label.cpu().numpy()
        val_adjs[batch_idx] = adj.cpu().numpy()
        val_num_nodes[batch_idx] = torch.tensor(num_nodes)

        # if batch_idx == 0:
        #     val_feats = feats
        #     val_labels = label
        #     val_adjs = adj
        #     val_num_nodes = torch.tensor(num_nodes)

        # else:  # only storing 1st 20 batches, (20*20 graphs)why?
        #     val_feats = torch.cat((val_feats, feats), dim=0)
        #     val_labels = torch.cat((val_labels, label), dim=0)
        #     val_adjs = torch.cat((val_adjs, adj), dim=0)
        #     val_num_nodes = torch.cat((val_num_nodes, num_nodes), dim=0)

        ypred, att_adj = model(h0, adj, batch_num_nodes)
        val_preds[batch_idx] = ypred.cpu().detach().numpy()
        val_acc += torch.argmax(ypred) == label
    
    max_feat = 0
    feat_dim = all_feats[0].shape[-1]
    for k, v in all_adjs.items():
        max_feat = max_feat if max_feat >= v.shape[1] else v.shape[1]
 
    for k, v in val_adjs.items():
        max_feat = max_feat if max_feat >= v.shape[1] else v.shape[1]
 

   
    all_adjs_v = None
    all_feats_v = None
    all_labels_v = None
    all_preds_v = None
    all_num_nodes_v = None
    for k in all_adjs.keys():
        this_feat = all_adjs[k].shape[1]
        adj = np.zeros((max_feat, max_feat))
        adj[:this_feat, :this_feat] = all_adjs[k][:this_feat, :this_feat]
        feat = np.zeros((max_feat, feat_dim))
        feat[:this_feat] = all_feats[k][:this_feat, :]
        adj = np.expand_dims(adj, 0)
        feat = np.expand_dims(feat, 0)
        label = all_labels[k]
        label = np.expand_dims(label, 0)
        pred = all_preds[k][0]
        num_nodes = all_num_nodes[k]
        num_nodes = np.expand_dims(num_nodes, 0)


        adj = torch.tensor(adj)
        feat = torch.tensor(feat)
        label = torch.tensor(label)
        num_nodes = torch.tensor(num_nodes)

        if all_adjs_v is None:
            all_adjs_v = adj
            all_feats_v = feat
            all_labels_v = label
            all_preds_v = []
            all_preds_v.append(pred.tolist())
            all_num_nodes_v = num_nodes
        else:
            all_adjs_v = torch.cat((all_adjs_v, adj))
            all_feats_v = torch.cat((all_feats_v, feat))
            all_labels_v = torch.cat((all_labels_v, label))
            all_preds_v.append(pred.tolist())
            all_num_nodes_v = torch.cat((all_num_nodes_v, num_nodes))     

    val_adjs_v = None
    val_feats_v = None
    val_labels_v = None
    val_preds_v = None
    val_num_nodes_v = None
    for k in val_adjs.keys():
        this_feat = val_adjs[k].shape[1]
        adj = np.zeros((max_feat, max_feat))
        adj[:this_feat, :this_feat] = val_adjs[k][:this_feat, :this_feat]
        feat = np.zeros((max_feat, feat_dim))
        feat[:this_feat] = val_feats[k][:this_feat, :]
        adj = np.expand_dims(adj, 0)
        feat = np.expand_dims(feat, 0)
        label = val_labels[k]
        label = np.expand_dims(label, 0)
        pred = val_preds[k][0]
        num_nodes = val_num_nodes[k]
        num_nodes = np.expand_dims(num_nodes, 0)


        adj = torch.tensor(adj)
        feat = torch.tensor(feat)
        label = torch.tensor(label)
        num_nodes = torch.tensor(num_nodes)

        if val_adjs_v is None:
            val_adjs_v = adj
            val_feats_v = feat
            val_labels_v = label
            val_preds_v = []
            val_preds_v.append(pred.tolist())
            val_num_nodes_v = num_nodes
        else:
            val_adjs_v = torch.cat((val_adjs_v, adj))
            val_feats_v = torch.cat((val_feats_v, feat))
            val_labels_v = torch.cat((val_labels_v, label))
            val_preds_v.append(pred.tolist())
            val_num_nodes_v = torch.cat((val_num_nodes_v, num_nodes))    
    all_labels_v = all_labels_v.squeeze(1)
    val_labels_v = val_labels_v.squeeze(1)

    print(all_adjs_v.shape, all_feats_v.shape, all_labels_v.shape, all_num_nodes_v.shape, val_labels_v.shape, all_num_nodes_v[:5])
    cg_data = {
        "adj": all_adjs_v,
        "feat": all_feats_v,
        "label": all_labels_v,
        "pred": np.expand_dims(all_preds_v, axis=0),
        "num_nodes": all_num_nodes_v,
        "val_adj": val_adjs_v,
        "val_feat": val_feats_v,
        "val_label": val_labels_v,
        "val_pred": np.expand_dims(val_preds_v, axis=0),
        "val_num_nodes": val_num_nodes_v,
        "train_idx": len(train_loader)
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    
    print("ckpt saved")



#############################
#
# Training 
#
#############################

def train_pyg(args, model, device, train_loader, val_loader, test_loader):

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    print("epochs : ", args.num_epochs)
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        train_acc, val_acc = 0, 0

        model.train()
        loss_ep = 0.
        loss_count = 0.
        wrong_samples = 0
        for batch_idx, data in enumerate(train_loader):

            feats = data.x
            if (data.edge_index.shape[1] == 0):
                continue
            adj = to_dense_adj(data.edge_index)
            num_nodes = min(len(data.x), adj.shape[1])
            label = data.y
            if args.bmname == 'tox21':
                label = label[:,0]

            adj = Variable(adj, requires_grad=False).float().to(device)
            h0 = Variable(feats, requires_grad=False).float().to(device)
            label = Variable(label.long(), requires_grad=False).to(device)
            batch_num_nodes = [num_nodes]
            label = label.reshape((-1))

            if len(h0.shape) == 2:
                h0 = h0.unsqueeze(0) 
            if h0.shape[1] > num_nodes:
                wrong_samples += 1
                h0 = h0[:,:num_nodes,:]


            ypred, att_adj = model(h0, adj, batch_num_nodes)

            train_acc += torch.argmax(ypred) == label

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            loss_ep += loss
            loss_count += 1.0
            if loss_count > 200:
                model.zero_grad()
                loss_ep.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                loss_ep = 0.
                loss_count = 0.

            #model.zero_grad()
            #loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.clip)
            #optimizer.step() 


        
        if epoch%100 == 0 and epoch > 0:
            print("ckpt saved for epoch: ", epoch)
            cg_data = {}
            io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

        model.eval()
        for batch_idx, data in enumerate(val_loader):
            
            feats = data.x
            if (data.edge_index.shape[1] == 0):
                continue

            adj = to_dense_adj(data.edge_index)
            num_nodes = min(len(data.x), adj.shape[1])
            label = data.y
            if args.bmname == 'tox21':
                label = label[:,0]


            adj = Variable(adj, requires_grad=False).float().to(device)
            h0 = Variable(feats, requires_grad=False).float().to(device)
            label = Variable(label.long(), requires_grad=False).to(device)
            batch_num_nodes = [num_nodes]
            label = label.reshape((-1))
            
 
            if len(h0.shape) == 2:
                h0 = h0.unsqueeze(0) 
            if h0.shape[1] > num_nodes:
                h0 = h0[:,:num_nodes,:]


            ypred, att_adj = model(h0, adj, batch_num_nodes)

            val_acc += torch.argmax(ypred) == label

        print("Epoch: {}\tTrain accuracy: {}\tValid accuracy: {}".format(epoch, train_acc.item() / len(train_loader), val_acc.item() / len(val_loader)))

    print("wrong samples: ",wrong_samples, len(train_loader))
    test_acc = 0
    model.eval()
    for batch_idx, data in enumerate(test_loader):

        feats = data.x
        adj = to_dense_adj(data.edge_index)
        num_nodes = min(len(data.x), adj.shape[1])
        label = data.y
        if args.bmname == 'tox21':
            label = label[:,0]


        adj = Variable(adj, requires_grad=False).float().to(device)
        h0 = Variable(feats, requires_grad=False).float().to(device)
        label = Variable(label.long(), requires_grad=False).to(device)
        batch_num_nodes = [num_nodes]
        label = label.reshape((-1))
             
        if len(h0.shape) == 2:
            h0 = h0.unsqueeze(0) 
        if h0.shape[1] > num_nodes:
            h0 = h0[:,:num_nodes,:] 

       
        ypred, att_adj = model(h0, adj, batch_num_nodes)

        test_acc += torch.argmax(ypred) == label
    
    print("Test accuracy: {}".format(test_acc.item() / len(test_loader)))
    cg_data = {}
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)



def pyg_task(args, writer=None, feat="node-label"):
    dataset_name = args.bmname
    path = args.datadir
    #path = "/home/mohit/Mohit/gcn_interpretation/data/Graph-SST2"
    #path = "/home/mohit/Mohit/gcn_interpretation/data"
    #path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'datasets')
    dataset = datasets.get_dataset(dataset_dir=path, dataset_name=dataset_name)
    dataset.process()
    if args.cg:
        args.batch_size = 1
        batch_size = 1
    else:
        args.batch_size = 1
        batch_size = 1
    dataloaders = datasets.get_dataloader(dataset, batch_size=args.batch_size, split_ratio=[0.7, 0.2], random_split_flag=True)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    input_dim = next(iter(test_loader)).x.shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).to(device)
    else:
        print("Method: base")
        print("embed:", args.add_embedding)
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            device=device,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).to(device)
    if args.cg:
        #print("Loading ckpt .....")
        #ckpt_dict = io_utils.load_ckpt(args)
        #model.load_state_dict(ckpt_dict['model_state'])
        extract_cg_pyg(args, model, device, train_loader, val_loader)
        return
    train_pyg(args, model, device, train_loader, val_loader, test_loader)



#############################
#
# Prepare Data
#
#############################
def prepare_synthetic_data(args):

    if args.bmname == 'old_synthetic':
        #data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_2label_3sublabel/synthetic_data.p","rb"))

        data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_nofake.p","rb"))
        #data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20.p","rb"))
        train_idx = 7000
        val_idx = 7500
    # # # indices = np.random.permutation(4000)
        indices = np.array(list(range(8000)))



    #data = pickle.load(open("../data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20.p","rb")) #good
    else:
        data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p","rb"))
        train_idx = 7000
        val_idx = 7500
        indices = np.array(list(range(8000)))

    X_train_feat = torch.from_numpy(data['feat'][indices[:train_idx]]).float()
    X_train_adj = torch.from_numpy(data['adj'][indices[:train_idx]]).float()
    X_train_nodes = torch.from_numpy(data['num_nodes'][indices[:train_idx]])
    y_train_label = torch.from_numpy(data['label'][:train_idx])
    # y_train_label = torch.from_numpy(data['sub_label'][indices[:train_idx]]) + 1

    # data['sub_label']

    X_val_feat = torch.from_numpy(data['feat'][indices[train_idx:val_idx]]).float()
    X_val_adj = torch.from_numpy(data['adj'][indices[train_idx:val_idx]]).float()
    X_val_nodes = torch.from_numpy(data['num_nodes'][indices[train_idx:val_idx]])
    y_val_label = torch.from_numpy(data['label'][indices[train_idx:val_idx]])
    # y_val_label = torch.from_numpy(data['sub_label'][indices[train_idx:val_idx]]) + 1



    X_test_feat = torch.from_numpy(data['feat'][indices[val_idx:]]).float()
    X_test_adj = torch.from_numpy(data['adj'][indices[val_idx:]]).float()
    X_test_nodes = torch.from_numpy(data['num_nodes'][indices[val_idx:]])
    y_test_label = torch.from_numpy(data['label'][indices[val_idx:]])
    # y_test_label = torch.from_numpy(data['sub_label'][indices[val_idx:]]) + 1



    print("debug: ", X_train_feat.shape, X_train_adj.shape, X_train_nodes.shape, y_train_label.shape)
    train_dataset = torch.utils.data.TensorDataset(X_train_feat, X_train_adj, X_train_nodes, y_train_label)
    print("debug: ", X_val_feat.shape, X_val_adj.shape, X_val_nodes.shape, y_val_label.shape)
    val_dataset = torch.utils.data.TensorDataset(X_val_feat, X_val_adj, X_val_nodes, y_val_label)
    print("debug: ", X_test_feat.shape, X_test_adj.shape, X_test_nodes.shape, y_test_label.shape)
    test_dataset = torch.utils.data.TensorDataset(X_test_feat, X_test_adj, X_test_nodes, y_test_label)

    if args.cg:
        train_shuffle = False
    else:
        train_shuffle = True

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
    )

    val_dataset_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        X_train_feat.shape[1], #max nodes
        X_train_feat.shape[2],#32, #max feat dim after embedding
        X_train_feat.shape[2],#32, #assign feat dim?
    )




def prepare_genome_data(args,max_nodes=0):
    df = pd.read_csv('../data/gcn_data/SC980可解释性数据/code/gene_data_filter_980_x_train.csv', index_col=0)
    lab = pd.read_csv('../data/gcn_data/SC980可解释性数据/code/gene_data_filter_980_y_train.csv', index_col=0)
    X_all = torch.from_numpy(df.values)
    y_all = torch.from_numpy(lab.iloc[:, 0].values)

    # shuffle
    indices = torch.randperm(X_all.shape[0])
    X_all = X_all[indices, :]
    y_all = y_all[indices]


    train_idx = int(X_all.shape[0] * args.train_ratio)
    test_idx = int(X_all.shape[0] * (1 - args.test_ratio))
    X_train = X_all[:train_idx]
    y_train = y_all[:train_idx]
    X_val = X_all[train_idx:test_idx]
    y_val = y_all[train_idx:test_idx]
    X_test = X_all[test_idx:]
    y_test = y_all[test_idx:]

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)



    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataset_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        5129, #max nodes
        1,#32, #max feat dim after embedding
        1,#32, #assign feat dim?
    )



def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    adjs_v = None
    adjs_2 = None
    # adjs_v = pickle.load(open("../data/Mutagenicity/adjs_v.p", "rb"))
    # adjs_2 = pickle.load(open("../data/Mutagenicity/adjs_2.p", "rb"))

    if adjs_v is not None:
        c = list(zip(graphs, adjs_v, adjs_2))
        random.shuffle(c)
        graphs, adjs_v, adjs_2 = zip(*c)
    # else:
    #if not only_save_cg:
    random.shuffle(graphs)

    # print(len(graphs), len(adjs_v), len(adjs_2))
    # exit()
    train_adjs_v = None
    val_adjs_v = None
    test_adjs_v = None
    train_adjs_2 = None
    val_adjs_2 = None
    test_adjs_2 = None

    if args.cg:
        train_idx = len(graphs) - 2
        test_idx = len(graphs) - 1
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:test_idx]
        test_graphs = graphs[test_idx:]
    else:
        if test_graphs is None:
            train_idx = int(len(graphs) * args.train_ratio)
            test_idx = int(len(graphs) * (1 - args.test_ratio))
            train_graphs = graphs[:train_idx]
            val_graphs = graphs[train_idx:test_idx]
            test_graphs = graphs[test_idx:]
            if adjs_v is not None:
                train_adjs_v = adjs_v[:train_idx]
                val_adjs_v = adjs_v[train_idx:test_idx]
                test_adjs_v = adjs_v[test_idx:]
                train_adjs_2 = adjs_2[:train_idx]
                val_adjs_2 = adjs_2[train_idx:test_idx]
                test_adjs_2 = adjs_2[test_idx:]
        else:
            train_idx = int(len(graphs) * args.train_ratio)
            train_graphs = graphs[:train_idx]
            val_graphs = graph[train_idx:]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    print("Number of graphs: ", len(graphs))
    print("Number of edges: ", sum([G.number_of_edges() for G in graphs]))
    print(
        "Max, avg, std of graph size: ",
        max([G.number_of_nodes() for G in graphs]),
        ", " "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
        ", " "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])),
    )

    # minibatch
    dataset_sampler = graph_utils.GraphSampler(
        train_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
        adjs_v= train_adjs_v,
        adjs_2=train_adjs_2

    )
    if args.cg:
        train_shuffle = False
    else:
        train_shuffle = True

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        val_graphs, 
        normalize=False, 
        max_num_nodes=max_nodes, 
        features=args.feature_type,
        adjs_v = val_adjs_v,
        adjs_2=val_adjs_2

    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_sampler = graph_utils.GraphSampler(
        test_graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
        adjs_v=test_adjs_v,
        adjs_2=test_adjs_2

    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return (
        train_dataset_loader,
        val_dataset_loader,
        test_dataset_loader,
        dataset_sampler.max_num_nodes,
        dataset_sampler.feat_dim,
        dataset_sampler.assign_feat_dim
    )


#############################
#
# Training 
#
#############################
def train_synthetic(
    dataset,
    model,
    args,
    device,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):

    if args.cg:
        print("Loading ckpt .....")
        ckpt_dict = io_utils.load_ckpt(args)
        model.load_state_dict(ckpt_dict['model_state'])
        args.num_epochs = 1
        model.eval()
        torch.no_grad()
    breaking = False
    save_batches = 350
    #save_batches = 150
    # best_model = copy.deepcopy(model.cpu())

    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    print("epochs : ", args.num_epochs)

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        all_embs = []

        print("Epoch: ", epoch)
        for batch_idx, (feats, adj, num_nodes, label) in enumerate(dataset):
            # feats = feats.unsqueeze(2)
           
            model.zero_grad()
            if args.cg:
                if batch_idx == 0:
                    prev_feats = feats
                    prev_labels = label
                    prev_adjs = adj
                    prev_num_nodes = num_nodes
                    all_feats = prev_feats
                    all_labels = prev_labels
                    all_adjs = prev_adjs
                    all_num_nodes = prev_num_nodes

                elif batch_idx < save_batches:  # only storing 1st 20 batches, (20*20 graphs)why?
                    prev_feats = feats
                    prev_labels = label
                    prev_adjs = adj
                    prev_num_nodes = num_nodes
                    all_feats = torch.cat((all_feats, prev_feats), dim=0)
                    all_labels = torch.cat((all_labels, prev_labels), dim=0)
                    all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                    all_num_nodes = torch.cat((all_num_nodes, prev_num_nodes), dim=0)
            adj = Variable(adj, requires_grad=False).to(device)
            h0 = Variable(feats, requires_grad=False).to(device)
            label = Variable(label.long(), requires_grad=False).to(device)
            batch_num_nodes = num_nodes.numpy().tolist()
            ypred, att_adj = model(h0, adj, batch_num_nodes)
            emb = model.getEmbeddings(h0, adj, batch_num_nodes)
            if batch_idx < save_batches:
                if args.cg:
                    predictions += ypred.cpu().detach().numpy().tolist()
                    all_embs += emb.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            if not args.cg:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            else:
                if batch_idx > save_batches:
                    breaking = True
                    print("Breaking....................")
                    break

            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if args.cg:
            break

        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate_synthetic(dataset, model, args, device, name="Train", max_num_examples=100)

        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate_synthetic(val_dataset, model, args, device, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
            # best_model = copy.deepcopy(model.cpu())
        # test_dataset = None
        if test_dataset is not None:
            test_result = evaluate_synthetic(test_dataset, model, args, device, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])
    if not args.cg:
        matplotlib.style.use("seaborn")
        plt.switch_backend("agg")
        plt.figure()
        plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
        if test_dataset is not None:
            plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
            plt.legend(["train", "val", "test"])
        else:
            plt.plot(best_val_epochs, best_val_accs, "bo")
            plt.legend(["train", "val"])
        plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
        plt.close()
        matplotlib.style.use("default")


    # print(all_adjs.shape, all_feats.shape, all_labels.shape)
    if args.cg:
        cg_data = {
            "adj": all_adjs,
            "feat": all_feats,
            "label": all_labels,
            "emb": np.expand_dims(all_embs, axis=0)[0],
            "pred": np.expand_dims(predictions, axis=0),
            "num_nodes": all_num_nodes,
            "train_idx": list(range(len(dataset))),
        }
    else:
        cg_data = {}
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs


def train_genome(
    adj_np,
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []


    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []

        print("Epoch: ", epoch)
        for batch_idx, (feats, label) in enumerate(dataset):
            feats = feats.unsqueeze(2)
            model.zero_grad()
            if batch_idx == 0:
                prev_feats = feats
                prev_labels = label
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:  #only storing 1st 20 batches, (20*20 graphs)why?
                prev_feats = feats
                prev_labels = label
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)

            adj = Variable(torch.from_numpy(adj_np).float(), requires_grad=False).cuda()
            adj = adj.unsqueeze(0).expand(feats.shape[0], adj.shape[0], adj.shape[1])
            h0 = Variable(feats.float(), requires_grad=False).cuda()
            label = Variable(label.long(), requires_grad=False).cuda()

            batch_num_nodes = None



            ypred, att_adj = model(h0, adj, batch_num_nodes)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate_genome(dataset, model, args, adj_np, name="Train", max_num_examples=100)

        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate_genome(val_dataset, model, args, adj_np, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        test_dataset = None
        if test_dataset is not None:
            test_result = evaluate_genome(test_dataset, model, args, adj_np, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)

    cg_data = {
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    return model, val_accs



def train(
    dataset,
    model,
    args,
    same_feat=True,
    val_dataset=None,
    test_dataset=None,
    writer=None,
    mask_nodes=True,
    train_adjs = None,
    val_adjs = None
):
    if args.cg:
        print("Loading ckpt .....")
        ckpt_dict = io_utils.load_ckpt(args)
        model.load_state_dict(ckpt_dict['model_state'])
    breaking = False
    save_batches = 150

    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        all_embs = []
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            if args.cg:
                if batch_idx == 0:
                    prev_adjs = data["adj"]
                    prev_feats = data["feats"]
                    prev_labels = data["label"]
                    prev_num_nodes = data["num_nodes"].int()
                    all_adjs = prev_adjs
                    all_feats = prev_feats
                    all_labels = prev_labels
                    all_num_nodes = prev_num_nodes
                elif batch_idx < save_batches:  #only storing 1st 20 batches, (20*20 graphs)why?
                    prev_adjs = data["adj"]
                    prev_feats = data["feats"]
                    prev_labels = data["label"]
                    prev_num_nodes = data["num_nodes"].int()

                    all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                    all_feats = torch.cat((all_feats, prev_feats), dim=0)
                    all_labels = torch.cat((all_labels, prev_labels), dim=0)
                    all_num_nodes = torch.cat((all_num_nodes, prev_num_nodes), dim=0)
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            if "adj_v" in data:
                adj_2 = Variable(data["adj_2"].float(), requires_grad=False).cuda()
                adj_v = Variable(data["adj_v"].float(), requires_grad=False).cuda()
            else:
                adj_v = None
                adj_2 = None
            h0 = Variable(data["feats"].float(), requires_grad=False).cuda()
            label = Variable(data["label"].long()).cuda()
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()

            # ypred, att_adj = model(h0, adj_v, batch_num_nodes, assign_x=assign_input)
            if adj_v is not None:
                ypred, att_adj = model(h0, adj_2, batch_num_nodes, assign_x=assign_input, adj_v=adj_v)
            else:
                ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            # print("pred: ", ypred.shape) #bs*nclasses
            emb = model.getEmbeddings(h0, adj, batch_num_nodes)
            if batch_idx < save_batches:
                if args.cg:
                    all_embs += emb.cpu().detach().numpy().tolist()
                    predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            if not args.cg:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            else:
                if batch_idx > save_batches:
                    breaking = True
                    print("Breaking....................")
                    break


            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time

        if args.cg:
            result = evaluate(dataset, model, args, name="Train", max_num_examples=200)
            print("Train accuracy: ", result["acc"])
            val_result = evaluate(val_dataset, model, args, name="Validation")
            print("val accuracy: ", val_result["acc"])
            break
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar("loss/linkpred_loss", model.link_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(dataset, model, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    if not args.cg:
        matplotlib.style.use("seaborn")
        plt.switch_backend("agg")
        plt.figure()
        plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
        if test_dataset is not None:
            plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
            plt.legend(["train", "val", "test"])
        else:
            plt.plot(best_val_epochs, best_val_accs, "bo")
            plt.legend(["train", "val"])
        plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
        plt.close()
        matplotlib.style.use("default")

    if args.cg:
        print(all_adjs.shape, all_feats.shape, all_labels.shape)

        print("preds:", len(predictions), np.expand_dims(predictions, axis=0).shape)

        cg_data = {
            "adj": all_adjs,
            "feat": all_feats,
            "label": all_labels,
            "emb": np.expand_dims(all_embs, axis=0)[0],
            "pred": np.expand_dims(predictions, axis=0),
            "train_idx": list(range(len(dataset))),
            "num_nodes": all_num_nodes
        }
    else:
        cg_data = {}
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
    print("ckpt saved!!!")
    return model, val_accs


def train_node_classifier(G, labels, model, args, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None

    all_embs = []
    
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        if args.gpu:
            loss = model.loss(ypred_train, labels_train.cuda())
        else:
            loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        if epoch == args.num_epochs - 1:
            emb = model.getEmbeddings(x.cuda(), adj.cuda())
            all_embs += emb.cpu().detach().numpy().tolist()

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred, _ = model(x.cuda(), adj.cuda())
    else:
        ypred, _ = model(x, adj)
    cg_data = {
        "adj": data["adj"],
        "feat": data["feat"],
        "label": data["labels"],
        "emb": np.expand_dims(all_embs, axis=0)[0],
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    print(data["adj"].shape, data["feat"].shape)
    # import pdb
    # pdb.set_trace()
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

activation = []
def get_activation(name):
    def hook(model, input, output):
        activation.append(output)
    return hook


def evaluate_node_classifier(G, labels, model, args, label_dict=None, writer=None):
    # train/test split only for nodes
    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    model.conv_last.register_forward_hook(get_activation('conv_last'))

    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred, adj_att = model(x.cuda(), adj.cuda())
        else:
            ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), data["labels"], train_idx, test_idx
        )

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import seaborn as sns
    import pandas as pd
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    acts = pd.DataFrame(activation[0][0][0])
    results = tsne.fit_transform(acts)
    df = pd.DataFrame(results, columns = ["ax_1", "ax_2"])

    if label_dict is not None: 
        label_dict = [label_dict[v] for v in label_dict.keys()]
    else:
        label_dict = [labels[v] for v in G.nodes()]
    df['label'] = label_dict
    # df['label'] = data["labels"][0]

    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="ax_1", y="ax_2",
        data = df,
        hue="label",
        legend="full"
    )
    plt.savefig("test.png")


    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # # computation graph
    # model.eval()
    # if args.gpu:
    #     ypred, _ = model(x.cuda(), adj.cuda())
    # else:
    #     ypred, _ = model(x, adj)
    # cg_data = {
    #     "adj": data["adj"],
    #     "feat": data["feat"],
    #     "label": data["labels"],
    #     "pred": ypred.cpu().detach().numpy(),
    #     "train_idx": train_idx,
    # }
    # import pdb
    # pdb.set_trace()
    # io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

def train_node_classifier_multigraph(G_list, labels, model, args, writer=None):
    train_idx_all, test_idx_all = [], []
    # train/test split only for nodes
    num_nodes = G_list[0].number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    train_idx_all.append(train_idx)
    test_idx = idx[num_train:]
    test_idx_all.append(test_idx)

    data = gengraph.preprocess_input_graph(G_list[0], labels[0])
    all_labels = data["labels"]
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)
    adj = torch.tensor(data["adj"], dtype=torch.float)
    x = torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)

    for i in range(1, len(G_list)):
        np.random.shuffle(idx)
        train_idx = idx[:num_train]
        train_idx_all.append(train_idx)
        test_idx = idx[num_train:]
        test_idx_all.append(test_idx)
        data = gengraph.preprocess_input_graph(G_list[i], labels[i])
        all_labels = np.concatenate((all_labels, data["labels"]), axis=0)
        labels_train = torch.cat(
            [
                labels_train,
                torch.tensor(data["labels"][:, train_idx], dtype=torch.long),
            ],
            dim=0,
        )
        adj = torch.cat([adj, torch.tensor(data["adj"], dtype=torch.float)])
        x = torch.cat(
            [x, torch.tensor(data["feat"], requires_grad=True, dtype=torch.float)]
        )

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()

        if args.gpu:
            ypred = model(x.cuda(), adj.cuda())
        else:
            ypred = model(x, adj)
        # normal indexing
        ypred_train = ypred[:, train_idx, :]
        # in multigraph setting we can't directly access all dimensions so we need to gather all the training instances
        all_train_idx = [item for sublist in train_idx_all for item in sublist]
        ypred_train_cmp = torch.cat(
            [ypred[i, train_idx_all[i], :] for i in range(10)], dim=0
        ).reshape(10, 146, 6)
        if args.gpu:
            loss = model.loss(ypred_train_cmp, labels_train.cuda())
        else:
            loss = model.loss(ypred_train_cmp, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #for param_group in optimizer.param_groups:
        #    print(param_group["lr"])
        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(
            ypred.cpu(), all_labels, train_idx_all, test_idx_all
        )
        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss, epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )

        print(
            "epoch: ",
            epoch,
            "; loss: ",
            loss.item(),
            "; train_acc: ",
            result_train["acc"],
            "; test_acc: ",
            result_test["acc"],
            "; epoch time: ",
            "{0:0.2f}".format(elapsed),
        )

        if scheduler is not None:
            scheduler.step()
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])

    # computation graph
    model.eval()
    if args.gpu:
        ypred = model(x.cuda(), adj.cuda())
    else:
        ypred = model(x, adj)
    cg_data = {
        "adj": adj.cpu().detach().numpy(),
        "feat": x.cpu().detach().numpy(),
        "label": all_labels,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx_all,
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


#############################
#
# Evaluate Trained Model
#
#############################
def evaluate_synthetic(dataset, model, args, device='cpu', name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, (feats, adj, num_nodes, label) in enumerate(dataset):
        # feats = feats.unsqueeze(2)
        adj = Variable(adj, requires_grad=False).to(device)
        h0 = Variable(feats.float(), requires_grad=False).to(device)
        labels.append(label.numpy())


        label = Variable(label.long(), requires_grad=False).to(device)

        batch_num_nodes = num_nodes.numpy().tolist()

        ypred, att_adj = model(h0, adj, batch_num_nodes)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    # print("labels: ", len(labels), labels[:20])
    # print("preds: ", len(preds), preds[:20])
    # exit()

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_genome(dataset, model, args, adj_np, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, (feats, label) in enumerate(dataset):
        feats = feats.unsqueeze(2)
        adj = Variable(torch.from_numpy(adj_np).float(), requires_grad=False).cuda()
        adj = adj.unsqueeze(0).expand(feats.shape[0], adj.shape[0], adj.shape[1])
        h0 = Variable(feats.float(), requires_grad=False).cuda()
        labels.append(label.numpy())

        label = Variable(label.long(), requires_grad=False).cuda()

        batch_num_nodes = None

        ypred, att_adj = model(h0, adj, batch_num_nodes)


        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result

def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).cuda()
        if "adj_v" in data:
            adj_2 = Variable(data["adj_2"].float(), requires_grad=False).cuda()

            adj_v = Variable(data["adj_v"].float(), requires_grad=False).cuda()
        else:
            adj_v = None
            adj_2 = None


        h0 = Variable(data["feats"].float()).cuda()
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(
            data["assign_feats"].float(), requires_grad=False
        ).cuda()

        # ypred, att_adj = model(h0, adj_v, batch_num_nodes, assign_x=assign_input)
        if adj_v is not None:
            ypred, att_adj = model(h0, adj_2, batch_num_nodes, assign_x=assign_input, adj_v=adj_v)
        else:
            ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)


        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test



#############################
#
# Run Experiments
#
#############################
def ppi_essential_task(args, writer=None):
    feat_file = "G-MtfPathways_gene-motifs.csv"
    # G = io_utils.read_biosnap('data/ppi_essential', 'PP-Pathways_ppi.csv', 'G-HumanEssential.tsv',
    #        feat_file=feat_file)
    G = io_utils.read_biosnap(
        "data/ppi_essential",
        "hi-union-ppi.tsv",
        "G-HumanEssential.tsv",
        feat_file=feat_file,
    )
    labels = np.array([G.nodes[u]["label"] for u in G.nodes()])
    num_classes = max(labels) + 1
    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        args.loss_weight = torch.tensor([1, 5.0], dtype=torch.float).cuda()
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task1(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn1(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if args.method == "att":
        print("Method: att")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            args=args,
        )
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
	    device=device,
            bn=args.bn,
            args=args,
        )
    if args.gpu:
        model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task2(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn2()
    input_dim = len(G.nodes[0]["feat"])
    num_classes = max(labels) + 1

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task3(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn3(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    num_classes = max(labels) + 1
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            device = device,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task4(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn4(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    num_classes = max(labels) + 1

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method:", args.method)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)


def syn_task5(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn5(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

def syn_task6(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn6(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        

        if args.gpu:
            model = model.cuda()
    # model.load_state_dict(torch.load('./ckpt/syn6_base_h20_o20.pth.tar')['model_state'])
    # evaluate_node_classifier(G, labels, model, args, writer=writer)

def syn_task7(args, writer=None):
    # data
    G, labels, name, label_dict = gengraph.gen_syn7(
        feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    )
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]


    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

    # model.load_state_dict(torch.load('./ckpt/syn7_base_h20_o20.pth.tar')['model_state'])
    # evaluate_node_classifier(G, labels, model, args, label_dict, writer=writer)

def syn_task8(args, writer=None):
    # data
    G, labels, name = gengraph.gen_syn8()
    print(labels)
    print("Number of nodes: ", G.number_of_nodes())
    num_classes = max(labels) + 1

    input_dim = G.nodes[next(iter(G.nodes()))]["feat"].shape[0]


    if args.method == "attn":
        print("Method: attn")
    else:
        print("Method: base")
        model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )

        if args.gpu:
            model = model.cuda()

    train_node_classifier(G, labels, model, args, writer=writer)

    # model.load_state_dict(torch.load('./ckpt/syn8_base_h20_o20.pth.tar')['model_state'])
    # evaluate_node_classifier(G, labels, model, args, writer=writer)



def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), "rb") as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph["label"] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph["label"] = test_labels[i]

    if feat is None:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(
        graphs, args, test_graphs=test_graphs
    )
    model = models.GcnEncoderGraph(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.num_classes,
        args.num_gc_layers,
        bn=args.bn,
    ).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, "Validation")


def enron_task_multigraph(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            net.add_nodes_from(range(max_enron_id))
            labels = [n[1].get("role", "None") for n in net.nodes(data=True)]
            labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            labels_list.append(labels_num)
        # train_dataset, test_dataset, max_num_nodes = prepare_data(G_list, args)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        if args.gpu:
            model = model.cuda()
        print(labels_num)
        train_node_classifier_multigraph(
            G_list, labels_list, model, args, writer=writer
        )
    else:
        print("Running Enron full task")


def enron_task(args, idx=None, writer=None):
    labels_dict = {
        "None": 5,
        "Employee": 0,
        "Vice President": 1,
        "Manager": 2,
        "Trader": 3,
        "CEO+Managing Director+Director+President": 4,
    }
    max_enron_id = 183
    if idx is None:
        G_list = []
        labels_list = []
        for i in range(10):
            net = pickle.load(
                open("data/gnn-explainer-enron/enron_slice_{}.pkl".format(i), "rb")
            )
            # net.add_nodes_from(range(max_enron_id))
            # labels=[n[1].get('role', 'None') for n in net.nodes(data=True)]
            # labels_num = [labels_dict[l] for l in labels]
            featgen_const = featgen.ConstFeatureGen(
                np.ones(args.input_dim, dtype=float)
            )
            featgen_const.gen_node_features(net)
            G_list.append(net)
            print(net.number_of_nodes())
            # labels_list.append(labels_num)

        G = nx.disjoint_union_all(G_list)
        model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            len(labels_dict),
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
        labels = [n[1].get("role", "None") for n in G.nodes(data=True)]
        labels_num = [labels_dict[l] for l in labels]
        for i in range(5):
            print("Label ", i, ": ", labels_num.count(i))

        print("Total num nodes: ", len(labels_num))
        print(labels_num)

        if args.gpu:
            model = model.cuda()
        train_node_classifier(G, labels_num, model, args, writer=writer)
    else:
        print("Running Enron full task")

def synthetic_task(args, writer=None, feat="node-label"):

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_synthetic_data(
        args
    )
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).cuda()
    else:
        print("Method: base")
        print("embed:", args.add_embedding)

        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).to(device)

    train_synthetic(
        train_dataset,
        model,
        args,
        device,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )
    evaluate_synthetic(test_dataset, model, args, device, "Validation")

def genome_task(args, writer=None, feat="node-label"):
    adj = pickle.load(open("./genome_adj.p", "rb"))

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_genome_data(
        args, max_nodes=args.max_nodes
    )
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).cuda()
    else:
        print("Method: base")
        print("embed:", args.add_embedding)

        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

    train_genome(
        adj,
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )
    evaluate_genome(test_dataset, model, args, "Validation")

def benchmark_task(args, writer=None, feat="node-label"):
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )
    print(max([G.graph["label"] for G in graphs]))

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
                # make it -1/1 instead of 0/1
                # feat = np.array(G.nodes[u]['label'])
                # G.nodes[u]['feat'] = feat * 2 - 1
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    #all the graphs of data are already processed here as nx graphs

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(
        graphs, args, max_nodes=args.max_nodes
    )
    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).cuda()
    else:
        print("Method: base")
        print("embed:", args.add_embedding)
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()


    train(
        train_dataset,
        model,
        args,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        writer=writer,
    )
    # evaluate(test_dataset, model, args, "Validation")

def pyg_task(args, writer=None, feat="node-label"):
    dataset_name = args.bmname
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'datasets')
    dataset = datasets.get_dataset(dataset_dir=path, dataset_name=dataset_name)
    dataset.process()
    dataloaders = datasets.get_dataloader(dataset, batch_size=1, split_ratio=[0.7, 0.2], random_split_flag=True)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    print(next(iter(train_loader)))

    input_dim = next(iter(train_loader)).x.shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    max_num_nodes = 0
    for i, data in enumerate(train_loader):
        max_num_nodes = max_num_nodes if max_num_nodes > len(data.x) else len(data.x)
    for i, data in enumerate(val_loader):
        max_num_nodes = max_num_nodes if max_num_nodes > len(data.x) else len(data.x)
    for i, data in enumerate(test_loader):
        max_num_nodes = max_num_nodes if max_num_nodes > len(data.x) else len(data.x)

    if args.method == "soft-assign":
        print("Method: soft-assign")
        model = models.SoftPoolingGcnEncoder(
            max_num_nodes,
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args.hidden_dim,
            assign_ratio=args.assign_ratio,
            num_pooling=args.num_pool,
            bn=args.bn,
            dropout=args.dropout,
            linkpred=args.linkpred,
            args=args,
            assign_input_dim=assign_input_dim,
        ).to(device)
    else:
        print("Method: base")
        print("embed:", args.add_embedding)
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            device=device,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).to(device)
    if args.cg:
        print("Loading ckpt .....")
        ckpt_dict = io_utils.load_ckpt(args)
        model.load_state_dict(ckpt_dict['model_state'])
        extract_cg_pyg(args, model, device, train_loader, val_loader)
        return
    train_pyg(args, model, device, train_loader, val_loader, test_loader)#, batch_size=args.batch_size)

def benchmark_task_val(args, writer=None, feat="node-label"):
    all_vals = []
    graphs = io_utils.read_graphfile(
        args.datadir, args.bmname, max_nodes=args.max_nodes
    )

    if feat == "node-feat" and "feat_dim" in graphs[0].graph:
        print("Using node features")
        input_dim = graphs[0].graph["feat_dim"]
    elif feat == "node-label" and "label" in graphs[0].nodes[0]:
        print("Using node labels")
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]["feat"] = np.array(G.nodes[u]["label"])
    else:
        print("Using constant labels")
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # 10 splits
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(
            graphs, args, i, max_nodes=args.max_nodes
        )
        print("Method: base")
        model = models.GcnEncoderGraph(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            bn=args.bn,
            dropout=args.dropout,
            args=args,
        ).cuda()

        _, val_accs = train(
            train_dataset,
            model,
            args,
            val_dataset=val_dataset,
            test_dataset=None,
            writer=writer,
        )
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphPool arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument(
        "--assign-ratio",
        dest="assign_ratio",
        type=float,
        help="ratio of number of nodes in consecutive layers",
    )
    softpool_parser.add_argument(
        "--num-pool", dest="num_pool", type=int, help="number of pooling layers"
    )
    parser.add_argument(
        "--linkpred",
        dest="linkpred",
        action="store_const",
        const=True,
        default=False,
        help="Whether link prediction side objective is used",
    )

    parser_utils.parse_optimizer(parser)

    parser.add_argument(
        "--datadir", dest="datadir", help="Directory where benchmark is located"
    )
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--max-nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs training set to all graphs.",
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument(
        "--input-dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )

    parser.add_argument(
        "--cg",
        dest="cg",
        action="store_const",
        const=True,
        default=False,
        help="save cg",
    )

    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--add-self", dest="add_self", help="add self "
    )

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    parser.add_argument(
        "--add_embedding", dest="add_embedding", default=False, help="add embedding layer "
    )

    parser.add_argument(
        "--pred-hidden-dim",
        dest="pred_hidden_dim",
        type=int,
        help="hidden dims",
        default=20
    )

    parser.add_argument(
        "--pred-num-layers",
        dest="pred_num_layers",
        type=int,
        help="num layers",
        default=0
    )
    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=300,
        train_ratio=0.8,
        test_ratio=0.1,
        num_workers=1,
        input_dim=10,
        hidden_dim=20,
        output_dim=20,
        num_classes=2,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        add_self="none",
        name_suffix="",
        assign_ratio=0.1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    if prog_args.bmname is not None:
        if prog_args.bmname == 'genome':
            genome_task(prog_args, writer=writer)
        elif prog_args.bmname == 'synthetic' or prog_args.bmname == 'old_synthetic':
            synthetic_task(prog_args, writer=writer)
        else:
            pyg_task(prog_args, writer=writer)
            #benchmark_task(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == "syn1":
            syn_task1(prog_args, writer=writer)
        elif prog_args.dataset == "syn2":
            syn_task2(prog_args, writer=writer)
        elif prog_args.dataset == "syn3" or prog_args.dataset == "repeat_syn3":
            syn_task3(prog_args, writer=writer)
        elif prog_args.dataset == "syn4" or prog_args.dataset == "dense_syn4":
            syn_task4(prog_args, writer=writer)
        elif prog_args.dataset == "syn5":
            syn_task5(prog_args, writer=writer)
        elif prog_args.dataset == "syn6":
            syn_task6(prog_args, writer=writer)
        elif prog_args.dataset == "syn7":
            syn_task7(prog_args, writer=writer)
        elif prog_args.dataset == "syn8":
            syn_task8(prog_args, writer=writer)
        elif prog_args.dataset == "enron":
            enron_task(prog_args, writer=writer)
        elif prog_args.dataset == "ppi_essential":
            ppi_essential_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()



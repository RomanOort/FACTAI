""" explainer_main.py

     Main user interface for the explainer module.
"""
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import random

import pickle
import shutil
import torch

import models
import utils.io_utils as io_utils
# from explainer import explain_node_bondary as explain
# from explainer import explain_boundary_nn as explain

# from explainer import explain_boundary_joint
from explainer import explain_gnnexplainer
from explainer import explain_pgexplainer# _node as explain_pgexplainer
# from explainer import explain_twopgexplainer# _node as explain_pgexplainer
# from explainer import explain_pgexplainer_node
from explainer import explain_rcexplainer_noldb
from explainer import explain_rcexplainer
# from explainer import explain_tworcexplainer
# from explainer import explain_rcnoiseexplainer
# from explainer import explain_rcadversarialexplainer
# from explainer import explain_boundary_inverse
from explainer import explain_pgmexplainer
# from explainer import explain_attn
# from explainer import explain_grad
# from explainer import explain_random
# from explainer import explain_boundary_two_masks

# from explainer import explain_boundary_joint as explain

import utils.accuracy_utils as accuracy_utils

from gcn import *

import configs

def main(config=None):
    # Load a configuration
    if config is None:
        prog_args = configs.arg_parse()
    else:
        prog_args = config

    sparsity, fidelity, noise_level, roc_auc = None, None, None, None
    torch.manual_seed(prog_args.seed)
    random.seed(prog_args.seed)
    np.random.seed(prog_args.seed)

    if prog_args.gpu:
        torch.cuda.manual_seed(prog_args.seed)
        torch.cuda.manual_seed_all(prog_args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None


    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph
    print("Loaded model from {}".format(prog_args.ckptdir))

    input_dim = cg_dict["feat"].shape[2] #n*nodes*dim
    num_classes = cg_dict["pred"].shape[2] #n*2 why?
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    if prog_args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # build model
    print("Method: ", prog_args.method)
    if graph_mode:

        pred_hidden_dims = [prog_args.pred_hidden_dim] * prog_args.pred_num_layers

        if prog_args.bmname == "MNISTSuperpixels":
            # Hardcode to match train.py
            pred_hidden_dims = [20, 10]

        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            pred_hidden_dims=pred_hidden_dims,
            bn=prog_args.bn,
            args=prog_args,
            device=device
        )
    else:
        if prog_args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).cuda() 
        # Explain Node prediction
        # NOTE FIXME: MODEL ABOVE IS USUALLY USED
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            pred_hidden_dims=[prog_args.pred_hidden_dim] * prog_args.pred_num_layers,
            bn=prog_args.bn,
            args=prog_args,
            device=device
        )

    model = model.to(device)

    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    # print("===== checkpoint")
    # for k,v in ckpt["model_state"].items():
    #     print(k, v.shape)
    #
    # print("===== model")
    # for k,v in model.state_dict().items():
    #     print(k, v.shape)
    # # print(model.state_dict())
    # s = model.conv_first.weight.shape
    # print("model.conv_first.weight.shape", s)
    # model.load_state_dict(ckpt["model_state"])
    #
    # print("---DONE")



    # Create explainer
    dict_num_nodes = None
    if "num_nodes" in cg_dict:
        dict_num_nodes = cg_dict["num_nodes"]
    
    try:
        if "num_nodes" in cg_dict:
            dict_num_nodes = torch.cat([dict_num_nodes, cg_dict["val_num_nodes"]])
    
        adj = torch.cat([cg_dict["adj"], cg_dict["val_adj"]])
        feat = torch.cat([cg_dict["feat"], cg_dict["val_feat"]])
        label = torch.cat([cg_dict["label"], cg_dict["val_label"]])
        pred = np.concatenate((cg_dict["pred"], cg_dict["val_pred"]), axis=1)
    except:
        adj = cg_dict["adj"]
        feat = cg_dict["feat"]
        label = cg_dict["label"]
        pred = cg_dict["pred"]
    
    if 'emb' not in cg_dict:
        cg_dict['emb'] = None
    if prog_args.explainer_method == "gnnexplainer":
        explainer = explain_gnnexplainer.ExplainerGnnExplainer(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=False,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device,
        )
    elif prog_args.explainer_method == "random":
        explainer = explain_random.ExplainerRandom(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=False,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device,
        )
    elif prog_args.explainer_method == "attn":
        explainer = explain_attn.ExplainerAttn(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device,
        )
    elif prog_args.explainer_method == "grad":
        explainer = explain_grad.ExplainerGrad(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device,
        )
    elif prog_args.explainer_method == "boundary":
        explainer = explain_boundary_joint.ExplainerBoundaryJoint(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )
    elif prog_args.explainer_method == "boundary_inverse":
        explainer = explain_boundary_inverse.ExplainerBoundaryInverse(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx
        )    

    elif prog_args.explainer_method == "pgexplainer":
        explainer = explain_pgexplainer.ExplainerPGExplainer(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "twopgexplainer":
        explainer = explain_twopgexplainer.ExplainerTwoPGExplainer(
            model=model,
            adj=adj,
            feat=feat,
            label=label,
            pred=pred,
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "rcexplainer":
        explainer = explain_rcexplainer.ExplainerRCExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            emb=cg_dict["emb"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "rcadversarialexplainer":
        explainer = explain_rcadversarialexplainer.ExplainerRCExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            emb=cg_dict["emb"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "tworcexplainer":
        explainer = explain_tworcexplainer.ExplainerTwoRCExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            emb=cg_dict["emb"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "rcnoiseexplainer":
        explainer = explain_rcnoiseexplainer.ExplainerRCExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            emb=cg_dict["emb"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "rcexp_noldb":
        explainer = explain_rcexplainer_noldb.ExplainerRCExplainerNoLDB(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            emb=cg_dict["emb"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            num_nodes=dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )
    

    elif prog_args.explainer_method == "two_masks":
        explainer = explain_boundary_two_masks.ExplainerBoundaryTwoMasks(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            emb=cg_dict["emb"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "pgmexplainer":
        explainer = explain_pgmexplainer.ExplainerPgmExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )

    elif prog_args.explainer_method == "pgexplainer_node":
        explainer = explain_pgexplainer_node.ExplainerPGExplainer(
            model=model,
            adj=cg_dict["adj"],
            feat=cg_dict["feat"],
            label=cg_dict["label"],
            pred=cg_dict["pred"],
            train_idx=cg_dict["train_idx"],
            num_nodes = dict_num_nodes,
            args=prog_args,
            writer=writer,
            print_training=True,
            graph_mode=graph_mode,
            graph_idx=prog_args.graph_idx,
            device=device
        )


    if prog_args.bmname == 'PROTEINS':
        range_g = range(1044)
    elif prog_args.bmname == 'ER_MD':
        range_g = range(444)
    elif prog_args.bmname == 'REDDIT-BINARY':
        range_g = range(552)
    elif prog_args.bmname == 'MUTAG':
        # range_g = range(186)
        range_g = range(168) # train dataset
    elif prog_args.bmname == 'bbbp':
        range_g = range(1427)
        # range_g = [i for i, x in enumerate(label.tolist())]
    elif prog_args.bmname == 'Mutagenicity':
        range_g = range(3000)
    elif prog_args.bmname == "NCI1":
        range_g = range(2877)
        # range_g = random.sample(range_g, 100)
    elif prog_args.bmname == "BA_2Motifs":
        # range_g = range(len(label.tolist()))
        range_g = range(700)

    else:
        range_g = range(3000)	
    

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support 
    # We could even move each mode to a different method (even file)
    if prog_args.explain_node is not None:
        explainer.explain(prog_args.explain_node, unconstrained=False)
    elif graph_mode:
        if prog_args.multigraph_class >= 0:  #explain particular class
            # print(cg_dict["label"])
            # only run for graphs with label specified by multigraph_class
            labels = cg_dict["label"]
            preds = np.argmax(cg_dict['pred'][0,:,:], axis=1)
            graph_indices = []
            for i, l in enumerate(preds):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
                # if len(graph_indices) > 30:
                #     break

            print(
                "Graph indices for label ",
                prog_args.multigraph_class,
                " : ",
            )
            # orig_graph_indices=graph_indices
            # test_graph_indices = []
            random.shuffle(graph_indices)

            if prog_args.explainer_method == "pgexplainer_boundary":
                raise NotImplemented("Oepsie floepsie")
                # if prog_args.train_data_sparsity is not None:
                #     graph_indices = random.sample(graph_indices, int(len(graph_indices) * prog_args.train_data_sparsity))
                # explainer.explain_graphs(prog_args, graph_indices=graph_indices, test_graph_indices=orig_graph_indices)
            else:
                if prog_args.train_data_sparsity == 0.8:
                    N = int(len(graph_indices)*prog_args.train_data_sparsity)
                    train_graph_indices = graph_indices[:N]
                    test_graph_indices = graph_indices[N:]
                elif prog_args.train_data_sparsity == 1.0:
                    train_graph_indices = random.sample(graph_indices, int(len(graph_indices) * prog_args.train_data_sparsity))
                    test_graph_indices = train_graph_indices
                else:
                    raise NotImplementedError("Chose data sparsity 0.8 or "
                                              "1.0. Got", prog_args.train_data_sparsity)

                print("Train", len(train_graph_indices))
                print("Test", len(test_graph_indices))
                train, test, _, _ = explainer.explain_graphs(prog_args, graph_indices=train_graph_indices, test_graph_indices=test_graph_indices)


        elif prog_args.graph_idx == -1:
            raise NotImplemented("Oepsie 1")
            
            # orig_graph_indices=range_g
            #
            # if prog_args.train_data_sparsity is not None:
            #     range_g = random.sample(range_g, int(len(range_g) * prog_args.train_data_sparsity))
            # explainer.explain_graphs(prog_args, graph_indices=range_g, test_graph_indices=orig_graph_indices)
        else:
            raise NotImplemented("Oepsie 2")
            # explainer.explain(
            #     node_idx=0,
            #     graph_idx=prog_args.graph_idx,
            #     graph_mode=True,
            #     unconstrained=False,
            # )
            # io_utils.plot_cmap_tb(writer, "tab20", 20, "tab20_cmap")
    else:
        if prog_args.multinode_class >= 0:
            raise NotImplemented("Niet de bedoeling")
            print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            labels = cg_dict["label"][0]  # already numpy matrix

            node_indices = []
            for i, l in enumerate(labels):
                if len(node_indices) > 4:
                    break
                if l == prog_args.multinode_class:
                    node_indices.append(i)
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                node_indices,
            )
            explainer.explain_nodes(node_indices, prog_args)

        else:
            # explain a set of nodes
            # masked_adj = explainer.explain_nodes_gnn_stats(
            #     range(400, 700, 5), prog_args
            # )
            if prog_args.bmname == "syn1":
                # 0 - 699
                # 400 - 699
                full_node_list = list(range(0, 700, 1))
                node_list = list(range(400, 700, 1))
                if prog_args.train_data_sparsity is not None:
                    node_list = random.sample(node_list, int(len(node_list) * prog_args.train_data_sparsity))
                masked_adj = explainer.explain_nodes_gnn_stats(
                    node_list, full_node_list, prog_args
                )

            elif prog_args.bmname == "syn2":
                # 0 - 1399
                # 400-699, 1100-1399
                full_node_list = list(range(0, 1400, 1))
                node_list = list(range(400, 700, 1)) + list(range(1100, 1400, 1))

                masked_adj = explainer.explain_nodes_gnn_stats(
                    node_list, full_node_list, prog_args
                )

            elif prog_args.bmname == "syn3" or prog_args.bmname == "repeat_syn3":
                prog_args.bmname = "syn3"

                #0-1019
                #300-1019

                full_node_list = list(range(0, 1020, 1))
                node_list = list(range(300, 1020, 1))

                masked_adj = explainer.explain_nodes_gnn_stats(
                    node_list, full_node_list, prog_args
                )
            elif prog_args.bmname == "syn4" or prog_args.bmname == "dense_syn4":
                prog_args.bmname = "syn4"


                #0-870
                #511-870
                full_node_list = list(range(0, 871, 1))
                node_list = list(range(511, 871, 1))


                masked_adj = explainer.explain_nodes_gnn_stats(
                    node_list, full_node_list, prog_args
                )
            
            elif prog_args.bmname == "syn8":
                # 0 - 1399
                # 400-699, 1100-1399
                full_node_list = list(range(0, 370, 1))
                node_list = list(range(10, 370, 3))

                masked_adj = explainer.explain_nodes_gnn_stats(
                    node_list, full_node_list, prog_args
                )


            # masked_adj = explainer.explain_nodes_gnn_stats(
            #     range(0, 1020, 1), prog_args
            # )
            # masked_adj = explainer.explain_nodes_gnn_stats(
            #     range(400, 450, 1), prog_args
            # )
    return train, test


if __name__ == "__main__":
    print(main())

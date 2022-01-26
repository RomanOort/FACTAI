

import time
step_1 = time.time()

import sys
import pickle

# sys.path.insert(0, "/home/fisher/GCN-Group-interpretation/ai-adversarial-detection")

# # print(sys.path)
from dnn_invariant.utilities.environ import *
# from dnn_invariant.models.models4invariant import *
from dnn_invariant.models.models_gcn import *

import scipy.sparse as sp
import time
import dill
# from dnn_invariant.utilities.visualization import *
import cv2
import networkx as nx
import matplotlib.pyplot as plt

from dnn_invariant.algorithms.mine_gcn_invariant import *
from dnn_invariant.algorithms.gradcam import *
from dnn_invariant.algorithms.lime_image import *
import dnn_invariant.utilities.trainer_node as trainer_node
import dnn_invariant.utilities.datasets_node as datasets_node

import dnn_invariant.utilities.trainer_graph as trainer_graph
import dnn_invariant.utilities.datasets_graph as datasets_graph

import torch
import torch.nn as nn
from collections import Counter
import os
import shutil
import matplotlib.pyplot as plt

import random

import tqdm

BASE_DIRECTORY = "/home/fisher/GCN-Group-interpretation/ai-adversarial-detection/dnn_invariant"

def extract_rules(dataset_name, train_data, test_data, args,  model_state_dict=None, graph_indices=None, pool_size=50):
    from os import path
    pickle_path = "./data/rule_dict_MNISTSuperpixel.pickle"

    if path.exists(pickle_path):
        print("NOTE: Rules already extracted")
        print("Using file", pickle_path)
        rule_dict_save = pickle.load(open(pickle_path, 'rb'))
        return rule_dict_save
    else:
        print("No rule file found, extracting a new one.")

    #===================   Parameter settings Start   ========================

    """
    The training and testing data will be loaded using utilities.datasets.
    They are already normalized and are ready to feed into the model.
    To visualize the images, they need to be denormalized first. 
    """
    np.set_printoptions(threshold=np.inf, precision=20)
    np.random.seed(args.seed)
    torch.set_printoptions(precision=6)
    torch.manual_seed(args.seed)

    '''
    The layer to perform rule extraction
    '''
    # I = -1
    #I = 6
    # I = 34
    I = 36

    '''
    Number of rules for each class, and number of classes
    '''
    length = 2
    exper_visual = False # do not change this
    exper_robust_gaussian = False # do not change this
    exper_robust_model = False # do not change this

    # model = VGG19(num_classes_=num_classes).cuda()

    if dataset_name == 'syn1':
        class_list = ['none','bottom', 'middle','top']
        node_labels = ['A','A','A','A']

        num_classes = len(class_list)
        model = GcnEncoderNode(
            input_dim=len(node_labels),
            hidden_dim=20,
            embedding_dim=20,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=False,
            concat=False,
            args=None,
        ).cuda()

        # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
        dict_t = torch.load(BASE_DIRECTORY + "/mdls/syn1/4inp_20emb_1gc.pth.tar")
        is_graph_classification = False
    elif dataset_name == 'syn2':
        class_list = ['none','bottom', 'middle','top','none2','bottom2', 'middle2','top2']
        node_labels = ['A','A','A','A','A','A','A','A','A','A']
        num_classes = len(class_list)

        model = GcnEncoderNode(
            input_dim=10,
            hidden_dim=20,
            embedding_dim=20,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=False,
            concat=False,
            args=None,
        ).cuda()

        # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
        # dict_t = torch.load("./dnn_invariant/mdls/syn2/4inp_20emb_3gc.pth.tar")
        # dict_t = torch.load("./dnn_invariant/mdls/syn2/10inp_20emb_3gc_binary.pth.tar")
        dict_t = torch.load(BASE_DIRECTORY + "/mdls/syn2/10inp_20emb_3gc_housemod.pth.tar")

        is_graph_classification = False

    elif dataset_name == 'syn3':
        class_list = ['tree', 'grid']
        node_labels = ['A','A','A','A','A','A','A','A','A','A']
        num_classes = len(class_list)

        model = GcnEncoderNode(
            input_dim=10,
            hidden_dim=20,
            embedding_dim=20,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=False,
            concat=False,
            args=None,
        ).cuda()

        # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
        dict_t = torch.load(BASE_DIRECTORY + "/mdls/syn3/10inp_20emb_3gc_repeat.pth.tar")
        # dict_t = torch.load("./dnn_invariant/mdls/syn3/10inp_20emb_3gc.pth.tar")
        
        is_graph_classification = False

    elif dataset_name == 'syn4':
        class_list = ['tree', 'cycle']
        node_labels = ['A','A','A','A','A','A','A','A','A','A']
        num_classes = len(class_list)

        model = GcnEncoderNode(
            input_dim=10,
            hidden_dim=20,
            embedding_dim=20,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=False,
            concat=False,
            args=None,
        ).cuda()

        # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
        # dict_t = torch.load("./dnn_invariant/mdls/syn4/10inp_20emb_3gc.pth.tar")
        dict_t = torch.load(BASE_DIRECTORY + "/mdls/syn4/10inp_20emb_3gc_dense.pth.tar")
        is_graph_classification = False

    elif dataset_name == 'syn8':
        class_list = ['NONE', 'AB', 'BC', 'CD']
        node_labels = ['A','A','A','A','A']
        num_classes = len(class_list)

        model = GcnEncoderNode(
            input_dim=5,
            hidden_dim=20,
            embedding_dim=20,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=False,
            concat=False,
            args=None,
        ).cuda()

        # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
        # dict_t = torch.load("./dnn_invariant/mdls/syn4/10inp_20emb_3gc.pth.tar")
        dict_t = torch.load(BASE_DIRECTORY + "/mdls/syn8/syn8_base_h20_o20.pth.tar")
        is_graph_classification = False

    if dataset_name == 'synthetic':

        model = GcnEncoderGraph(
            12,# 6 def is input_dim,
            20,#20,#args.hidden_dim,
            20,#20,#args.output_dim,
            3,#args.num_classes,
            args.num_gc_layers,#3,#args.num_gc_layers,
            pred_hidden_dims = [],
            bn=False,
            dropout=0.0,
            args=None,
            add_self= (args.add_self == "none")
        ).cuda()
        num_classes = 3


        dict_t = torch.load(BASE_DIRECTORY + "/mdls/synthetic_norep/synthetic_3gc_1fc_20dim_4k8000_comb_12dlbls_nofake_1gc.pth.tar")

        is_graph_classification = True

    if dataset_name == 'old_synthetic':

        model = GcnEncoderGraph(
            6,# 6 def is input_dim,
            20,#20,#args.hidden_dim,
            20,#20,#args.output_dim,
            3,#args.num_classes,
            args.num_gc_layers,#3,#args.num_gc_layers,
            pred_hidden_dims = [],
            bn=False,
            dropout=0.0,
            args=None,
            add_self= (args.add_self == "none")
        ).cuda()
        num_classes = 3



        is_graph_classification = True



    elif dataset_name == 'Mutagenicity':
        class_list = ['mutagenic', 'non-mutagenic']
        num_classes = len(class_list)
        model = GcnEncoderGraph(
            14,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True



    elif dataset_name == 'PROTEINS':
        num_classes = 2
        model = GcnEncoderGraph(
            3,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'REDDIT-BINARY':
        num_classes = 2
        model = GcnEncoderGraph(
            10,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'ER_MD':
        num_classes = 2
        model = GcnEncoderGraph(
            10,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'COLLAB':
        num_classes = 3
        model = GcnEncoderGraph(
            10,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'NCI1':
        num_classes = 2
        model = GcnEncoderGraph(
            37,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'MUTAG':
        num_classes = 2
        model = GcnEncoderGraph(
            7,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True

    elif dataset_name == 'bbbp':
        num_classes = 2
        model = GcnEncoderGraph(
            9,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True
    elif dataset_name == 'BA_2Motifs':
        num_classes = 2
        model = GcnEncoderGraph(
            10,  # input_dim,
            20,  # args.hidden_dim,
            20,  # args.output_dim,
            num_classes,  # args.num_classes,
            args.num_gc_layers,  # args.num_gc_layers,
            pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        #dict_t = torch.load(BASE_DIRECTORY + "/dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")
        is_graph_classification = True
    elif dataset_name == "MNIST":
        num_classes = 10
        model = GcnEncoderGraph(
            1, # input_dim?????
            20,
            20,
            num_classes,
            args.num_gc_layers,
            pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        is_graph_classification = True
    elif dataset_name == "MNISTSuperpixels":
        num_classes = 10
        model = GcnEncoderGraph(
            input_dim=1,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            # pred_hidden_dims=[args.pred_hidden_dim] * args.pred_num_layers,
            # Hardcode to match train.py
            pred_hidden_dims=[20, 10],
            bn=False,
            dropout=0.0,
            args=None,
            add_self = (args.add_self == "none")
        ).cuda()

        dict_t = None
        is_graph_classification = True

    if is_graph_classification:
        mDataSet = datasets_graph.mDataSet
        Trainer = trainer_graph.Trainer
    else:
        mDataSet = datasets_node.mDataSet
        Trainer = trainer_node.Trainer

    train_data = mDataSet(train_data)
    test_data = mDataSet(test_data)

    #model = CNN4MNIST(num_classes_=num_classes).cuda()

    model_name = 'gcn.mdl'
    #model_name = 'ZhangLab_3epochs.mdl'
    #model_name = 'MNIST_24.mdl'
    #model_name = 'FMNIST_24.mdl'

    top_similar = 5

    # # print('Train Data Size: ', train_data._data[0].shape)

    # model.loadModel(model._model_rootpath + model_name)

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(dict_t["model_state"])
    # print(model)
    is_set_cover    = False

    # Experiment Settings

    if os.path.exists('node_logs') and os.path.isdir('node_logs'):
        shutil.rmtree('node_logs') # remove old logs

    os.mkdir('node_logs')


    #===================   Parameter settings End   ==========================

    #===================   Define Utility Functions Start  ===================

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

        return top4_acc/hnode_count, top6_acc/hnode_count, top8_acc/ hnode_count


    # evaluate the accuracy of a single rule on mdata
    def evalSingleRule(inv_classifier, mdata):
        # Evaluate performance of each boundary
        # inv_classifier.classify_one_boundary(mdata._getArrayFeatures(), mdata._getArrayLabels())
        pred_labels, cover_indi   = inv_classifier.classify(mdata._getArrayFeatures())
        label_diff                = pred_labels[cover_indi] - mdata._getArrayLabels()[cover_indi]
        accuracy                  = np.sum(label_diff == 0)/label_diff.size

        '''
        # Count size of each class covered by the rule
        for i in range(len(class_list)):
            # print(class_list[i], np.count_nonzero(mdata._getArrayLabels()[cover_indi] == i))
        '''
        return accuracy, cover_indi

    def evalSingleRuleByNN(inv_classifier, mdata):
        tnsr_label = []
        for i in range(mdata.__len__()):
            input = mdata.getCNNFeature(i).cuda()
            output = model._getOutputOfOneLayer(input).cpu().detach().numpy()
            tnsr_label.append(np.argmax(output[0]))

        tnsr_label = np.asarray(tnsr_label)

        pred_labels, cover_indi   = inv_classifier.classify(mdata._getArrayFeatures())
        label_diff                = pred_labels[cover_indi] - tnsr_label[cover_indi]
        accuracy                  = np.sum(label_diff == 0)/label_diff.size
        return accuracy, cover_indi

    # evaluate the accuracy of all selected rules on mdata
    def evalAllRules(rule_list, cover_list, mdata, is_set_cover = False):
        # select rules by set cover if allowed
        if is_set_cover:
            rule_sel_indi = setCover(cover_list)
            # print('Set cover ==> num selected rules: %d/%d' % (np.sum(rule_sel_indi), rule_list.__len__()))
        else:
            rule_sel_indi = np.ones(rule_list.__len__(), dtype='bool')

        # evaluate the accuracy of the selected set of rules
        pred_votes = {}
        tnsr_feat = mdata._getArrayFeatures()
        tnsr_label = mdata._getArrayLabels()

        for c_idx in np.where(rule_sel_indi)[0]:
            pred_labels, cover_indi = rule_list[c_idx].classify(tnsr_feat)
            for idx in np.where(cover_indi)[0]:
                if idx in pred_votes:
                    pred_votes[idx].append(pred_labels[idx])
                else:
                    pred_votes[idx] = [pred_labels[idx]]

        num_correct = 0
        num_covered = 0
        num_total = 0
        for i in range(tnsr_label.shape[0]):
            num_total += 1
            if i in pred_votes:
                num_covered += 1
                pred_label = sorted([(np.sum(pred_votes[i] == pred), pred) for pred in set(pred_votes[i])])[-1][-1]
                gt_label = tnsr_label[i]
                if pred_label == gt_label:
                    num_correct += 1

        if num_covered > 0:
            accuracy = num_correct / num_covered
        else:
            accuracy = -1

        return accuracy, num_covered

    def printEvalInfo(rule_list, cover_list, is_set_cover = False):
        accuracy_test, num_covered_test           = evalAllRules(rule_list, cover_list, test_data_upt, is_set_cover)
        #accuracy_valid, num_covered_valid = evalAllRules(rule_list, cover_list, valid_data, is_set_cover)
        accuracy_train_upt, num_covered_train_upt = evalAllRules(rule_list, cover_list, train_data_upt, is_set_cover)
        #accuracy_train_org, num_covered_train_org = evalAllRules(rule_list, cover_list, train_data, is_set_cover)

        # print('num rules: %d' % (rule_list.__len__()))
        # print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
        # print('Org test accuracy: %.5f' % (org_test_acu))
        # print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))


    def evalAllRulesByNN(rule_list, cover_list, mdata, is_set_cover = False):
        # select rules by set cover if allowed
        if is_set_cover:
            rule_sel_indi = setCover(cover_list)
            # print('Set cover ==> num selected rules: %d/%d' % (np.sum(rule_sel_indi), rule_list.__len__()))
        else:
            rule_sel_indi = np.ones(rule_list.__len__(), dtype='bool')

        # evaluate the accuracy of the selected set of rules
        pred_votes = {}
        tnsr_feat = mdata._getArrayFeatures()
        tnsr_label = []
        for i in range(mdata.__len__()):
            input = mdata.getCNNFeature(i).cuda()
            output = model._getOutputOfOneLayer(input).cpu().detach().numpy()
            #tnsr_label.append(np.amax(output[0]).item())
            tnsr_label.append(np.argmax(output[0]))

        for c_idx in np.where(rule_sel_indi)[0]:
            pred_labels, cover_indi = rule_list[c_idx].classify(tnsr_feat)
            for idx in np.where(cover_indi)[0]:
                if idx in pred_votes:
                    pred_votes[idx].append(pred_labels[idx])
                else:
                    pred_votes[idx] = [pred_labels[idx]]

        num_correct = 0
        num_covered = 0
        num_total = 0
        for i in range(len(tnsr_label)):
            num_total += 1
            if i in pred_votes:
                num_covered += 1
                pred_label = sorted([(np.sum(pred_votes[i] == pred), pred) for pred in set(pred_votes[i])])[-1][-1]
                gt_label = tnsr_label[i]
                if pred_label == gt_label:
                    num_correct += 1

        if num_covered > 0:
            accuracy = num_correct / num_covered
        else:
            accuracy = -1

        return accuracy, num_covered


    def printEvalInfoByNN(rule_list, cover_list, is_set_cover = False):
        accuracy_test, num_covered_test           = evalAllRulesByNN(rule_list, cover_list, test_data_upt, is_set_cover)
        #accuracy_valid, num_covered_valid = evalAllRulesByNN(rule_list, cover_list, valid_data, is_set_cover)
        accuracy_train_upt, num_covered_train_upt = evalAllRulesByNN(rule_list, cover_list, train_data_upt, is_set_cover)
        #accuracy_train_org, num_covered_train_org = evalAllRulesByNN(rule_list, cover_list, train_data, is_set_cover)

        # print('num rules: %d' % (rule_list.__len__()))
        # print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
        # print('Org test accuracy: %.5f' % (org_test_acu))
        '''
        # print('valid_accuracy_org: %.5f\tnum_covered: %d/%d' % (
            accuracy_valid, num_covered_valid, valid_data.__len__()))
        # print('Org valid accuracy: %.5f' % (org_valid_acu))
        '''
        # print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (
        # accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))
        ## print('train_accuracy_org: %.5f\tnum_covered_train_org: %d/%d' % (
        #accuracy_train_org, num_covered_train_org, train_data_upt.__len__()))

    # select a set of rules from rule_list and return the indi of selection
    def setCover(cover_list):
        # start set cover selection of rules
        cover_matrix = np.array(cover_list).T
        covered_indi = np.zeros(cover_matrix.shape[0], dtype='bool')
        sel_indi = np.zeros(cover_matrix.shape[1], dtype='bool')

        iter = 0
        while True:
            iter += 1
            # greedy find the best rule
            row_sum = np.sum(cover_matrix, axis=0)

            max_idx = np.argmax(row_sum)

            # if iter % 500 == 0:
            # print('iter: %d\tmax_val: %d\ttotal_covered: %d/%d\n' % (
            # iter, row_sum[max_idx], np.sum(covered_indi), covered_indi.size))

            if row_sum[max_idx] <= 0:
                break

            # update sel_indi and covered_indi
            sel_indi[max_idx] = True
            covered_indi = (covered_indi | cover_matrix[:, max_idx])

            # remove selected samples
            cover_matrix[covered_indi, :] = False

        return sel_indi

    def NNpic(model, img_seed, top=top_similar):
        weighted_distances = np.zeros(train_data_upt.__len__())
        rules = model._bb.bb[:, model._invariant].copy()
        for t in range(model.getNumBoundaries()):
            rule = rules[:, t]
            rule_no_bias = rule[:-1]

            proj = np.dot(img_seed, rule_no_bias)

            for j in range(train_data_upt.__len__()):
                if cover_indi_train[j] == True:
                    img = train_data_upt.getCNNFeature(j).numpy()
                    img = img.flatten()
                    distance = abs(np.dot(img, rule_no_bias) - proj)
                    weighted_distances[j] += distance
                else:
                    weighted_distances[j] += 10000

        #return weighted_distances

        weighted_distances_order = np.argsort(weighted_distances)
        return weighted_distances_order[:top]


    #===================   Define Utility Functions End  =====================

    # substitute the labels of training data
    train_data_upt = train_data.loadHardCopy()
    test_data_upt = test_data.loadHardCopy()

    # check training and testing accuracy
    org_train_acu, pred_labels, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
    org_test_acu, pred_labels_test, _       = Trainer.evalAccuracyOfModel(model, test_data)
    # print('org_train_acu: %.5f' % (org_train_acu))
    # print('org_test_acu: %.5f' % (org_test_acu))

    # print('Original Training Data Shape:', train_data_upt._data[0].shape)

    train_data_upt.updateLabels(model)
    train_data_upt.updateData(model, I)
    test_data_upt.updateData(model, I)
    # train_data_upt.validateData(model)

    # print('Modified Training Data Shape:', train_data_upt._data[0].shape)

    step_2 = time.time()
    # print('Time:', step_2 - step_1)

    if exper_robust_gaussian:
        train_data_gaussian = []
        train_data_upt_gaussian = []
        for turn in range(3):
            train_data_gaussian0 = train_data.loadHardCopy()
            train_data_gaussian0.addGaussianNoise(0.3 + 0.2 * turn)

            train_data_upt_gaussian0 = train_data_gaussian0.loadHardCopy()
            train_data_upt_gaussian0.updateLabels(model)
            train_data_upt_gaussian0.updateData(model, I)

            train_data_gaussian.append(train_data_gaussian0)
            train_data_upt_gaussian.append(train_data_upt_gaussian0)
            # print('Adding Gaussian Noise with Standard Derivation ', 0.3 + 0.2 * turn, '...Done')

    if exper_robust_model:
        model_list = []
        train_data_upt_model = []

        for epoch in range(5):
            model_temp = VGG19Assira(num_classes_=2).cuda()
            model_temp.loadModel(model_temp._model_rootpath + model_temp.__class__.__name__ + str(epoch) + '.mdl')
            model_list.append(model_temp)

            org_train_acu, pred_labels, pred_probs = Trainer.evalAccuracyOfModel(model_temp, train_data)
            org_test_acu, pred_labels_test, _      = Trainer.evalAccuracyOfModel(model_temp, test_data)
            # print('Model ' + str(epoch))
            # print('org_train_acu: %.5f' % (org_train_acu))
            # print('org_test_acu: %.5f' % (org_test_acu))

            train_data_upt_model_temp = train_data.loadHardCopy()
            train_data_upt_model_temp.updateLabels(model_temp)
            train_data_upt_model_temp.updateData(model_temp, I)
            train_data_upt_model.append(train_data_upt_model_temp)
            # print('Updating Data using Model ' + str(epoch) + '...Done')



    # show statics
    train_labels = train_data._getTensorLabels().numpy()
    test_labels  = test_data._getTensorLabels().numpy()
    updated_train_labels = train_data_upt._getTensorLabels().numpy()
    # print('Training labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
    # print('Updated Training labels: ', [(lb, np.sum(updated_train_labels == lb)) for lb in set(updated_train_labels)], '\n')
    # print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')

    # torch.save(train_data_upt._data, "./temp_train_data_upt_mutag.pth")
    # exit()

    iter = 0
    np.random.seed(args.seed)
    # print("train_data_upt: len: ", train_data_upt.__len__())
    # np.random.seed()

    if graph_indices is None:
        indices = np.random.permutation(train_data_upt.__len__())[0:2000]
    else:
        indices = np.random.permutation(graph_indices)
    
    check_repeat = np.repeat(False, train_data_upt.__len__())
    check_repeat_test = np.repeat(False, test_data.__len__())

    # for i in range(len(model._layers_list)):
    #     model._layers_list[i].requires_grad_ = True

    for l, p in model.named_children():
        # print(l)
        p.requires_grad_ = True
    pivots_list = []
    opposite_list = []
    rule_list_all = []
    cover_list_all = []
    check_repeat = np.repeat(False, train_data_upt.__len__())
    check_repeat_test = np.repeat(False, test_data.__len__())
    check_balance = []
    rule_dict_list = []
    idx2rule = {}
    # print("indices: ", len(indices))

    _train_labels = train_data_upt._getArrayLabels()
    # print("labels: ", _train_labels.shape, pred_labels.shape, np.sum(_train_labels==pred_labels),  np.sum(_train_labels==0))


    # print("total seed images: ", len(indices))

    # all_data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_2label_3sublabel/synthetic_data.p", "rb"))
    # sublabel_nodes_array = all_data['sub_label_nodes']
    # sublabel_array = all_data['sub_label']

    
    seed_indices = np.random.choice(indices, size=pool_size)
    
    for i in range(len(indices)):
        idx2rule[i] = None
    
    for i in tqdm.tqdm(range(len(indices))):
        rule_label = train_data_upt._data[2][i]

        step_3 = time.time()

        # if iter > num_classes * length - 1:
        #     break

        if iter > num_classes * length - 1:
            if idx2rule[i] != None:
                continue

        # print(i)
        # print(np.sum(check_repeat))
        # print(len(train_data))

        if np.sum(check_repeat) >= len(train_data):
            break


        # if check_balance.count(pred_labels[i]) > length - 1:
        #     continue
        # else:
        #     check_balance.append(pred_labels[i])

        # print('========================= Next Rule in Progress ==========================')

        # print("Seed id : ", i)
        # print("Seed label: ", train_data_upt._data[2][i])
        # print("Orig label: ", train_data._data[2][i])

        # Initialize the rule. Need to input the label for the rule and which layer we are using (I)
        rule_miner_train = RuleMinerLargeCandiPool(model, train_data_upt, pred_labels[i], I)

        # Compute the feature for the seed image
        feature = train_data_upt.getCNNFeature(i).cuda()
        # # print("feature:", feature.shape)
        # label_update = model._getOutputOfOneLayer(feature)
        # # print("label update: ", label_update)
        # exit()


        # Create candidate pool
        rule_miner_train.CandidatePoolLabel(feature, pool=100)
        # print('Candidate Pool Created')

        # Perform rule extraction
        # initial is not in use currently
        inv_classifier, pivots, opposite, initial = rule_miner_train.getInvariantClassifier(i, feature.cpu().numpy(), pred_labels[i], train_data_upt._getArrayLabels(), delta_constr_=0)
        # print('Rule Extraction Completed\n')

        # # print(pivots)
        # true = 0
        # false = 0
        # from sklearn.decomposition import PCA
        # import matplotlib.pyplot as plt
        # X = np.zeros((len(indices), 20))
        # for cnt, j in enumerate(indices):
        #     test_feature = train_data_upt.getCNNFeature(j)
        #     X[cnt,:] = test_feature.numpy()
        #     pivots_n = pivots[0][0].cpu()
        #     test_feature = test_feature[0].cpu()
        #     label = train_data_upt._data[2][j]
        #     basis = rule_miner_train._bb.bb[:20, 0]
        #     basis = torch.tensor(basis)
        #     bias = rule_miner_train._bb.bb[20, 0]
        #     bias = torch.tensor(bias)

        #     true += (torch.dot(basis, test_feature.T) + bias).sign() == label
        #     false +=  (torch.dot(basis, test_feature.T) + bias).sign() != label
        #     # # print(torch.dot(basis, test_feature.T) + bias, label)
        # pca = PCA(n_components=2).fit(X)
        # X_pca = pca.transform(X)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1])
        # plt.savefig('pca' + str(i) + '.png')
        # # print(true, false)
        pivots_list.append(pivots)
        opposite_list.append(opposite)


        #saving info for gnnexplainer
        rule_dict = {}
        inv_bbs = inv_classifier._bb.bb

        inv_invariant = inv_classifier._invariant
        boundaries_info = []
        b_count = 0
        assert(len(opposite) == np.sum(inv_invariant))
        for inv_ix in range(inv_invariant.shape[0]):
            if inv_invariant[inv_ix] == False:
                continue
            boundary_dict = {}
            # boundary_dict['basis'] = inv_bbs[:-1,inv_ix]
            boundary_dict['basis'] = inv_bbs[:,inv_ix]
            boundary_dict['label'] = opposite[b_count]
            b_count += 1
            boundaries_info.append(boundary_dict)
        rule_dict['boundary'] = boundaries_info
        rule_dict['label'] = rule_label.cpu().item()
        #print("Rules extracted: ", rule_dict)
        rule_dict_list.append(rule_dict)
        #end saving info for gnn-explainer



        # evaluate classifier
        accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data_upt)
        accuracy_test, cover_indi_test   = evalSingleRule(inv_classifier, test_data_upt)
        accuracy_train_NN, _ = evalSingleRuleByNN(inv_classifier, train_data_upt)
        accuracy_test_NN, _ = evalSingleRuleByNN(inv_classifier, test_data_upt)

        assert(cover_indi_train[i] == True)
        for c_ix in range(cover_indi_train.shape[0]):
            if cover_indi_train[c_ix] == True:
                if is_graph_classification:
                    idx2rule[c_ix] = len(rule_list_all)
                else:
                    if c_ix not in idx2rule:
                        idx2rule[c_ix] = []
                    idx2rule[c_ix].append(len(rule_list_all))

    
        rule_list_all.append(inv_classifier)
        cover_list_all.append(cover_indi_train)

        for j in range(train_data_upt.__len__()):
            if cover_indi_train[j] == True:
                check_repeat[j] = True

        for k in range(test_data.__len__()):
            if cover_indi_test[k] == True:
                check_repeat_test[k] = True

        # show info
        # print('========================= done %d/%d tgt_label: %d ==========================' %
        #    (iter, len(indices), inv_classifier._label))
        iter += 1

        # print('Seed point ID: %d\n' % (i))
        # print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))
        # print('Training ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train, np.sum(cover_indi_train)))
        # print('Testing ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test, np.sum(cover_indi_test)))
        # print('Training (Model Decision) ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train_NN, np.sum(cover_indi_train)))
        # print('Testing (Model Decision) ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test_NN, np.sum(cover_indi_test)))
        # print('Covered training data: %d\n' % np.sum(check_repeat))
        # print('Covered testing data: %d\n' % np.sum(check_repeat_test))

        step_4 = time.time()
        # print('Time for this rule:', step_4 - step_3)


    # print('RULE_LIST_ALL => ')
    printEvalInfo(rule_list_all, cover_list_all, is_set_cover)
    printEvalInfoByNN(rule_list_all, cover_list_all, is_set_cover)
    # # print(np.sum(np.array(cover_list_all[0])))
    # # print(len(rule_list_all[0]))


    rule_dict_save = {}
    rule_dict_save['rules'] = rule_dict_list
    rule_dict_save['idx2rule'] = idx2rule
    # pickle.dump(rule_dict_save, open("./data/synthetic/rule_dict_synthetic_train_4k8000_comb_12dlbls_nofake.p","wb"))

    if dataset_name == "MNISTSuperpixels":
        try:
            os.mkdir("data")
        except FileExistsError as e:
            pass
        print("Stored rules dict to:", pickle_path)
        pickle.dump(rule_dict_save, open(pickle_path,"wb"))
    return rule_dict_save

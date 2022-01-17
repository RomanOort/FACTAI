import time
step_1 = time.time()

import sys
import pickle

sys.path.insert(0, "/home/mohit/Mohit/model_interpretation/ai-adversarial-detection")
# print(sys.path)
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
from dnn_invariant.utilities.trainer_node import *
from dnn_invariant.utilities.datasets_node import *

import torch
import torch.nn as nn
from collections import Counter
import os
import shutil
import matplotlib.pyplot as plt

#===================   Parameter settings Start   ========================

"""
The training and testing data will be loaded using utilities.datasets.
They are already normalized and are ready to feed into the model.
To visualize the images, they need to be denormalized first. 
"""

np.set_printoptions(threshold=np.inf, precision=20)
np.random.seed(0)
torch.set_printoptions(precision=6)
torch.manual_seed(0)




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
num_classes = len(class_list)
exper_visual = False # do not change this
exper_robust_gaussian = False # do not change this
exper_robust_model = False # do not change this



# model = VGG19(num_classes_=num_classes).cuda()

if dataset_name == 'syn1':

    model = GcnEncoderNode(
        input_dim=len(node_labels),
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        concat=False,
        args=None,
    ).cuda()

    # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
    dict_t = torch.load("./dnn_invariant/mdls/syn1/4inp_20emb_1gc.pth.tar")

elif dataset_name == 'syn2':

    model = GcnEncoderNode(
        input_dim=10,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        concat=False,
        args=None,
    ).cuda()

    # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
    # dict_t = torch.load("./dnn_invariant/mdls/syn2/4inp_20emb_3gc.pth.tar")
    # dict_t = torch.load("./dnn_invariant/mdls/syn2/10inp_20emb_3gc_binary.pth.tar")
    dict_t = torch.load("./dnn_invariant/mdls/syn2/10inp_20emb_3gc_housemod.pth.tar")


elif dataset_name == 'syn3':

    model = GcnEncoderNode(
        input_dim=10,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        concat=False,
        args=None,
    ).cuda()

    # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
    dict_t = torch.load("./dnn_invariant/mdls/syn3/10inp_20emb_3gc_repeat.pth.tar")
    # dict_t = torch.load("./dnn_invariant/mdls/syn3/10inp_20emb_3gc.pth.tar")

elif dataset_name == 'syn4':

    model = GcnEncoderNode(
        input_dim=10,
        hidden_dim=20,
        embedding_dim=20,
        label_dim=num_classes,
        num_layers=3,
        bn=False,
        concat=False,
        args=None,
    ).cuda()

    # dict_t = torch.load("./dnn_invariant/mdls/synthetic_base_h20_o20.pth.tar") #simple one with 3gc and 1 fc
    # dict_t = torch.load("./dnn_invariant/mdls/syn4/10inp_20emb_3gc.pth.tar")
    dict_t = torch.load("./dnn_invariant/mdls/syn4/10inp_20emb_3gc_dense.pth.tar")

elif dataset_name == 'Mutagenicity':
    model = GcnEncoderGraph(
        14,  # input_dim,
        20,  # args.hidden_dim,
        20,  # args.output_dim,
        len(class_list),  # args.num_classes,
        3,  # args.num_gc_layers,
        bn=False,
        dropout=0.0,
        args=None,
    ).cuda()

    dict_t = torch.load("./dnn_invariant/mdls/Mutagenicity_base_h20_o20.pth.tar")

#model = CNN4MNIST(num_classes_=num_classes).cuda()

model_name = 'gcn.mdl'
#model_name = 'ZhangLab_3epochs.mdl'
#model_name = 'MNIST_24.mdl'
#model_name = 'FMNIST_24.mdl'

top_similar = 5









print('Train Data Size: ', train_data._data[0].shape)

# model.loadModel(model._model_rootpath + model_name)


model.load_state_dict(dict_t["model_state"])
print(model)
is_set_cover    = False

# Experiment Settings

if os.path.exists('node_logs') and os.path.isdir('node_logs'):
    shutil.rmtree('node_logs') # remove old logs

os.mkdir('node_logs')
path = '/home/mohit/Mohit/model_interpretation/ai-adversarial-detection/dnn_invariant'





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
        print(class_list[i], np.count_nonzero(mdata._getArrayLabels()[cover_indi] == i))
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
        print('Set cover ==> num selected rules: %d/%d' % (np.sum(rule_sel_indi), rule_list.__len__()))
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

    print('num rules: %d' % (rule_list.__len__()))
    print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
    print('Org test accuracy: %.5f' % (org_test_acu))
    print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))


def evalAllRulesByNN(rule_list, cover_list, mdata, is_set_cover = False):
    # select rules by set cover if allowed
    if is_set_cover:
        rule_sel_indi = setCover(cover_list)
        print('Set cover ==> num selected rules: %d/%d' % (np.sum(rule_sel_indi), rule_list.__len__()))
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

    print('num rules: %d' % (rule_list.__len__()))
    print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
    print('Org test accuracy: %.5f' % (org_test_acu))
    '''
    print('valid_accuracy_org: %.5f\tnum_covered: %d/%d' % (
        accuracy_valid, num_covered_valid, valid_data.__len__()))
    print('Org valid accuracy: %.5f' % (org_valid_acu))
    '''
    print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (
    accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))
    #print('train_accuracy_org: %.5f\tnum_covered_train_org: %d/%d' % (
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

        if iter % 500 == 0:
            print('iter: %d\tmax_val: %d\ttotal_covered: %d/%d\n' % (
            iter, row_sum[max_idx], np.sum(covered_indi), covered_indi.size))

        if row_sum[max_idx] <= 0:
            break

        # update sel_indi and covered_indi
        sel_indi[max_idx] = True
        covered_indi = (covered_indi | cover_matrix[:, max_idx])

        # remove selected samples
        cover_matrix[covered_indi, :] = False

    return sel_indi

'''
def get_image(idx, data):
    img = data.getCNNFeature(idx).numpy().copy()
    img = np.squeeze(img, axis=0)
    img = process_img(img)
    return img


def save_numpy_img(fname, img):
    img = np.squeeze(img)

    if img.ndim == 2:
        img = gray2rgb(img)

    if len(img[:, 0, 0]) == 1 or len(img[:, 0, 0]) == 3:
        img = np.moveaxis(img, 0, -1)

    img = cv2.resize(img, (output_size, output_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.amax(img) > 10:
        plt.imsave(fname, img / 255)
    else:
        plt.imsave(fname, img)
    #cv2.imwrite(fname, np.uint8(255 * img))
'''

def savefig(idx, name, data):
    img = data.getCNNFeature(idx).numpy().copy() # Get a 1x3xMxN tensor for RGB image, or 1xMxN for BW image
    img = np.squeeze(img, axis=0) # 3xMxN or MxN now
    img = process_img(img, ImageNetNormalize) # 3xMxN or 1xMxN now, without ImageNet normalization
    img = np.squeeze(img)

    # MxN becomes MxNx3
    if img.ndim == 2:
        img = gray2rgb(img)

    if len(img[:, 0, 0]) == 1 or len(img[:, 0, 0]) == 3:
        img = np.moveaxis(img, 0, -1)

    img = cv2.resize(img, (output_size, output_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.amax(img) > 100:
        plt.imsave(name + '_' + str(idx) + '.jpg', img / 255)
    else:
        plt.imsave(name + '_' + str(idx) + '.jpg', img)

def savefig_Rule_Gradcam(idx, name, data, model):
    img = data.getCNNFeature(idx).cuda()
    mask, cam = Rule_GradCam(model, img)
    cv2.imwrite(name + '_' + str(idx) + '_gradcam.jpg', np.uint8(255 * cam))

def savefig_Rule_LIME(idx, name, data, model):
    img = data.getCNNFeature(idx).cuda()
    mask, cam = Rule_LIME(model, img)
    cv2.imwrite(name + '_' + str(idx) + '_lime.jpg', np.uint8(255 * cam))

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


def saveAndDrawGraph2(mask, rule, db, close_id, sublabel_nodes=None, dbl = None):
    fig_size = (4,3)
    dpi = 300

    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    num_nodes = train_data._data[3][close_id]

    # num_nodes = train_data._data[3][close_id].numpy()
    # print("num nodes: ",num_nodes)
    weight_nodes = mask[:num_nodes]
    adj = (train_data._data[0][close_id].numpy())[:num_nodes,:num_nodes]
    feats = train_data._data[1][close_id].numpy()[:num_nodes,:]

    g_label = train_data._data[2][close_id].numpy()
    pred = pred_labels[close_id]
    G = nx.from_numpy_array(adj)
    pos_layout = nx.kamada_kawai_layout(G, weight=None)

    labels_dict = {}
    for n in range(num_nodes):
        node_l = np.argmax(feats[n])
        labels_dict[n] = node_labels[node_l]
    # fig, ax_l = plt.subplots(2, 1, figsize=(15, 20))
    node_colors = []
    color_list = [(0.9, 0.9, 0.9), (0.9, 0.7, 0.7), (0.9, 0.4, 0.4), (0.9, 0.1, 0.1)]

    if sublabel_nodes is None:
        weights = np.zeros_like(weight_nodes,dtype=np.int32)
        # max_weight = np.max(weight_nodes)
        topk_nodes = weight_nodes.argsort()[-6:][::-1]

        weights[topk_nodes[0]] = 3
        for i in range(1,4):
            weights[topk_nodes[i]] = 2
        for i in range(4,6):
            weights[topk_nodes[i]] = 1
        # weights[weight_nodes > (0.25*max_weight)] = 1
        # weights[weight_nodes > (0.50*max_weight)] = 2
        # weights[weight_nodes > (0.75*max_weight)] = 3
        for i in range(weights.shape[0]):
            node_colors.append(color_list[weights[i]])

    else:
        for i in range(num_nodes):
            if i not in sublabel_nodes:
                node_colors.append((0.9,0.9,0.9))
            elif i in sublabel_nodes[:4]:
                node_colors.append((0.1,0.5,0.1))
            else:
                node_colors.append((0.1, 0.9, 0.1))

    # labels = labels_dict
    nx.draw_networkx(G, pos=pos_layout, font_size=12,
                     node_size=150, labels = labels_dict, node_color=node_colors, cmap=cmap,vmax=8,vmin=0, alpha=0.8)
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()
    # nx.draw_networkx(G,  node_color = node_colors, ax=ax_l[1])
    if sublabel_nodes is None:
        filename = "./node_logs/visual/Rule_{}/{}_db{}_dbl{}_label{}_pred{}.pdf".format(rule, close_id,db, dbl, g_label,pred)
    else:
        filename = "./node_logs/visual/Rule_{}/{}_db{}_dbl{}_label{}_pred{}_gt.pdf".format(rule, close_id,db, dbl, g_label,pred)
    plt.savefig(filename)



def saveAndDrawGraph(mask, rule, db, close_id):
    num_nodes = train_data._data[3][close_id].numpy()
    print("num nodes: ",num_nodes)
    weight_nodes = mask[:num_nodes]
    adj = (train_data._data[0][close_id].numpy())[:num_nodes,:num_nodes]
    feats = train_data._data[1][close_id].numpy()[:num_nodes,:]
    g_label = train_data._data[2][close_id].numpy()
    pred = pred_labels[close_id]
    G = nx.from_numpy_array(adj)
    labels_dict = {}
    for n in range(num_nodes):
        node_l = np.argmax(feats[n])
        labels_dict[n] = node_labels[node_l]
    fig, ax_l = plt.subplots(2, 1, figsize=(15, 20))
    color_list = [ (0.9,0.9,0.9), (0.9,0.7,0.7), (0.9,0.4,0.4), (0.9,0.1,0.1)]
    node_colors = []
    weights = np.zeros_like(weight_nodes,dtype=np.int32)
    max_weight = np.max(weight_nodes)
    weights[weight_nodes > (0.5*max_weight)] = 1
    weights[weight_nodes > (0.75*max_weight)] = 2
    weights[weight_nodes > (0.9*max_weight)] = 3
    for i in range(weights.shape[0]):
        node_colors.append(color_list[weights[i]])


    nx.draw_networkx(G, labels=labels_dict, node_color = node_colors, ax=ax_l[0])
    # nx.draw_networkx(G,  node_color = node_colors, ax=ax_l[1])

    plt.savefig("./node_logs/visual/Rule_{}/{}_db{}_label{}_pred{}.pdf".format(rule, close_id,db,g_label,pred))


def get_hnodes(t_ix, rule_label, boundary_label):
    h_nodes = sublabel_nodes_array[t_ix]
    order_sl = sublabel_array[t_ix,:]

    order = []
    for sl in order_sl:
        if sl == -1:
            continue
        order.append(sl)
    #0,1 => 0 , 0,2 => 1, 1,0  => 2
    p_label = -1
    if (rule_label,boundary_label) == (0,1):
        p_label = 0
    elif (rule_label,boundary_label) == (0,2):
        p_label = 1
    elif (rule_label, boundary_label) == (1, 0):
        p_label = 2
    elif (rule_label, boundary_label) == (1, 2):
        p_label = 1
    elif (rule_label, boundary_label) == (2, 1):
        p_label = 0
    elif (rule_label, boundary_label) == (2, 0):
        p_label = 2
    if p_label not in order:
        return [], h_nodes

    h_index = order.index(p_label)
    return h_nodes[h_index*4:(h_index*4+4)], h_nodes[(1-h_index)*4:((1-h_index)*4+4)]









def load_sublabel_data():
    synthetic_data = pickle.load(open("../../gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20_12dlbls_nofake.p", "rb"))

    # synthetic_data = pickle.load(open("../../gcn_interpretation/data/synthetic_data_3label_3sublabel/synthetic_data_8000_comb_norep_max20.p", "rb"))
    sublabel_array = synthetic_data['sub_label']
    sublabel_nodes_array = synthetic_data['sub_label_nodes']
    # label = synthetic_data['label']

    highlight_nodes = {}
    for ix in range(sublabel_array.shape[0]):
        if ix not in highlight_nodes:
            highlight_nodes[ix] = []
        for hx in range(sublabel_array.shape[1]):
            if sublabel_array[ix, hx] == -1:
                continue
            for n in sublabel_nodes_array[ix, hx, 0]:
                highlight_nodes[ix].append(n)
        # assert (len(highlight_nodes[ix]) == 4 * label[ix])
    return highlight_nodes, sublabel_array

#===================   Define Utility Functions End  =====================

# substitute the labels of training data
train_data_upt = train_data.loadHardCopy()
test_data_upt = test_data.loadHardCopy()

# check training and testing accuracy
org_train_acu, pred_labels, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
org_test_acu, pred_labels_test, _                      = Trainer.evalAccuracyOfModel(model, test_data)
print('org_train_acu: %.5f' % (org_train_acu))
print('org_test_acu: %.5f' % (org_test_acu))

print('Original Training Data Shape:', train_data_upt._data[0].shape)

train_data_upt.updateLabels(model)
train_data_upt.updateData(model, I)
test_data_upt.updateData(model, I)
# train_data_upt.validateData(model)

print('Modified Training Data Shape:', train_data_upt._data[0].shape)

step_2 = time.time()
print('Time:', step_2 - step_1)

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
        print('Adding Gaussian Noise with Standard Derivation ', 0.3 + 0.2 * turn, '...Done')

if exper_robust_model:
    model_list = []
    train_data_upt_model = []

    for epoch in range(5):
        model_temp = VGG19Assira(num_classes_=2).cuda()
        model_temp.loadModel(model_temp._model_rootpath + model_temp.__class__.__name__ + str(epoch) + '.mdl')
        model_list.append(model_temp)

        org_train_acu, pred_labels, pred_probs = Trainer.evalAccuracyOfModel(model_temp, train_data)
        org_test_acu, pred_labels_test, _ = Trainer.evalAccuracyOfModel(model_temp, test_data)
        print('Model ' + str(epoch))
        print('org_train_acu: %.5f' % (org_train_acu))
        print('org_test_acu: %.5f' % (org_test_acu))

        train_data_upt_model_temp = train_data.loadHardCopy()
        train_data_upt_model_temp.updateLabels(model_temp)
        train_data_upt_model_temp.updateData(model_temp, I)
        train_data_upt_model.append(train_data_upt_model_temp)
        print('Updating Data using Model ' + str(epoch) + '...Done')



# show statics
train_labels = train_data._getTensorLabels().numpy()
test_labels  = test_data._getTensorLabels().numpy()
updated_train_labels = train_data_upt._getTensorLabels().numpy()
print('Training labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
print('Updated Training labels: ', [(lb, np.sum(updated_train_labels == lb)) for lb in set(updated_train_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')

# torch.save(train_data_upt._data, "./temp_train_data_upt_mutag.pth")
# exit()

iter = 0
np.random.seed(0)
print("train_data_upt: len: ", train_data_upt.__len__())
np.random.seed()

indices = np.random.permutation(train_data_upt.__len__())[0:2000]
check_repeat = np.repeat(False, train_data_upt.__len__())
check_repeat_test = np.repeat(False, test_data.__len__())

# for i in range(len(model._layers_list)):
#     model._layers_list[i].requires_grad_ = True

for l, p in model.named_children():
    print(l)
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
print("indices: ", len(indices))

_train_labels = train_data_upt._getArrayLabels()


print("labels: ", _train_labels.shape, pred_labels.shape, np.sum(_train_labels==pred_labels),  np.sum(_train_labels==0))


print("total seed images: ", len(indices))

import pickle
# all_data = pickle.load(open("/home/mohit/Mohit/gcn_interpretation/data/synthetic_data_2label_3sublabel/synthetic_data.p", "rb"))
# sublabel_nodes_array = all_data['sub_label_nodes']
# sublabel_array = all_data['sub_label']


sublabel_nodes_array, sublabel_array = load_sublabel_data()


for i in range(len(indices)):

    rule_label = train_data_upt._data[2][indices[i]]

    step_3 = time.time()

    if iter > num_classes * length - 1:
        break

    if check_balance.count(pred_labels[indices[i]]) > length - 1:
        continue
    else:
        check_balance.append(pred_labels[indices[i]])



    print('========================= Next Rule in Progress ==========================')

    print("Seed id : ", indices[i])
    print("Seed label: ", train_data_upt._data[2][indices[i]])
    print("Orig label: ", train_data._data[2][indices[i]])

    # Initialize the rule. Need to input the label for the rule and which layer we are using (I)
    rule_miner_train = RuleMinerLargeCandiPool(model, train_data_upt, pred_labels[indices[i]], I)

    # Compute the feature for the seed image
    feature = train_data_upt.getCNNFeature(indices[i]).cuda()
    # print("feature:", feature.shape)
    # label_update = model._getOutputOfOneLayer(feature)
    # print("label update: ", label_update)
    # exit()


    # Create candidate pool
    rule_miner_train.CandidatePoolLabel(feature)
    print('Candidate Pool Created')

    # Perform rule extraction
    # initial is not in use currently
    inv_classifier, pivots, opposite, initial = rule_miner_train.getInvariantClassifier(indices[i], feature.cpu().numpy(), pred_labels[indices[i]], train_data_upt._getArrayLabels(), delta_constr_=0)
    print('Rule Extraction Completed\n')
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
    rule_dict_list.append(rule_dict)
    #end saving info for gnn-explainer



    # evaluate classifier
    accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data_upt)
    accuracy_test, cover_indi_test   = evalSingleRule(inv_classifier, test_data_upt)
    accuracy_train_NN, _ = evalSingleRuleByNN(inv_classifier, train_data_upt)
    accuracy_test_NN, _ = evalSingleRuleByNN(inv_classifier, test_data_upt)

    assert(cover_indi_train[indices[i]] == True)
    for c_ix in range(cover_indi_train.shape[0]):
        if cover_indi_train[c_ix] == True:
            # idx2rule[c_ix] = len(rule_list_all)

            if c_ix not in idx2rule:
                idx2rule[c_ix] = []
            idx2rule[c_ix].append(len(rule_list_all))

    '''
    savefig(indices[i], 'Rule_' + str(i), train_data)
    savefig_Rule_Gradcam(indices[i], 'Rule_' + str(i), train_data)
    savefig_Rule_LIME(indices[i], 'Rule_' + str(i), train_data)

    img = train_data.getCNNFeature(indices[i]).cuda()
    img_processed = train_data_upt.getCNNFeature(indices[i]).cuda()

    for db, pivot in enumerate(pivots):
        mask = Boundary_Visualization(model, pivot, img_processed, I)
        cam = show_cam_on_image(img, mask)
        cv2.imwrite('Rule_' + str(i) + '_' + str(indices[i]) + 'db' + str(db) + '.jpg', np.uint8(255 * cam))
    '''


    if exper_visual:

        if not os.path.exists('node_logs/visual'):
            os.mkdir('node_logs/visual')

        dir = 'node_logs/visual/Rule_' + str(iter)
        os.mkdir(dir)
        dir_path = os.path.join(path, dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        img_seed = train_data_upt.getCNNFeature(indices[i]).numpy() # for finding nearest images
        img_seed = img_seed.flatten()


        for num, close_id in enumerate(NNpic(inv_classifier, img_seed)):

            # print(close_id, )

            if num == 0:
                print('Opposite Class for each Decision Boundary:')

            img = train_data.getCNNFeature(close_id).cuda() # original image
            img_processed = train_data_upt.getCNNFeature(close_id).cuda() # to compute heatmap
            feat_exp = train_data_upt._data_exp[close_id:close_id+1]
            img_label = train_data._data[2][close_id]

            for db, pivot in enumerate(pivots): #pivot is the boundary pt. on decision boundary

                # print("boundary :", rule_dict['boundary'][db]['basis'])


                if num == 0:
                    print('Decision Boundary ', str(db), ': ', class_list[opposite[db]]) # the class the boundary is trying to guard against

                # mask = Boundary_Visualization_Graph(model, pivot, img_processed, 1, I)
                sublabel_nodes = sublabel_nodes_array[close_id]

                # sublabel = sublabel_array[close_id]

                # gt_highlight_nodes, ngt_highlight_nodes = get_hnodes(close_id, rule_label, opposite[db])
                # print("hnodes: ", gt_highlight_nodes, ngt_highlight_nodes)
                mask = Boundary_Visualization_Graph(model, pivot, img_processed, feat_exp, 1, I)

                print("Image Close id: ", close_id)
                print("gt label: ", train_data._data[2][close_id].numpy())
                print("pred label: ", train_data_upt._data[2][close_id].numpy())
                assert(cover_indi_train[close_id] == True)
                print("\n")

                # print("mask: ", np.argmax(mask), np.max(mask))
                # print(mask)
                # print("num nodes:", train_data_upt._data[3][close_id])

                # mask_t = np.zeros_like(mask) #100
                # mask_max = np.max(mask)
                # mask_t[mask>0.5*mask_max] = 1.
                # mask_t = torch.from_numpy(mask_t).float().cuda().unsqueeze(1)
                # mask_t = mask_t.unsqueeze(0) #1*nodes*1
                # feat_exp_t = torch.from_numpy(feat_exp).float().cuda()
                # masked_img = torch.sum(feat_exp_t*mask_t, dim=1)/torch.sum(mask_t)
                #
                # old_preds = model._getOutputOfOneLayer(img_processed.detach())
                # old_preds = torch.nn.functional.softmax(old_preds, dim=1)
                # new_preds = model._getOutputOfOneLayer(masked_img)
                # new_preds = torch.nn.functional.softmax(new_preds,dim=1)
                # print("Old preds: ", old_preds.detach().cpu().numpy())
                # print("New preds: ", new_preds.detach().cpu().numpy())

                saveAndDrawGraph2(mask, iter, db, close_id, dbl=opposite[db])
                # if sublabel == -1:
                #     sublabel_nodes = []
                saveAndDrawGraph2(mask, iter, db, close_id, sublabel_nodes, dbl=opposite[db])



                if db == 0:
                    mask_average = mask
                else:
                    mask_average += mask
            #
            #
            mask_average = mask_average / len(pivots)
            # cam = show_heatmap_on_image(img, heatmap_average)
            # cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num) + '_' + str(close_id) + 'db_avg.jpg'), np.uint8(255 * cam))
            # print('Predicted Label: ', class_list[pred_labels[close_id]])
            # print('Ground Truth Label: ', class_list[train_data._data[2][close_id]])

            # Saving the original image, Grad-CAM result, and LIME result. Note that they all use the original data (train_data)
            # savefig(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data)
            # savefig_Rule_Gradcam(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data, model)
            # savefig_Rule_LIME(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data, model)


    if exper_robust_gaussian:

        if not os.path.exists('logs/robust_gaussian'):
            os.mkdir('logs/robust_gaussian')

        dir = 'logs/robust_gaussian/Rule_' + str(i)
        os.mkdir(dir)
        dir_path = os.path.join(path, dir)

        savefig(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_0'), train_data)
        savefig_Rule_Gradcam(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_0'), train_data, model)
        savefig_Rule_LIME(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_0'), train_data, model)

        img = train_data.getCNNFeature(indices[i]).cuda()
        img_processed = train_data_upt.getCNNFeature(indices[i]).cuda()

        for db, pivot in enumerate(pivots):
            mask = Boundary_Visualization(model, pivot, img_processed, I)
            cam = show_cam_on_image(img, mask)
            cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_0_' + str(indices[i]) + 'db' + str(db) + '.jpg'), np.uint8(255 * cam))

        for turn in range(3):
            rule_miner_train_shifted = RuleMinerLargeCandiPool(model, train_data_upt, pred_labels[indices[i]], I)

            feature_shifted = train_data_upt_gaussian[turn].getCNNFeature(indices[i]).cuda()

            rule_miner_train_shifted.CandidatePoolLabel(feature_shifted)
            inv_classifier_shifted, pivots, opposite = rule_miner_train_shifted.getInvariantClassifier(indices[i], feature_shifted.cpu().numpy(), pred_labels[indices[i]], train_data_upt_gaussian[turn]._getArrayLabels(), delta_constr_=0)
            savefig(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_' + str(turn+1)), train_data_gaussian[turn])
            savefig_Rule_Gradcam(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_' + str(turn+1)), train_data_gaussian[turn], model)
            savefig_Rule_LIME(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_' + str(turn+1)), train_data_gaussian[turn], model)

            img = train_data_gaussian[turn].getCNNFeature(indices[i]).cuda()
            img_processed = train_data_upt_gaussian[turn].getCNNFeature(indices[i]).cuda()

            for db, pivot in enumerate(pivots):
                mask = Boundary_Visualization(model, pivot, img_processed, I)
                cam = show_cam_on_image(img, mask)
                cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Noise_' + str(turn+1) + '_' + str(indices[i]) + 'db' + str(db) + '.jpg'), np.uint8(255 * cam))

    if exper_robust_model:

        if not os.path.exists('logs/robust_model'):
            os.mkdir('logs/robust_model')

        dir = 'logs/robust_model/Rule_' + str(i)
        os.mkdir(dir)
        dir_path = os.path.join(path, dir)

        for epoch in range(5):
            rule_miner_train_shifted = RuleMinerLargeCandiPool(model_list[epoch], train_data_upt_model[epoch], pred_labels[indices[i]], I)

            feature_shifted = train_data_upt_model[epoch].getCNNFeature(indices[i]).cuda()

            rule_miner_train_shifted.CandidatePoolLabel(feature_shifted)
            inv_classifier_shifted, pivots, opposite = rule_miner_train_shifted.getInvariantClassifier(indices[i], feature_shifted.cpu().numpy(), pred_labels[indices[i]], train_data_upt_model[epoch]._getArrayLabels(), delta_constr_=0)
            savefig(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Model_' + str(epoch)), train_data)
            savefig_Rule_Gradcam(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Model_' + str(epoch)), train_data, model_list[epoch])
            savefig_Rule_LIME(indices[i], os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Model_' + str(epoch)), train_data, model_list[epoch])

            img = train_data.getCNNFeature(indices[i]).cuda()
            img_processed = train_data_upt_model[epoch].getCNNFeature(indices[i]).cuda()

            for db, pivot in enumerate(pivots):
                mask = Boundary_Visualization(model_list[epoch], pivot, img_processed, I)
                cam = show_cam_on_image(img, mask)
                cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Model_' + str(epoch) + '_' + str(indices[i]) + 'db' + str(db) + '.jpg'), np.uint8(255 * cam))


    rule_list_all.append(inv_classifier)
    cover_list_all.append(cover_indi_train)

    for j in range(train_data_upt.__len__()):
        if cover_indi_train[j] == True:
            check_repeat[j] = True

    for k in range(test_data.__len__()):
        if cover_indi_test[k] == True:
            check_repeat_test[k] = True

    # show info
    print('========================= done %d/%d tgt_label: %d ==========================' %
          (iter, len(indices), inv_classifier._label))
    iter += 1

    print('Seed point ID: %d\n' % (indices[i]))
    print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))
    print('Training ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train, np.sum(cover_indi_train)))
    print('Testing ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test, np.sum(cover_indi_test)))
    print('Training (Model Decision) ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train_NN, np.sum(cover_indi_train)))
    print('Testing (Model Decision) ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test_NN, np.sum(cover_indi_test)))
    print('Covered training data: %d\n' % np.sum(check_repeat))
    print('Covered testing data: %d\n' % np.sum(check_repeat_test))

    step_4 = time.time()
    print('Time for this rule:', step_4 - step_3)


print('RULE_LIST_ALL => ')
printEvalInfo(rule_list_all, cover_list_all, is_set_cover)
printEvalInfoByNN(rule_list_all, cover_list_all, is_set_cover)
# print(np.sum(np.array(cover_list_all[0])))
# print(len(rule_list_all[0]))


rule_dict_save = {}
rule_dict_save['rules'] = rule_dict_list
rule_dict_save['idx2rule'] = idx2rule
# pickle.dump(rule_dict_save, open("./data/synthetic/rule_dict_synthetic_train_4k8000_comb_12dlbls_nofake.p","wb"))
pickle.dump(rule_dict_save, open("./data/syn4/rule_dict_syn4_train_all_dense.p","wb"))

exit()

acc_dict = {}
count_dict = {}
rule_boundary_w_hn_dict = {}
rule_boundary_w_ngt_dict = {}
rule_boundary_count_dict = {}
rule_max_weight_acc_dict = {}
rule_max_weight_ngt_acc_dict = {}
zero_count = 0

avg_top4_acc = 0.
avg_top6_acc = 0.
avg_top8_acc = 0.
acc_count = 0.
rule_top8_acc = 0.
rule_acc_count = 0.

hnodes_dict = {}
for ix, r in enumerate(rule_list_all):
    r_indices = (cover_list_all[ix] == True).nonzero()[0]
    pivots_ix = pivots_list[ix]
    opposite_ix = opposite_list[ix]
    rule_sublabel_acc = 0.
    rule_sublabel_acc_top6 = 0.
    rule_label = r._label
    rule_weight_hnodes = 0.
    rule_weight_ngtnodes = 0.
    rule_pos_count_acc = 0.
    # if rule_label == 0:
    #     continue

    for t_ix in r_indices:
        img_processed = train_data_upt.getCNNFeature(t_ix).cuda()
        feat_exp = train_data_upt._data_exp[t_ix:t_ix + 1]
        sublabel_nodes = sublabel_nodes_array[t_ix]
        sublabel = sublabel_array[t_ix]

        # if sublabel not in acc_dict:
        #     acc_dict[sublabel] = 0.
        #     count_dict[sublabel] = 0.

        rule_imp_nodes = []
        rule_h_nodes = []
        mask_cat = np.zeros((0))
        for pix, pivot in enumerate(pivots_ix):
            if (ix, pix) not in rule_boundary_w_hn_dict:
                rule_boundary_w_hn_dict[(ix, pix)] = 0.0
                rule_boundary_w_ngt_dict[(ix, pix)] = 0.0
                rule_boundary_count_dict[(ix, pix)] = 0.0
                rule_max_weight_acc_dict[(ix, pix)] = 0.0
                rule_max_weight_ngt_acc_dict[(ix, pix)] = 0.0

            boundary_label = opposite_ix[pix]
            mask = Boundary_Visualization_Graph(model, pivot, img_processed, feat_exp, 1, I)
            gt_highlight_nodes, ngt_highlight_nodes = get_hnodes(t_ix, rule_label, boundary_label)
            # gt_highlight_nodes = sublabel_nodes_array[t_ix].tolist()
            # ngt_highlight_nodes = []

            assert(-1 not in gt_highlight_nodes)
            hnodes_dict[(t_ix,pix)] = gt_highlight_nodes

            if np.argmax(mask) in gt_highlight_nodes:
                rule_pos_count_acc += 1.0
                rule_max_weight_acc_dict[(ix, pix)] += 1.0
            elif np.argmax(mask) in ngt_highlight_nodes:
                rule_max_weight_ngt_acc_dict[(ix, pix)] += 1.0

            imp_nodes = mask.argsort()[-8:][::-1]
            top4_acc, top6_acc, top8_acc = getAcc(imp_nodes, gt_highlight_nodes)


            avg_top4_acc += top4_acc
            avg_top6_acc += top6_acc
            avg_top8_acc += top8_acc
            acc_count += 1
            # assert(len(gt_highlight_nodes) + len(ngt_highlight_nodes) <= 8)
            # top4_acc, top6_acc, top8_acc = getAcc(imp_nodes, (gt_highlight_nodes+ngt_highlight_nodes))
            # rule_top8_acc += top8_acc
            # rule_acc_count += 1.0

            mask_cat = np.concatenate((mask_cat,mask))
            # rule_imp_nodes.extend(imp_nodes[:4])
            rule_h_nodes.extend(gt_highlight_nodes)

            if (np.sum(mask) == 0.):
                zero_count += 1
                denominator = 1.
            else:
                denominator = np.sum(mask)

            mask = mask/denominator

            weight_hnodes = 0.

            for n in gt_highlight_nodes:
                weight_hnodes += mask[n]

            weight_ngtnodes = 0.

            for n in ngt_highlight_nodes:
                weight_ngtnodes += mask[n]

            rule_boundary_w_hn_dict[(ix,pix)] += weight_hnodes
            rule_boundary_w_ngt_dict[(ix, pix)] += weight_ngtnodes
            rule_boundary_count_dict[(ix, pix)] += 1.0

            rule_weight_hnodes += weight_hnodes
            rule_weight_ngtnodes += weight_ngtnodes

        sort_nodes = mask_cat.argsort()[::-1]
        for s_ix in range(sort_nodes.shape[0]):
            node_ix = sort_nodes[s_ix]%(sort_nodes.shape[0]/2)
            if node_ix not in rule_imp_nodes:
                rule_imp_nodes.append(node_ix)
            if len(rule_imp_nodes) == 8:
                break
        top4_acc, top6_acc, top8_acc = getAcc(rule_imp_nodes, rule_h_nodes)
        rule_top8_acc += top8_acc
        rule_acc_count += 1.0



    rule_weight_hnodes = rule_weight_hnodes/(len(r_indices)*len(pivots_ix))
    rule_weight_ngtnodes = rule_weight_ngtnodes/(len(r_indices)*len(pivots_ix))
    rule_pos_count_acc = rule_pos_count_acc/(len(r_indices)*len(pivots_ix))


    print("Rule number: {}, Rule label: {}, Rule hgt weights: {}, Rule ngt weights: {}, Rule max acc: {}".format(ix+1, r._label, rule_weight_hnodes, rule_weight_ngtnodes, rule_pos_count_acc))

final_dict = []

for k,v in rule_boundary_w_hn_dict.items():

    rt_w = v/rule_boundary_count_dict[k]
    wg_w = rule_boundary_w_ngt_dict[k]/rule_boundary_count_dict[k]
    max_w_acc = rule_max_weight_acc_dict[k]/rule_boundary_count_dict[k]
    max_w_acc_ngt = rule_max_weight_ngt_acc_dict[k]/rule_boundary_count_dict[k]
    rb_dict = {'Weight_rt': rt_w, 'Weight_wg': wg_w, 'Max weight acc': max_w_acc, 'Max weight acc ngt': max_w_acc_ngt }
    final_dict.append(rb_dict)
    print("\n")
    print("Rule: {}, boundary: {}, 'Weight_rt': {}, 'Weight_wg': {}, 'Weight sum': {}, 'Max weight acc': {}, 'Max weight acc ngt': {}".format(
        k[0],k[1], rt_w, wg_w, (rt_w + wg_w), max_w_acc, max_w_acc_ngt))


pickle.dump(final_dict, open("./results_final_results.p", "wb"))
pickle.dump(hnodes_dict, open("./data/synthetic/hnodes_dict_synthetic_train_4k8000_comb_12dlbls_nofake.p", "wb"))

print("zero count: ", zero_count)

print(
    "Boundary wise top4 acc: {}, top6 acc: {}, top8 acc: {}".format(avg_top4_acc / acc_count, avg_top6_acc / acc_count,
                                                                    avg_top8_acc / acc_count)
)

print(
    "Rule wise top8 acc: {}".format(rule_top8_acc / rule_acc_count)
)

exit()
# acc_dict = {}
# count_dict = {}
# for ix, r in enumerate(rule_list_all):
#     r_indices = (cover_list_all[ix] == True).nonzero()[0]
#     pivots_ix = pivots_list[ix]
#     opposite_ix = opposite_list[ix]
#     rule_sublabel_acc = 0.
#     rule_sublabel_acc_top6 = 0.
#     rule_label = r._label
#
#     for t_ix in r_indices:
#         img_processed = train_data_upt.getCNNFeature(t_ix).cuda()
#         feat_exp = train_data_upt._data_exp[t_ix:t_ix + 1]
#         sublabel_nodes = sublabel_nodes_array[t_ix]
#         sublabel = sublabel_array[t_ix]
#
#         if sublabel not in acc_dict:
#             acc_dict[sublabel] = 0.
#             count_dict[sublabel] = 0.
#
#
#
#         for pix, pivot in enumerate(pivots_ix):
#             boundary_label = opposite_ix[pix]
#             mask = Boundary_Visualization_Graph(model, pivot, img_processed, feat_exp, 1, I)
#             top4 = mask.argsort()[-4:][::-1]
#             top6 = mask.argsort()[-6:][::-1]
#
#             l_count = 0.
#             acc_top6 = 0.0
#             for l in sublabel_nodes:
#                 if l in top4:
#                     l_count += 1.0
#                 if l in top6:
#                     acc_top6 += 1.0
#             acc = l_count/4.0
#             acc_top6 = acc_top6/4.0
#             rule_sublabel_acc += acc
#             rule_sublabel_acc_top6 += acc_top6
#
#             acc_dict[sublabel] += acc
#             count_dict[sublabel] += 1.0
#
#     rule_sublabel_acc = rule_sublabel_acc/(len(r_indices)*len(pivots_ix))
#     rule_sublabel_acc_top6 = rule_sublabel_acc_top6/(len(r_indices)*len(pivots_ix))
#
#     print("Rule number: {}, Rule label: {}, Rule accuracy: {}, Rule top6 accuracy: {}".format(ix+1, r._label, rule_sublabel_acc, rule_sublabel_acc_top6))
# for k,v in acc_dict.items():
#     print("Accuracy for sublabel {}: {}".format(k, v/count_dict[k]))




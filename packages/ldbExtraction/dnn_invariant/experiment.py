
import time
step_1 = time.time()

import sys

sys.path.insert(0, "/home/mohit/Mohit/model_interpretation/ai-adversarial-detection")
# print(sys.path)
from dnn_invariant.utilities.environ import *
from dnn_invariant.models.models4invariant import *
from dnn_invariant.utilities.datasets import *
from dnn_invariant.utilities.trainer import *
import scipy.sparse as sp
import time
import dill
# from dnn_invariant.utilities.visualization import *
import cv2
from dnn_invariant.algorithms.mine_cnn_invariant import *
from dnn_invariant.algorithms.gradcam import *
from dnn_invariant.algorithms.lime_image import *
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
num_classes = 2
exper_visual = True # do not change this
exper_robust_gaussian = False # do not change this
exper_robust_model = False # do not change this



model = VGG19(num_classes_=num_classes).cuda()
#model = CNN4MNIST(num_classes_=num_classes).cuda()

model_name = 'Assira_3epochs.mdl'
#model_name = 'ZhangLab_3epochs.mdl'
#model_name = 'MNIST_24.mdl'
#model_name = 'FMNIST_24.mdl'

top_similar = 5









print('Train Data Size: ', train_data._data[0].shape)

model.loadModel(model._model_rootpath + model_name)

print(model._layers_list)

is_set_cover    = False

# Experiment Settings

if os.path.exists('logs') and os.path.isdir('logs'):
    shutil.rmtree('logs') # remove old logs

os.mkdir('logs')
path = '/home/mohit/Mohit/model_interpretation/ai-adversarial-detection/dnn_invariant'





#===================   Parameter settings End   ==========================

#===================   Define Utility Functions Start  ===================
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
        output = model._getOutputOfOneLayer(input, model._layers_list.__len__() - 1, I + 1).cpu().detach().numpy()
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
        output = model._getOutputOfOneLayer(input, model._layers_list.__len__() - 1, I + 1).cpu().detach().numpy()
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


iter = 0
np.random.seed(0)
indices = np.random.permutation(train_data_upt.__len__())[0:2000]

check_repeat = np.repeat(False, train_data_upt.__len__())
check_repeat_test = np.repeat(False, test_data.__len__())

for i in range(len(model._layers_list)):
    model._layers_list[i].requires_grad_ = True


rule_list_all = []
cover_list_all = []
check_repeat = np.repeat(False, train_data_upt.__len__())
check_repeat_test = np.repeat(False, test_data.__len__())
check_balance = []

for i in range(len(indices)):

    step_3 = time.time()

    if iter > num_classes * length - 1:
        break

    if check_balance.count(pred_labels[indices[i]]) > length - 1:
        continue
    else:
        check_balance.append(pred_labels[indices[i]])

    print('========================= Next Rule in Progress ==========================')

    # Initialize the rule. Need to input the label for the rule and which layer we are using (I)
    rule_miner_train = RuleMinerLargeCandiPool(model, train_data_upt, pred_labels[indices[i]], I)

    # Compute the feature for the seed image
    feature = train_data_upt.getCNNFeature(indices[i]).cuda()


    # Create candidate pool
    rule_miner_train.CandidatePoolLabel(feature)
    print('Candidate Pool Created')

    # Perform rule extraction
    # initial is not in use currently
    inv_classifier, pivots, opposite, initial = rule_miner_train.getInvariantClassifier(indices[i], feature.cpu().numpy(), pred_labels[indices[i]], train_data_upt._getArrayLabels(), delta_constr_=0)
    print('Rule Extraction Completed\n')

    # evaluate classifier
    accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data_upt)
    accuracy_test, cover_indi_test   = evalSingleRule(inv_classifier, test_data_upt)
    accuracy_train_NN, _ = evalSingleRuleByNN(inv_classifier, train_data_upt)
    accuracy_test_NN, _ = evalSingleRuleByNN(inv_classifier, test_data_upt)


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

        if not os.path.exists('logs/visual'):
            os.mkdir('logs/visual')

        dir = 'logs/visual/Rule_' + str(i)
        os.mkdir(dir)
        dir_path = os.path.join(path, dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        img_seed = train_data_upt.getCNNFeature(indices[i]).numpy() # for finding nearest images
        img_seed = img_seed.flatten()

        for num, close_id in enumerate(NNpic(inv_classifier, img_seed)):

            if num == 0:
                print('Opposite Class for each Decision Boundary:')

            img = train_data.getCNNFeature(close_id).cuda() # original image
            img_processed = train_data_upt.getCNNFeature(close_id).cuda() # to compute heatmap

            for db, pivot in enumerate(pivots):

                if num == 0:
                    print('Decision Boundary ', str(db), ': ', class_list[opposite[db]]) # the class the boundary is trying to guard against

                mask = Boundary_Visualization(model, pivot, img_processed, 1, I)
                #mask[mask < np.quantile(mask, 0.90)] = 0 # top 10% of heatmap
                cam = show_cam_on_image(img, mask)
                cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num) + '_' + str(close_id) + 'db' + str(db) + '.jpg'), np.uint8(255 * cam))

                if db == 0:
                    heatmap_average = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap_average = np.float32(heatmap_average) / 255
                else:
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap_average += heatmap


            heatmap_average = heatmap_average / len(pivots)
            cam = show_heatmap_on_image(img, heatmap_average)
            cv2.imwrite(os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num) + '_' + str(close_id) + 'db_avg.jpg'), np.uint8(255 * cam))
            print('Predicted Label: ', class_list[pred_labels[close_id]])
            print('Ground Truth Label: ', class_list[train_data._data[1][close_id]])

            # Saving the original image, Grad-CAM result, and LIME result. Note that they all use the original data (train_data)
            savefig(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data)
            savefig_Rule_Gradcam(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data, model)
            savefig_Rule_LIME(close_id, os.path.join(dir_path, 'Rule_' + class_list[pred_labels[indices[i]]] + '_' + str(i) + '_Close_' + str(num)), train_data, model)


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


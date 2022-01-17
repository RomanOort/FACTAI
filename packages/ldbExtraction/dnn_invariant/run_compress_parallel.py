
from dnn_invariant.utilities.environ import *
from models.models4invariant import *
from dnn_invariant.utilities.datasets import *
from dnn_invariant.utilities.trainer import *
import scipy.sparse as sp
import time
import dill
from dnn_invariant.utilities.visualization import *
from algorithms.mine_cnn_invariant_SG import *
import copy
import multiprocessing


#===================   Define Utility Functions Start  ===================
# evaluate the accuracy of a single rule on mdata
def evalSingleRule(inv_classifier, mdata):
    pred_labels, cover_indi   = inv_classifier.classify(mdata._getArrayFeatures())
    label_diff                = pred_labels[cover_indi] - mdata._getArrayLabels()[cover_indi]
    accuracy                  = np.sum(label_diff == 0)/label_diff.size
    return accuracy, cover_indi

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
    accuracy_test, num_covered_test           = evalAllRules(rule_list, cover_list, test_data, is_set_cover)
    accuracy_valid, num_covered_valid = evalAllRules(rule_list, cover_list, valid_data, is_set_cover)
    accuracy_train_upt, num_covered_train_upt = evalAllRules(rule_list, cover_list, train_data_upt, is_set_cover)
    accuracy_train_org, num_covered_train_org = evalAllRules(rule_list, cover_list, train_data, is_set_cover)

    print('num rules: %d' % (rule_list.__len__()))
    print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
    print('Org test accuracy: %.5f' % (org_test_acu))
    print('valid_accuracy_org: %.5f\tnum_covered: %d/%d' % (
        accuracy_valid, num_covered_valid, valid_data.__len__()))
    print('Org valid accuracy: %.5f' % (org_valid_acu))
    print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (
    accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))
    print('train_accuracy_org: %.5f\tnum_covered_train_org: %d/%d' % (
    accuracy_train_org, num_covered_train_org, train_data_upt.__len__()))

def mThread(idx):
    inv_classifier = rule_miner_train.getInvariantClassifier(idx, train_data_upt._getArrayLabels(), delta_constr_=0)
    return inv_classifier

#===================   Define Utility Functions End  =====================

#===================   Parameter settings Start   ========================
#model           = MLP2D(num_classes_=2).cuda()
#model           = CNN4Invariant(num_classes_=2).cuda()
model           = CNN4CIFAR(num_classes_=2).cuda()
#model           = CNN_AvgPool_Small2(num_classes_=2).cuda()
#model           = MLP4Invariant(num_classes_=2).cuda()

model.loadModel()

is_set_cover    = False

#===================   Parameter settings End   ==========================

# check training and testing accuracy
org_train_acu, pred_labels, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
org_valid_acu, _, _                     = Trainer.evalAccuracyOfModel(model, valid_data)
org_test_acu, _, _                      = Trainer.evalAccuracyOfModel(model, test_data)
print('org_train_acu: %.5f' % (org_train_acu))
print('org_valid_acu: %.5f' % (org_valid_acu))
print('org_test_acu: %.5f' % (org_test_acu))

# substitute the labels of training data
train_data_upt = train_data.loadHardCopy()
train_data_upt.updateLabels(model)

# show statics
train_labels = train_data._getTensorLabels().numpy()
valid_labels = valid_data._getTensorLabels().numpy()
test_labels  = test_data._getTensorLabels().numpy()
print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
print('Validation labels: ', [(lb, np.sum(valid_labels == lb)) for lb in set(valid_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')


# preparing labours
rule_miner_train        = RuleMinerLargeCandiPool(model, train_data_upt)

# start running in parallel
iter = 0
np.random.seed(0)
indices = np.random.permutation(train_data_upt.__len__())[0:40]
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
rule_list_test = []
rule_list_valid = []
rule_list_all = []
cover_list_test = []
cover_list_valid = []
cover_list_all = []
for inv_classifier in pool.imap_unordered(mThread, indices, 4):
    # evaluate classifier
    accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data_upt)
    accuracy_valid, cover_indi_valid = evalSingleRule(inv_classifier, valid_data)
    accuracy_test, cover_indi_test   = evalSingleRule(inv_classifier, test_data)

    # log qualified rules
    if accuracy_valid > org_valid_acu:
        rule_list_valid.append(inv_classifier)
        cover_list_valid.append(cover_indi_train)
    if accuracy_test > org_test_acu:
        rule_list_test.append(inv_classifier)
        cover_list_test.append(cover_indi_train)

    rule_list_all.append(inv_classifier)
    cover_list_all.append(cover_indi_train)

    # show info
    print('========================= done %d/%d tgt_label: %d ==========================' %
          (iter, len(indices), inv_classifier._label))
    iter += 1
    print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))
    print('Training ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train, np.sum(cover_indi_train)))
    print('Validation ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_valid, np.sum(cover_indi_valid)))
    print('Testing ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test, np.sum(cover_indi_test)))

pool.close()

# FINAL EVALUATION
print('RULE_LIST_VALID => ')
printEvalInfo(rule_list_valid, cover_list_valid, is_set_cover)
print('RULE_LIST_TEST => ')
printEvalInfo(rule_list_test, cover_list_test, is_set_cover)
print('RULE_LIST_ALL => ')
printEvalInfo(rule_list_all, cover_list_all, is_set_cover)







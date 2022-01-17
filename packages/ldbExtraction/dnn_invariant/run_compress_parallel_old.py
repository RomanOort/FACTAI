
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


# parameter settings
#model           = MLP2D(num_classes_=2).cuda()
#model           = CNN_AvgPool_Small(num_classes_=2).cuda()
#model           = MLP4Invariant(num_classes_=2).cuda()
model           = CNN4Invariant(num_classes_=2).cuda()
#model           = CNN4CIFAR(num_classes_=2).cuda()

tgt_modus       = [nn.Linear, nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.BatchNorm1d, nn.AvgPool1d, nn.MaxPool1d]

# load model
model.loadModel()

# check training and testing accuracy
org_train_acu, pred_labels_gt, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
org_test_acu, _, _                         = Trainer.evalAccuracyOfModel(model, test_data)

print('org_train_acu: %.8f' % (org_train_acu))
print('org_test_acu: %.8f' % (org_test_acu))

# substitute the labels of training data
#train_data.expandByRand(0.1, 2, model)
org_train_labels = train_data._getArrayLabels()
train_data.updateLabels(model)

test_data_upt = test_data.loadHardCopy()
test_data_upt.updateLabels(model)

rule_miner_train = RuleMinerLargeCandiPool(model, train_data, [15])

# show statics
train_labels = train_data._getTensorLabels().numpy()
valid_labels = valid_data._getTensorLabels().numpy()
test_labels  = test_data._getTensorLabels().numpy()
print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
print('Validation labels: ', [(lb, np.sum(valid_labels == lb)) for lb in set(valid_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')


def mThread(idx):
    inv_classifier = rule_miner_train.getInvariantClassifier(idx, org_train_labels, delta_constr_=0)
    return inv_classifier


iter = 0
np.random.seed(0)
indices = np.random.permutation(train_data.__len__())[0:100]
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
rule_list_test = []
rule_list_valid = []
rule_list_all = []
for inv_classifier in pool.imap_unordered(mThread, indices, 4):
    print('========================= done %d/%d tgt_label: %d ==========================' % (iter, len(indices), inv_classifier._label))
    iter += 1

    print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))

    pred_labels, cover_indi_train = inv_classifier.classify(train_data._getArrayFeatures())
    label_diff = pred_labels[cover_indi_train] - train_data._getArrayLabels()[cover_indi_train]
    accuracy_train = np.sum(label_diff == 0) / label_diff.size
    num_samples_train = np.sum(cover_indi_train)
    print('Training ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_train, num_samples_train))

    # print(np.where(cover_indi_train)[0])

    #spt_score = np.sum(pred_probs[cover_indi_train])
    #inv_classifier.updateSupportScore(spt_score)

    pred_labels, cover_indi_valid = inv_classifier.classify(valid_data._getArrayFeatures())
    label_diff = pred_labels[cover_indi_valid] - valid_data._getArrayLabels()[cover_indi_valid]
    accuracy_valid = np.sum(label_diff == 0) / label_diff.size
    num_samples_valid = np.sum(cover_indi_valid)
    print('Validation ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_valid, num_samples_valid))

    pred_labels, cover_indi_test = inv_classifier.classify(test_data._getArrayFeatures())
    label_diff = pred_labels[cover_indi_test] - test_data._getArrayLabels()[cover_indi_test]
    accuracy_test = np.sum(label_diff == 0) / label_diff.size
    num_samples_test = np.sum(cover_indi_test)
    print('Testing ORG ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_test, num_samples_test))

    pred_labels, cover_indi_test_upt = inv_classifier.classify(test_data_upt._getArrayFeatures())
    label_diff = pred_labels[cover_indi_test_upt] - test_data_upt._getArrayLabels()[cover_indi_test_upt]
    accuracy_test_upt = np.sum(label_diff == 0) / label_diff.size
    num_samples_test_upt = np.sum(cover_indi_test_upt)
    print('Testing UPT ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_test_upt, num_samples_test_upt))

    if accuracy_valid > org_test_acu:
        rule_list_valid.append(inv_classifier)
    if accuracy_test > org_test_acu:
        rule_list_test.append(inv_classifier)

    rule_list_all.append(inv_classifier)

pool.close()

def evaluation(tnsr_feat_test, tnsr_label_test, rule_list):
    # test the accuracy of the selected set of rules
    from queue import PriorityQueue as PQ
    pred_votes = {}

    #for rule_idx in np.where(sel_indi)[0]:
    #    inv_classifier = rule_list[rule_idx]
    for inv_classifier in rule_list:

        pred_labels, cover_indi = inv_classifier.classify(tnsr_feat_test)
        #spt = inv_classifier._spt_score

        for idx in np.where(cover_indi)[0]:
            if idx in pred_votes:
                pred_votes[idx].append(pred_labels[idx])
            else:
                pred_votes[idx] = [pred_labels[idx]]

    num_correct = 0
    num_covered = 0
    num_total   = 0
    for i in range(tnsr_label_test.shape[0]):
        num_total += 1
        if i in pred_votes:
            num_covered += 1
            #pred_label  = np.median(pred_votes[i])
            pred_label = sorted([(np.sum(pred_votes[i] == pred), pred) for pred in set(pred_votes[i])])[-1][-1]
            gt_label     = tnsr_label_test[i]
            if pred_label == gt_label:
                num_correct += 1
            '''
            else:
                print('--------------------------------------')
                print(pred_votes[i])
                print('gt_label: %d\tpred_label: %d\n' % (gt_label, pred_label))
            '''

    test_accuracy = num_correct/(num_covered+1)

    return test_accuracy, num_covered


def print_info(rule_list):
    print(rule_list.__len__())

    test_accuracy_org, num_covered_org = evaluation(test_data._getArrayFeatures(), test_data._getArrayLabels(), rule_list)
    print('Test accuracy_org: %.3f\tnum_covered: %d\tnum_test_samples: %d\t' % (test_accuracy_org, num_covered_org, test_data.__len__()))

    test_accuracy_upt, num_covered_upt = evaluation(test_data_upt._getArrayFeatures(), test_data_upt._getArrayLabels(), rule_list)
    print('Test accuracy_upt: %.3f\tnum_covered: %d\tnum_test_samples: %d\t' % (test_accuracy_upt, num_covered_upt, test_data.__len__()))

    print('Org_test_acu: %.3f' % (org_test_acu))

    valid_accuracy_org, num_covered_valid = evaluation(valid_data._getArrayFeatures(), valid_data._getArrayLabels(), rule_list)
    print('Valid accuracy_org: %.3f\tnum_covered: %d\tnum_train_samples: %d\t' % (valid_accuracy_org, num_covered_valid, valid_data.__len__()))

    train_accuracy_org, num_covered_train = evaluation(train_data_org._getArrayFeatures(), train_data_org._getArrayLabels(), rule_list)
    print('Train accuracy_org: %.3f\tnum_covered: %d\tnum_train_samples: %d\t' % (train_accuracy_org, num_covered_train, train_data_org.__len__()))

    train_accuracy_upt, num_covered_train_upt = evaluation(train_data._getArrayFeatures(), train_data._getArrayLabels(), rule_list)
    print('Train accuracy_upt: %.3f\tnum_covered: %d\tnum_train_samples: %d\t' % (train_accuracy_upt, num_covered_train_upt, train_data.__len__()))

    print('Org_train_acu: %.3f' % (org_train_acu))

print('RULE_LIST_VALID => ')
print_info(rule_list_valid)

print('RULE_LIST_TEST => ')
print_info(rule_list_test)

print('RULE_LIST_ALL => ')
print_info(rule_list_all)





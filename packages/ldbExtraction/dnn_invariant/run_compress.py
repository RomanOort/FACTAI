
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

#===================   Define Utility Functions End  =====================

#===================   Parameter settings Start   ========================
#model           = MLP2D(num_classes_=2).cuda()
#model           = CNN4Invariant(num_classes_=2).cuda()
model           = CNN4CIFAR(num_classes_=2).cuda()
#model           = CNN_AvgPool_Small2(num_classes_=2).cuda()
#model           = MLP4Invariant(num_classes_=2).cuda()

model.loadModel()

is_svd_plot     = False
is_set_cover    = False

#===================   Parameter settings End   ==========================

# check training and testing accuracy
org_train_acu, pred_labels, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
org_test_acu, _, _                      = Trainer.evalAccuracyOfModel(model, test_data)
print('org_train_acu: %.3f' % (org_train_acu))
print('org_test_acu: %.3f' % (org_test_acu))

# substitute the labels of training data
train_data_upt = train_data.loadHardCopy()
train_data_upt.updateLabels(model)

# plot train data
if is_svd_plot:
    psvd = PlotSVD(train_data._getArrayFeatures())

# show statics
train_labels = train_data._getTensorLabels().numpy()
valid_labels = valid_data._getTensorLabels().numpy()
test_labels  = test_data._getTensorLabels().numpy()
print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
print('Validation labels: ', [(lb, np.sum(valid_labels == lb)) for lb in set(valid_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')

# prepare controlling variables
np.random.seed(0) # use quasi randomness, comment this line if need real randomness

rule_list               = []
cover_list              = []
coverage_ratio_train    = 0
train_data_cover_count  = np.zeros(train_data_upt.__len__())
valid_data_cover_count  = np.zeros(valid_data.__len__())
test_data_cover_count   = np.zeros(test_data.__len__())
cover_indi_train        = np.zeros(train_data_upt.__len__(), dtype='bool')
iter_count              = 0
rule_miner_train        = RuleMinerLargeCandiPool(model, train_data_upt)

# start iteration for training
for i in np.random.permutation(train_data_cover_count.shape[0]):
    # check out
    if iter_count >= 500 or coverage_ratio_train >= 0.99:
        break

    # show progress
    print('====================== idx = %d; iter = %d/%d; tgt label: %d ======================'
          % (i, iter_count, train_data.__len__(), train_data_upt._getTensorLabels()[i]))
    iter_count += 1

    # plot SVD of remaining data
    if is_svd_plot:
        psvd.drawPoints(train_data_upt._getArrayFeatures()[train_data_cover_count<=0, :],
                        train_data_upt._getArrayLabels()[train_data_cover_count<=0],
                        ('svd_' + str(iter_count) + '_' + str(i) + '.jpg'))

    # skip covered train data
    if train_data_cover_count[i] > 0:
        continue

    # get classifier
    peel_mask = train_data_cover_count > 0
    inv_classifier = rule_miner_train.getInvariantClassifier(i, train_data_upt._getArrayLabels(), delta_constr_=0)

    if inv_classifier is None:
        train_data_cover_count[i] += 1
        continue

    # evaluate classifier
    accuracy_train, cover_indi_train = evalSingleRule(inv_classifier, train_data_upt)
    accuracy_valid, cover_indi_valid = evalSingleRule(inv_classifier, valid_data)
    accuracy_test, cover_indi_test   = evalSingleRule(inv_classifier, test_data)

    print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))
    print('Training ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_train, np.sum(cover_indi_train)))
    print('Validation ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_valid, np.sum(cover_indi_valid)))
    print('Testing ==> ACU: %.5f\tnum_samples: %d\n' % (accuracy_test, np.sum(cover_indi_test)))

    # select classifier
    if accuracy_valid >= org_test_acu:
        # update cover count
        train_data_cover_count[cover_indi_train] += 1
        valid_data_cover_count[cover_indi_valid] += 1
        test_data_cover_count[cover_indi_test] += 1

        # update rule_list and cover_list with the SAME ORDER of list items
        rule_list.append(inv_classifier)
        cover_list.append(cover_indi_train)

    print('Training ==> num covered samples: %d\t coverage ratio: %.3f\n' %
         (np.sum(train_data_cover_count > 0),
          np.sum(train_data_cover_count > 0) / train_data_cover_count.__len__()))
    print('Validation ==> num covered samples: %d\t coverage ratio: %.3f\n' %
         (np.sum(valid_data_cover_count > 0),
          np.sum(valid_data_cover_count > 0) / valid_data_cover_count.__len__()))
    print('Testing ==> num covered samples: %d\t coverage ratio: %.3f\n' %
         (np.sum(test_data_cover_count > 0),
          np.sum(test_data_cover_count > 0) / test_data_cover_count.__len__()))
    print('Num of candidate rules: %d\n' % (rule_list.__len__()))

# FINAL EVALUATION
accuracy_test, num_covered_test = evalAllRules(rule_list, cover_list, test_data, is_set_cover)
accuracy_valid, num_covered_valid = evalAllRules(rule_list, cover_list, valid_data, is_set_cover)
accuracy_train_upt, num_covered_train_upt = evalAllRules(rule_list, cover_list, train_data_upt, is_set_cover)
accuracy_train_org, num_covered_train_org = evalAllRules(rule_list, cover_list, train_data, is_set_cover)

print('test_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_test, num_covered_test, test_data.__len__()))
print('Org test accuracy: %.5f' % (org_test_acu))
print('valid_accuracy_org: %.5f\tnum_covered: %d/%d' % (accuracy_valid, num_covered_valid, valid_data.__len__()))
print('train_accuracy_upt: %.5f\tnum_covered: %d/%d' % (accuracy_train_upt, num_covered_train_upt, train_data_upt.__len__()))
print('train_accuracy_org: %.5f\tnum_covered_train_org: %d/%d' % (accuracy_train_org, num_covered_train_org, train_data_upt.__len__()))

# dump uncovered data
train_data.dumpByIndi(train_data_cover_count <= 0)
test_data.dumpByIndi(test_data_cover_count <= 0)

# plot svd points
if is_svd_plot:
    psvd.drawPoints(train_data._getTensorFeatures().numpy(), train_data._getTensorLabels().numpy(), './svd_train2D.jpg')
    psvd.drawPoints(valid_data._getTensorFeatures().numpy(), valid_data._getTensorLabels().numpy(), './svd_valid2D.jpg')
    psvd.drawPoints(test_data._getTensorFeatures().numpy(), test_data._getTensorLabels().numpy(), './svd_test2D.jpg')

    uncovered_train_data = train_data._getTensorFeaturesByIndi(train_data_cover_count <= 0).numpy()
    uncovered_train_label= train_data._getTensorLabelsByIndi(train_data_cover_count <= 0).numpy()
    psvd.drawPoints(uncovered_train_data, uncovered_train_label, 'svd_uncovered_traindata.jpg')

    uncovered_valid_data = valid_data._getTensorFeaturesByIndi(valid_data_cover_count <= 0).numpy()
    uncovered_valid_label= valid_data._getTensorLabelsByIndi(valid_data_cover_count <= 0).numpy()
    psvd.drawPoints(uncovered_valid_data, uncovered_valid_label, 'svd_uncovered_validdata.jpg')

    uncovered_test_data  = test_data._getTensorFeaturesByIndi(test_data_cover_count <= 0).numpy()
    uncovered_test_label = test_data._getTensorLabelsByIndi(test_data_cover_count <= 0).numpy()
    psvd.drawPoints(uncovered_test_data, uncovered_test_label, 'svd_uncovered_testdata.jpg')









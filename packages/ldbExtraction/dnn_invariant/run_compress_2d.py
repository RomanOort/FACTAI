
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

# parameter settings
#model           = MLP2D(num_classes_=2).cuda()
#model           = CNN4Invariant(num_classes_=2).cuda()
model           = CNN4CIFAR(num_classes_=2).cuda()
#model           = CNN_AvgPool_Small2(num_classes_=2).cuda()
#model           = MLP4Invariant(num_classes_=2).cuda()

tgt_modus       = [nn.Linear, nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.BatchNorm1d, nn.AvgPool1d, nn.MaxPool1d]
#tgt_modus       = [nn.MaxPool2d]
is_2d_data      = False
is_svd_plot     = False

is_load_dill    = False
dill_filename   = './pkls/tmp.pkl'

# load model
model.loadModel()

# show dill filename
print(dill_filename)

# check training and testing accuracy
org_train_acu, pred_labels, pred_probs  = Trainer.evalAccuracyOfModel(model, train_data)
org_test_acu, _, _                      = Trainer.evalAccuracyOfModel(model, test_data)

print('org_train_acu: %.3f' % (org_train_acu))
print('org_test_acu: %.3f' % (org_test_acu))

# substitute the labels of training data
org_train_labels = np.array(train_data._getArrayLabels())

train_data.updateLabels(model)
valid_data.updateLabels(model)
test_data.updateLabels(model)

if is_2d_data:
    print('range: ', train_data.getRange(dim_=0, label_=1))

rule_miner_train = RuleMinerLargeCandiPool(model, train_data, [15])

if is_2d_data:
    Plot2D().drawLines(rule_miner_train._bb.bb, 'candidate_lines.jpg')

if is_2d_data:
    indi = pred_probs < 0.6
    ambiguous_train_data = train_data._getTensorFeaturesByIndi(indi).numpy()
    ambiguous_train_label = train_data._getTensorLabelsByIndi(indi).numpy()
    Plot2D().drawPoints(ambiguous_train_data, ambiguous_train_label, (
                'ambiguous_traindata.jpg'))

# show statics
train_labels = train_data._getTensorLabels().numpy()
valid_labels = valid_data._getTensorLabels().numpy()
test_labels  = test_data._getTensorLabels().numpy()
print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
print('Validation labels: ', [(lb, np.sum(valid_labels == lb)) for lb in set(valid_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')

# prepare controlling variables
rule_list   = []
cover_list  = []

train_data_cover_count  = np.zeros(train_data.__len__())
valid_data_cover_count  = np.zeros(valid_data.__len__())
test_data_cover_count   = np.zeros(test_data.__len__())

coverage_ratio_train = 0

cover_indi_train = np.zeros(train_data.__len__(), dtype='bool')

if is_svd_plot:
    psvd = PlotSVD(train_data._getArrayFeatures())

# start iteration for training
if not is_load_dill:
    iter_count = 0
    np.random.seed(0)

    for i in np.random.permutation(train_data_cover_count.shape[0]):

        if train_data_cover_count[i] > 0:
            continue

        print('====================== idx = %d; iter = %d/%d; tgt label: %d ======================'
              % (i, iter_count, train_data.__len__(), train_data._getTensorLabels()[i]))
        iter_count += 1

        peel_mask = train_data_cover_count > 0
        inv_classifier = rule_miner_train.getInvariantClassifier(i, org_train_labels, delta_constr_=0)

        # show SVD of remaining data
        if is_svd_plot:
            psvd.drawPoints(train_data._getArrayFeatures()[train_data_cover_count<=0, :],
                            train_data._getArrayLabels()[train_data_cover_count<=0],
                            ('svd_' + str(iter_count) + '_' + str(i) + '.jpg'))

        if inv_classifier is None:
            train_data_cover_count[i] += 1
            continue

        print('num_boundaries: %d\n' % (inv_classifier.getNumBoundaries()))

        pred_labels, cover_indi_train   = inv_classifier.classify(train_data._getArrayFeatures())
        label_diff                      = pred_labels[cover_indi_train] - train_data._getArrayLabels()[cover_indi_train]
        accuracy_train                  = np.sum(label_diff == 0)/label_diff.size
        num_samples_train               = np.sum(cover_indi_train)
        print('Training ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_train, num_samples_train))

        #print(np.where(cover_indi_train)[0])

        spt_score = np.sum(pred_probs[cover_indi_train])
        inv_classifier.updateSupportScore(spt_score)

        pred_labels, cover_indi_valid   = inv_classifier.classify(valid_data._getArrayFeatures())
        label_diff                      = pred_labels[cover_indi_valid] - valid_data._getArrayLabels()[cover_indi_valid]
        accuracy_valid                  = np.sum(label_diff == 0) / label_diff.size
        num_samples_valid               = np.sum(cover_indi_valid)
        print('Validation ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_valid, num_samples_valid))

        pred_labels, cover_indi_test    = inv_classifier.classify(test_data._getArrayFeatures())
        label_diff                      = pred_labels[cover_indi_test] - test_data._getArrayLabels()[cover_indi_test]
        accuracy_test                   = np.sum(label_diff == 0)/label_diff.size
        num_samples_test                = np.sum(cover_indi_test)
        print('Testing ==> ACU: %.3f\tnum_samples: %d\n' % (accuracy_test, num_samples_test))

        # we cheat here by updating only when test accuracy is good
        num_newly_covered_samples = np.sum(train_data_cover_count[cover_indi_train] <= 0)

        if accuracy_valid >= org_test_acu:
        #if accuracy_test >= 0:
        #if num_newly_covered_samples >= 0:
        #if True:
            # update cover count
            train_data_cover_count[cover_indi_train] += 1
            valid_data_cover_count[cover_indi_valid] += 1
            test_data_cover_count[cover_indi_test] += 1

            # visualize
            if is_2d_data:
                covered_train_data = train_data._getTensorFeaturesByIndi(cover_indi_train).numpy()
                covered_train_label = train_data._getTensorLabelsByIndi(cover_indi_train).numpy()
                pivot = train_data._getTensorFeaturesByIndi(i).numpy()

                print('pivot position: ', pivot)

                Plot2D().drawPointsWithPivotAndLines(pivot, covered_train_data, covered_train_label, inv_classifier._bb.bb,
                                         ('covered_traindata_' + str(iter_count - 1) + '_' + str(train_data.getSingleLabel(i)) + '.jpg'))

            # update rule_list and cover_list with the same order of list items
            rule_list.append(inv_classifier)
            cover_list.append(cover_indi_train)

        num_covered_total_train = np.sum(train_data_cover_count > 0)
        coverage_ratio_train    = num_covered_total_train/train_data_cover_count.__len__()
        print('Training ==> num covered samples: %d\t coverage ratio: %.3f\n' % (num_covered_total_train, coverage_ratio_train))

        num_covered_total_valid = np.sum(valid_data_cover_count > 0)
        coverage_ratio_valid    = num_covered_total_valid/valid_data_cover_count.__len__()
        print('Validation ==> num covered samples: %d\t coverage ratio: %.3f\n' % (num_covered_total_valid, coverage_ratio_valid))

        num_covered_total_test = np.sum(test_data_cover_count > 0)
        coverage_ratio_test    = num_covered_total_test / test_data_cover_count.__len__()
        print('Testing ==> num covered samples: %d\t coverage ratio: %.3f\n' % (num_covered_total_test, coverage_ratio_test))

        print('Num of candidate rules: %d\n' % (rule_list.__len__()))


        if is_2d_data:
            uncovered_train_data = train_data._getTensorFeaturesByIndi(train_data_cover_count <= 0).numpy()
            uncovered_train_label = train_data._getTensorLabelsByIndi(train_data_cover_count <= 0).numpy()
            Plot2D().drawPoints(uncovered_train_data, uncovered_train_label, ('uncovered_traindata_' + str(iter_count - 1) + '_' + str(train_data.getSingleLabel(i)) + '.jpg'))



        if iter_count >= 100 or coverage_ratio_train >= 0.99:
        #if coverage_ratio_train >= 0.5:
            break

        #break

    #dill.dump_session(dill_filename)
else:
    dill.load_session(dill_filename)

'''
# start set cover selection of rules
cover_matrix = np.array(cover_list).T
covered_indi = np.zeros(cover_matrix.shape[0], dtype='bool')
sel_indi     = np.zeros(cover_matrix.shape[1], dtype='bool')

iter = 0
while True:
    iter += 1
    # greedy find the best rule
    row_sum = np.sum(cover_matrix, axis=0)

    max_idx = np.argmax(row_sum)

    if iter % 500 == 0:
        print('iter: %d\tmax_val: %d\ttotal_covered: %d/%d\n' % (iter, row_sum[max_idx], np.sum(covered_indi), covered_indi.size))

    if row_sum[max_idx] <= 0:
        break

    # update sel_indi and covered_indi
    sel_indi[max_idx] = True
    covered_indi = (covered_indi | cover_matrix[:, max_idx])

    # remove selected samples
    cover_matrix[covered_indi, :] = False

num_selected_rules  = np.sum(sel_indi)
num_all_rules       = sel_indi.__len__()
sel_ratio           = num_selected_rules/num_all_rules
print('num of selected rules: %d/%d\tratio: %.3f' % (num_selected_rules, num_all_rules, sel_ratio))
'''

# test the accuracy of the selected set of rules
from queue import PriorityQueue as PQ
pred_votes = {}
tnsr_feat_test  = test_data._getArrayFeatures()
tnsr_label_test = test_data._getArrayLabels()

#for rule_idx in np.where(sel_indi)[0]:
#    inv_classifier = rule_list[rule_idx]
for inv_classifier in rule_list:

    pred_labels, cover_indi = inv_classifier.classify(tnsr_feat_test)
    spt = inv_classifier._spt_score

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
print('Test accuracy: %.3f\tnum_covered: %d\tnum_test_samples: %d\t' % (test_accuracy, num_covered, tnsr_label_test.shape[0]))

print('org_test_acu: %.3f' % (org_test_acu))

#print('num of selected rules: %d/%d\tratio: %.3f' % (num_selected_rules, num_all_rules, sel_ratio))

# draw for 2d data
if is_2d_data:
    Plot2D().drawPointsWithPivotAndLines(np.array([1.5848259, -0.8277271]),
                                         train_data._getTensorFeatures().numpy(),
                                         train_data._getTensorLabels().numpy(),
                                         rule_miner_train._bb.bb[:, -1],
                                         './train2D_with_gtline.jpg')

    Plot2D().drawPoints(train_data._getTensorFeatures().numpy(), train_data._getTensorLabels().numpy(), './train2D.jpg')
    Plot2D().drawPoints(valid_data._getTensorFeatures().numpy(), valid_data._getTensorLabels().numpy(), './valid2D.jpg')
    Plot2D().drawPoints(test_data._getTensorFeatures().numpy(), test_data._getTensorLabels().numpy(), './test2D.jpg')

    uncovered_train_data = train_data._getTensorFeaturesByIndi(train_data_cover_count <= 0).numpy()
    uncovered_train_label= train_data._getTensorLabelsByIndi(train_data_cover_count <= 0).numpy()
    Plot2D().drawPoints(uncovered_train_data, uncovered_train_label, 'uncovered_traindata.jpg')

    uncovered_valid_data = valid_data._getTensorFeaturesByIndi(valid_data_cover_count <= 0).numpy()
    uncovered_valid_label= valid_data._getTensorLabelsByIndi(valid_data_cover_count <= 0).numpy()
    Plot2D().drawPoints(uncovered_valid_data, uncovered_valid_label, 'uncovered_validdata.jpg')

    uncovered_test_data  = test_data._getTensorFeaturesByIndi(test_data_cover_count <= 0).numpy()
    uncovered_test_label = test_data._getTensorLabelsByIndi(test_data_cover_count <= 0).numpy()
    Plot2D().drawPoints(uncovered_test_data, uncovered_test_label, 'uncovered_testdata.jpg')

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









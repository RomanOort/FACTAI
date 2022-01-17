

from algorithms.mine_cnn_invariant import *
from dnn_invariant.utilities.datasets import *
from sklearn import tree

# load data
train_loader    = DataLoader(train_data, batch_size=1, shuffle=False)
test_loader     = DataLoader(test_data, batch_size=1, shuffle=False)

# load model
model = CNN4Invariant(num_classes_=2).cuda()
#model = MLP4Invariant(num_classes_=2).cuda()

model.loadModel()

layer_ptr_list = [1, 3, 5]

#====================================
# do test
#====================================
input_idx = 1311

#model._sanityCheck4CNN4Invariant(train_data.getCNNFeature(input_idx).cuda(), 1)

m_inv_miner = InvariantMiner(train_data, model, layer_ptr_list)
hashkeys, bb = m_inv_miner._getHashKeysOfAllData4OneInput(input_idx)


# compute top 100 nearest neighbors of input_idx
'''
all_features = train_data._getTensorFeatures()
distance_list = []
for i in range(0, all_features.size(0)):
    diff = all_features[i,:] -all_features[input_idx,:]
    distance = torch.sum(torch.mul(diff, diff)).item()

    distance_list.append(distance)

dist_tensor = torch.FloatTensor(distance_list)

_, index = torch.topk(dist_tensor, k=11900, largest=False)

hashkeys = hashkeys[index,:]

packed_hk = np.packbits(hashkeys, axis=0)

str_hk_dict = {}
for i in range(0, packed_hk.shape[1]):
    str_hk_i = packed_hk[:, i].view('c').tostring()

    if(str_hk_i in str_hk_dict):
        str_hk_dict[str_hk_i] = str_hk_dict[str_hk_i] + 1
        print('got: %d' % (str_hk_dict[str_hk_i]))
    else:
        str_hk_dict[str_hk_i] = 1

num_unique_keys = str_hk_dict.__len__()
redundant_ratio = (packed_hk.shape[1] - num_unique_keys)/packed_hk.shape[1]

print('num unique keys: %d/%d, redundant ratio: %.4f' % (num_unique_keys, packed_hk.shape[1], redundant_ratio))
'''

# start doing linear regression
all_features = train_data._getTensorFeatures()
all_labels = train_data._getTensorLabels()
neg_index  = (all_labels != all_labels[input_idx]).nonzero().squeeze()

#sample_index = torch.cat((torch.tensor([input_idx]), neg_index))
sample_index = range(0,12000)

sample_features = all_features[sample_index, :].numpy()
sample_labels   = all_labels[sample_index].numpy()
sample_hashkeys = hashkeys[sample_index, :].numpy()

'''
print('start logistic regression ...')
clf = LogisticRegression(random_state=0, solver='saga', multi_class='auto', max_iter=3000, penalty='l1').fit(sample_hashkeys, sample_labels)

score = clf.score(sample_hashkeys, sample_labels)
print('score: %.3f' % (score))
print('num non-zero coef: %d' % (np.sum(coef != 0)))
'''

clf = tree.DecisionTreeClassifier().fit(sample_hashkeys, sample_labels)
print(tree.export.export_text(clf))

# do testing
all_features_test = test_data._getTensorFeatures().cuda()
all_labels_test   = test_data._getTensorLabels()

test_hashvals = bb.computeHashVal(all_features_test)
test_hashkeys = (test_hashvals > 0).cpu().numpy()

test_pred_labels = clf.predict(test_hashkeys)

accuracy = np.sum(all_labels_test.numpy() == test_pred_labels)/test_pred_labels.shape[0]
print('ACU: %.4f' % (accuracy))




'''
for input_idx in range(0, train_data.__len__()):
    model._getHashKeysOfAllData4OneInput(train_data, input_idx, [0, 2, 4])
'''

'''
for step, (instance, label) in enumerate(train_loader):
    cnnfea   = train_data.getCNNFeature(step)
    tnsrfeas = train_data._getTensorFeatures()

    diff1 = torch.sum(torch.abs(cnnfea.squeeze_().reshape(1, -1) - tnsrfeas[step].squeeze_()))

    diff2 = torch.sum(torch.abs(cnnfea - instance))

    if(diff1 > 1e-10):
        print('WRONG 1 !')

    if(diff2 > 1e-10):
        print('WRONG 2 !')
'''

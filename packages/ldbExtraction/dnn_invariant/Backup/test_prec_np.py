import numpy as np
import pickle as pkl

train_mdata_, bb, invariant = pkl.load(open('snapshot.pkl', 'rb'))

tnsr_features_ = train_mdata_._getTensorFeatures().squeeze().cpu().numpy()
basis = bb.getBasisTensor().squeeze().cpu().numpy()
bias  = bb.getBiasTensor().squeeze().cpu().numpy()
sub_basis = basis[:, invariant]
sub_bias  = bias[invariant]

print('num boundaries: ', sub_bias.shape)


'''
invariant_2 = np.zeros(6452, dtype='bool')
dice = np.random.rand(6452)
invariant_2[dice < 0.8] = True
#invariant_2[invariant_1] = True

print(np.sum(invariant_2))

invariant = invariant_2

tnsr_features_ = np.random.rand(10808, 784)
basis = np.random.rand(784, 6452) - 0.5
bias  = np.random.rand(6452) - 0.5
sub_basis = basis[:, invariant]
sub_bias  = bias[invariant]
'''

print('=============================')

print(tnsr_features_.dtype, tnsr_features_.shape)
print(basis.dtype, basis.shape)
print(bias.dtype, bias.shape)
print(sub_basis.dtype, sub_basis.shape)
print(sub_bias.dtype, sub_bias.shape)

hashval = np.matmul(tnsr_features_, basis)
sub_hashval = hashval[:, invariant]
sub_config  = np.zeros(sub_hashval.shape)
sub_config[sub_hashval > 0] = 1

sub_basis = basis[:, invariant]
sub_bias  = bias[invariant]
new_sub_hashval = np.matmul(tnsr_features_, sub_basis)
new_sub_config = np.zeros(new_sub_hashval.shape)
new_sub_config[new_sub_hashval > 0] = 1

diff1 = np.sum(np.abs(sub_hashval - new_sub_hashval))
print('diff1: ', diff1)

diff2 = np.sum(np.abs(sub_config - new_sub_config))
print('diff2: ', diff2)

diff_mat = np.abs(sub_hashval - new_sub_hashval)
print(np.amax(diff_mat))

'''
print(np.amin(sub_hashval.numpy()), np.amax(sub_hashval.numpy()))
print(np.amin(new_sub_hashval.numpy()), np.amax(new_sub_hashval.numpy()))
'''

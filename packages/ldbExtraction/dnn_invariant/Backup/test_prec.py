
import torch
import numpy as np
import pickle as pkl

'''
train_mdata_, bb, invariant_1 = pkl.load(open('snapshot.pkl', 'rb'))

invariant = np.where(invariant_1)[0]

tnsr_features_ = train_mdata_._getTensorFeatures().cuda()
basis = bb.getBasisTensor()
bias  = bb.getBiasTensor()
sub_basis = basis[:, :, invariant]
sub_bias  = bias[:, invariant]

print('num boundaries: ', sub_bias.shape)

'''

invariant_2 = np.zeros(6452, dtype='bool')
dice = np.random.rand(6452)
invariant_2[dice < 0.01] = True
#invariant_2[invariant_1] = True

print(np.sum(invariant_2))

invariant = invariant_2

tnsr_features_ = torch.rand(10808, 784).cuda().squeeze()
basis = (torch.rand(1, 784, 6452) - 0.5).cuda().squeeze()
bias  = (torch.rand(1, 6452) - 0.5).cuda().squeeze()
sub_basis = basis[:, invariant]
sub_bias  = bias[invariant]


print('=============================')

print(tnsr_features_.dtype, tnsr_features_.shape)
print(basis.dtype, basis.shape)
print(bias.dtype, bias.shape)
print(sub_basis.dtype, sub_basis.shape)
print(sub_bias.dtype, sub_bias.shape)

hashval = torch.mm(tnsr_features_, basis)
sub_hashval = hashval[:, invariant]
sub_config  = np.zeros(sub_hashval.shape)
sub_config[sub_hashval.cpu().numpy() > 0] = 1

sub_basis = basis[:, invariant]
sub_bias  = bias[invariant]
new_sub_hashval = torch.mm(tnsr_features_, sub_basis)
new_sub_config = np.zeros(new_sub_hashval.shape)
new_sub_config[new_sub_hashval.cpu().numpy() > 0] = 1

diff1 = np.sum(np.abs(sub_hashval.cpu().numpy() - new_sub_hashval.cpu().numpy()))
print('diff1: ', diff1)

diff2 = np.sum(np.abs(sub_config - new_sub_config))
print('diff2: ', diff2)

diff_mat = np.abs(sub_hashval.cpu().numpy() - new_sub_hashval.cpu().numpy())
print(np.amax(diff_mat))

'''
print(np.amin(sub_hashval.numpy()), np.amax(sub_hashval.numpy()))
print(np.amin(new_sub_hashval.numpy()), np.amax(new_sub_hashval.numpy()))
'''

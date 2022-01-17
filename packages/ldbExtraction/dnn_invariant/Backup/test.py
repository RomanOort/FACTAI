
import torch
import numpy as np
import pickle as pkl

'''
train_mdata_, bb, invariant_1 = pkl.load(open('snapshot.pkl', 'rb'))


print(invariant.shape, invariant.dtype, type(invariant))


tnsr_features_ = train_mdata_._getTensorFeatures().cuda()
basis = bb.getBasisTensor()
bias  = bb.getBiasTensor()
sub_basis = basis[:, :, invariant]
sub_bias  = bias[:, invariant]
'''
np.random.seed(0)
invariant_2 = np.random.permutation(6452)
#dice = np.random.rand(6452)
#invariant_2[dice < 0.01] = 1
#invariant_2[invariant_1] = True
print(torch.__version__)
print(np.sum(invariant_2))


invariant0 = invariant_2[:10]
invariant1 = invariant_2[:129]

tnsr_features_ = torch.rand(10808, 784).cuda()
basis = (torch.rand(1, 784, 6452) - 0.5).cuda()
bias  = (torch.rand(1, 6452) - 0.5).cuda()
sub_basis = basis[:, :, invariant0]
sub_bias  = bias[:, invariant0]

sub_basis_ = basis[:, :, invariant1]
sub_bias_  = bias[:, invariant1]

print('=============================')

print(tnsr_features_.dtype, tnsr_features_.shape)
print(basis.dtype, basis.shape)
print(bias.dtype, bias.shape)
print(sub_basis.dtype, sub_basis.shape)
print(sub_bias.dtype, sub_bias.shape)

hashval = torch.matmul(tnsr_features_, basis)
sub_hashval = hashval[:, :, invariant0]
sub_hashval_ = hashval[:, :, invariant1]
sub_config  = np.zeros(sub_hashval.shape)
sub_config[sub_hashval.cpu().numpy() > 0] = 1

sub_basis = basis[:, :, invariant0]
sub_basis_ = basis[:, :, invariant1]
sub_bias  = bias[:, invariant0]
sub_bias_  = bias[:, invariant1]
new_sub_hashval = torch.matmul(tnsr_features_, sub_basis)
new_sub_hashval_ = torch.matmul(tnsr_features_, sub_basis_)
error = np.sum(np.abs(new_sub_hashval[:,:,:2].cpu().numpy() - new_sub_hashval_[:,:,:2].cpu().numpy()))
print("=================")
print('error:', error)
print("==================")


new_sub_config = np.zeros(new_sub_hashval.shape)
new_sub_config_ = np.zeros(new_sub_hashval_.shape)
new_sub_config[new_sub_hashval.cpu().numpy() > 0] = 1
new_sub_config_[new_sub_hashval_.cpu().numpy() > 0] = 1
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

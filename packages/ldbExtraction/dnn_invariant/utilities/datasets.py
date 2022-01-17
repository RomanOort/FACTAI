

import torch
from torch.utils.data import Dataset
import torch
import codecs
import numpy as np
import matplotlib.pyplot as plt
from dnn_invariant.utilities.trainer import *
from dnn_invariant.utilities.environ import *

H = 224
W = 224

class mDataSet(Dataset):
    def __init__(self, data_path_):
        self._data_path = data_path_
        self._data = torch.load(self._data_path)

        self._global_idx = np.arange(self.__len__())
        self._global_N   = self.__len__()

    def __getitem__(self, index):
        instances   = self._data[0][index].float()
        labels      = self._data[1][index]

        return instances, labels

    def __len__(self):
        return len(self._data[1])

    # for personal use
    def _getRawData(self):
        return self._data

    def _getTensorFeatures(self):
        instances = self._data[0].float()
        instances = instances.reshape(instances.size(0), -1)
        return instances

    def _getArrayFeatures(self):
        return self._getTensorFeatures().cpu().numpy()

    def _getNNIndi(self, radius_, center_idx_):
        array_features = self._getArrayFeatures()

        dist = np.sqrt(np.sum(np.power((array_features - array_features[center_idx_,:]), 2), axis=1))

        #nn = self.__len__() - 1
        nn = np.min([500, self.__len__()-1])
        thres = np.partition(dist, nn)[nn]
        if radius_ < thres:
            radius_ = thres

        return dist <= radius_, thres

    def _getTensorFeaturesByIndi(self, indi_):
        tnsr_feats = self._getTensorFeatures()
        return tnsr_feats[indi_]

    def _getTensorLabels(self):
        return self._data[1].long()

    def _getArrayLabels(self):
        return self._getTensorLabels().cpu().numpy()

    def _getTensorLabelsByIndi(self, indi_):
        tnsr_labels = self._getTensorLabels()
        return tnsr_labels[indi_]

    def getCNNFeature(self, index):
        instance = self._data[0][index].float()
        unsqueezed_instance = instance.unsqueeze_(0)

        return unsqueezed_instance

    def loadHardCopy(self):
        return mDataSet(self._data_path)

    def dumpByIndi(self, indi, filename=None):
        if filename == None:
            filename = self._data_path + '.dump'

        X = self._data[0][indi, :]
        y = self._data[1][indi]

        with open(filename, 'wb') as fid:
            torch.save((X, y), fid)
            print('save to ' + filename)


    # test utilities
    def getRange(self, dim_, label_):
        indi = (self._data[1] == label_)
        return (torch.min(self._data[0][indi, dim_]), torch.max(self._data[0][indi, dim_]))

    def updateLabels(self, model_):
        _, pred_labels, _ = Trainer.evalAccuracyOfModel(model_, self)
        self._data = (self._data[0], torch.tensor(pred_labels))

    def addGaussianNoise(self, std_):
        gaussian_mean = torch.zeros(1, 3, H, W)
        gaussian_std = torch.ones(1, 3, H, W) * std_
        gaussian_noise = torch.normal(mean=gaussian_mean, std=gaussian_std)

        data_loader = DataLoader(self, batch_size=30, shuffle=False)

        initial = True
        with torch.no_grad():
            for instances, _ in data_loader:
                instances = instances
                batch_outputs = instances + gaussian_noise
                if initial == True:
                    outputs = batch_outputs
                    initial = False
                else:
                    outputs = torch.cat((outputs, batch_outputs))

        self._data = (outputs.cpu(), self._data[1])

    def updateData(self, model_, layer_start):

        data_loader = DataLoader(self, batch_size=30, shuffle=False)

        initial = True
        model_.cuda()
        model_.eval()
        with torch.no_grad():
            for instances, _ in data_loader:
                instances = instances.cuda()
                batch_outputs = model_._getOutputOfOneLayer_Group(instances, layer_start, 0)
                if initial == True:
                    outputs = batch_outputs
                    initial = False
                else:
                    outputs = torch.cat((outputs, batch_outputs))

        #data = self._data[0].view(self._global_N, 1, 3, H, W)

        self._data = (outputs.cpu(), self._data[1])

    def removeData(self, indi_rmv_):
        self._data = (self._data[0][~indi_rmv_], self._data[1][~indi_rmv_])
        self._global_idx = self._global_idx[~indi_rmv_]

    def covertGlobalIndi(self, local_indi_):
        global_idx = self._global_idx[local_indi_]
        global_indi = np.zeros(self._global_N, dtype='bool')
        global_indi[global_idx] = True

        return global_indi

    def expandByRand(self, scale_, times_, model_):
        new_feat = self._data[0]
        for i in range(times_):
            perturb_data = self._data[0] + scale_*(torch.rand(self._data[0].size()) - 0.5)
            new_feat = torch.cat((new_feat, perturb_data), 0)

        rand_label = torch.rand(new_feat.size(0))

        self._data = (new_feat, rand_label)

        self._global_idx = np.arange(self.__len__())
        self._global_N   = self.__len__()

        self.updateLabels(model_)

    def rmvDataByChgLabels(self, indi_rmv_):
        new_labels = self._data[1]
        new_labels[indi_rmv_] = -1
        self._data = (self._data[0], new_labels)

    def getSingleLabel(self, index_):
        return self._data[1][index_].numpy()



# The following code loads the training and testing data set
# Load data by import this py file
# this makes sure different .py file are using the same data sets
'''
train_data      = mDataSet('./data/RAND_2D/RAND_2D_Train.ntvt')
valid_data      = mDataSet('./data/RAND_2D/RAND_2D_Valid.ntvt')
test_data       = mDataSet('./data/RAND_2D/RAND_2D_Test.ntvt')
'''

'''
train_data      = mDataSet('./data/FMNIST/FMNIST_234_Valid.ntvt')
#train_data      = mDataSet('./data/FMNIST/FMNIST_234_Train.ntvt')
valid_data      = mDataSet('./data/FMNIST/FMNIST_234_Valid.ntvt')
test_data       = mDataSet('./data/FMNIST/FMNIST_234_Test.ntvt')
'''


'''
train_data      = mDataSet('./data/FMNIST/FMNIST_24_Train.ntvt')
valid_data      = mDataSet('./data/FMNIST/FMNIST_24_Valid.ntvt')
test_data       = mDataSet('./data/FMNIST/FMNIST_24_Test.ntvt')
'''

'''
train_data      = mDataSet('./data/FMNIST/FMNIST_89_Train.ntvt')
valid_data      = mDataSet('./data/FMNIST/FMNIST_89_Valid.ntvt')
test_data       = mDataSet('./data/FMNIST/FMNIST_89_Test.ntvt')
'''


#train_data      = mDataSet('./data/FMNIST/FMNIST_29_Train.ntvt')
#valid_data      = mDataSet('./data/FMNIST/FMNIST_29_Valid.ntvt')
#test_data       = mDataSet('./data/FMNIST/FMNIST_29_Test.ntvt')


'''
train_data      = mDataSet('./data/CIFAR10/CIFAR10_04_Train.tvt')
valid_data      = mDataSet('./data/CIFAR10/CIFAR10_04_Valid.tvt')
test_data       = mDataSet('./data/CIFAR10/CIFAR10_04_Test.tvt')
'''

#train_data      = mDataSet('./data/CIFAR10/CIFAR10_29_Train.ntvt')
#valid_data      = mDataSet('./data/CIFAR10/CIFAR10_29_Valid.ntvt')
#test_data       = mDataSet('./data/CIFAR10/CIFAR10_29_Test.ntvt')

#train_data      = mDataSet('./data/CIFAR10/CIFAR10_01_Train.ntvt')
#valid_data      = mDataSet('./data/CIFAR10/CIFAR10_01_Valid.ntvt')
#test_data       = mDataSet('./data/CIFAR10/CIFAR10_01_Test.ntvt')







'''
Path to load the dataset
'''
# train_data      = mDataSet('./data/Assira_Train.ntvt')
# test_data       = mDataSet('./data/Assira_Test.ntvt')
#train_data = mDataSet('./data/ZhangLab_Train.ntvt')
#test_data = mDataSet('./data/ZhangLab_Test.ntvt')
#train_data = mDataSet('./data/FMNIST_24_Train.tvt')#
#test_data = mDataSet('./data/FMNIST_24_Test.tvt')
#train_data = mDataSet('./data/MNIST_24_Train.tvt')
#test_data = mDataSet('./data/MNIST_24_Test.tvt')



'''
Size of the visualization images: For MNIST/FMNIST, recommend 100. For natural iamges, recommend 600.
'''
#output_size = 100
output_size = 600



'''
Dimension of the images. Nothing need to be changed
'''
# C = train_data._data[0].shape[1]
# W = train_data._data[0].shape[2]
# H = train_data._data[0].shape[3]



'''
The name of the classes. It will amostly ffect the log and filenames for visualization images
'''
#class_list = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
class_list = ['cats', 'dogs']
#class_list = ['Two', 'Four']
#class_list = ['Pullover', 'Coat']



'''
Whether the data uses ImageNet Normalization
'''
ImageNetNormalize = True
#ImageNetNormalize = False













'''
import matplotlib.pyplot as plt
fig = plt.figure()
img = train_data.getCNNFeature(0).squeeze().cpu().numpy()
img = np.moveaxis(img, 0, -1)
plt.imshow(img/255)
plt.savefig('img.jpg')
plt.close(fig)
'''

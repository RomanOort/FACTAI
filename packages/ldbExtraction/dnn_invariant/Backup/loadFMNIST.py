
import torch
import numpy as np

class LoadFMNIST():
    def __init__(self, class_ids_=None, validation_ratio_=0.0, save_root_='./data/FMNIST', isNormalized_=False):
        self._save_pathheader = save_root_ + '/FMNIST' # the root path to save transformed loaded data
        self._isNormalized    = isNormalized_

        if(self._isNormalized == True):
            self._file_suffix = '.ntvt'
        else:
            self._file_suffix = '.tvt'

        # load all raw data
        X_train, y_train      = self.load_mnist('./data/FMNIST/RAW', kind='train')
        X_test, y_test        = self.load_mnist('./data/FMNIST/RAW', kind='t10k')

        # transform to torch.tensor
        X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        X_test  = torch.from_numpy(X_test).type(torch.FloatTensor)
        y_test  = torch.from_numpy(y_test).type(torch.LongTensor)

        X_train = X_train.view(X_train.size(0), 1, 28, 28)
        X_test  = X_test.view(X_test.size(0), 1, 28, 28)

        # divid X_train and y_train into _new_train and _new_validation
        num_validation = int(X_train.shape[0] * validation_ratio_)
        assert num_validation < X_train.shape[0]
        rand_idx = np.random.permutation(X_train.shape[0])

        X_new_train = X_train[rand_idx[num_validation:X_train.shape[0]]]
        y_new_train = y_train[rand_idx[num_validation:X_train.shape[0]]]

        X_new_valid = X_train[rand_idx[:num_validation]]
        y_new_valid = y_train[rand_idx[:num_validation]]

        print('org_train mean: ', X_new_train.mean())

        if (self._isNormalized == True):
            mean = X_new_train.mean()
            std = X_new_train.std()

            X_new_train = 1.0 * (X_new_train - mean) / std
            X_new_valid = 1.0 * (X_new_valid - mean) / std
            X_test      = 1.0 * (X_test - mean) / std

            print('normlaized_train mean: ', X_new_train.mean())

        # save data
        if class_ids_ == None:
            save_path_train = self._save_pathheader + '_all_Train' + self._file_suffix
            save_path_test  = self._save_pathheader  + '_all_Test' + self._file_suffix

            with open(save_path_train, 'wb') as fid:
                torch.save((X_train, y_train), fid)
                print('save to ' + save_path_train)
            with open(save_path_test, 'wb') as fid:
                torch.save((X_test, y_test), fid)
                print('save to' + save_path_test)
        else:
            idx_new_train    = self._getClassIdx(y_new_train, class_ids_)
            idx_new_valid    = self._getClassIdx(y_new_valid, class_ids_)
            idx_test         = self._getClassIdx(y_test, class_ids_)

            save_path_train = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Train' + self._file_suffix
            save_path_valid = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Valid' + self._file_suffix
            save_path_test  = self._save_pathheader + '_' + ''.join(map(str, class_ids_)) + '_Test' + self._file_suffix

            self._export(idx_new_train, X_new_train, y_new_train, save_path_train)
            self._export(idx_new_valid, X_new_valid, y_new_valid, save_path_valid)
            self._export(idx_test, X_test, y_test, save_path_test)

    def _export(self, idx_, X_, y_, save_path_):
        with open(save_path_, 'wb') as fid:
            num_ = 0
            for idx_cls in idx_:
                num_ += idx_cls.shape[0]

            y_ = torch.zeros(num_, dtype=torch.long)
            ptr = 0
            for i in range(idx_.__len__()):
                y_[ptr:ptr + idx_[i].__len__()] = i
                ptr += idx_[i].__len__()

            idx_ = np.concatenate(idx_, axis=0)
            X_ = X_[idx_]

            torch.save((X_, y_), fid)
            print('save to' + save_path_)


    def _getClassIdx(self, y, class_ids_):
        idx = []
        for cls in class_ids_:
            idx.append(np.where(y == cls)[0])

        return idx

    # This load_mnist method is directly copied from utils/mnist_reader.py
    # from https://github.com/zalandoresearch/fashion-mnist
    def load_mnist(self, path, kind='train'):
        import os
        import gzip
        import numpy as np

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

# ============================================
# Execute to make data
# ============================================

LoadFMNIST([2, 4], isNormalized_=True)
LoadFMNIST([2, 4], isNormalized_=False)

LoadFMNIST([2, 3, 4], isNormalized_=True)
LoadFMNIST([2, 3, 4], isNormalized_=False)

'''
LoadFMNIST(isNormalized_=True)
LoadFMNIST(isNormalized_=False)
'''

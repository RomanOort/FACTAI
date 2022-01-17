
from dnn_invariant.utilities.environ import *

class loadRAND_2D():
    def __init__(self, num_samples_train_=5000, num_samples_valid_ = 1000, num_samples_test_ = 2000, isNormalized_=False, save_root_='./data/RAND_2D'):
        self._save_pathheader = save_root_ + '/RAND_2D_'  # the root path to save transformed loaded data

        if(isNormalized_ == True):
            self._file_suffix = '.ntvt'
        else:
            self._file_suffix = '.tvt'

        X_train = torch.rand(num_samples_train_, 1, 1, 2, dtype=torch.float32)
        X_valid = torch.rand(num_samples_valid_, 1, 1, 2, dtype=torch.float32)
        X_test  = torch.rand(num_samples_test_, 1, 1, 2, dtype=torch.float32)

        center  = X_train.mean(dim=0)
        radius  = 0.4 * (X_train.max() - X_train.min())

        y_train = self._genLabelsByCircle(X_train, center, radius)
        y_valid = self._genLabelsByCircle(X_valid, center, radius)
        y_test  = self._genLabelsByCircle(X_test, center, radius)

        if isNormalized_:
            mean = X_train.mean()
            std  = X_train.std()

            X_train = 1.0 * (X_train - mean) / std
            X_valid = 1.0 * (X_valid - mean) / std
            X_test  = 1.0 * (X_test - mean)  / std

        # save data
        save_path_train = self._save_pathheader + 'Train' + self._file_suffix
        save_path_valid = self._save_pathheader + 'Valid' + self._file_suffix
        save_path_test  = self._save_pathheader + 'Test'  + self._file_suffix

        self._export(X_train, y_train, save_path_train)
        self._export(X_valid, y_valid, save_path_valid)
        self._export(X_test, y_test, save_path_test)

    def _genLabelsByCircle(self, X_, center_, radius_):
        # compute L2Dist between X and center_
        dist = torch.norm(X_.squeeze() - center_.squeeze(), dim=1)
        y_ = torch.zeros(dist.size(0), dtype=torch.long)
        y_[dist < radius_] = 1

        return y_

    def _export(self, X_, y_, save_path_):
        with open(save_path_, 'wb') as fid:
            torch.save((X_, y_), fid)
            print('save to' + save_path_)

loadRAND_2D(isNormalized_=True)
loadRAND_2D(isNormalized_=False)


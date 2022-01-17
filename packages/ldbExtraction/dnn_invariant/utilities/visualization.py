
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib as mpl
import matplotlib.pyplot as plt
from dnn_invariant.utilities.environ import *
from dnn_invariant.utilities.datasets import *

class PlotSVD():
    def __init__(self, X_):
        assert X_.shape[-1] == 784

        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=3)

        self.pca.fit(X_)

    def _drawScatter(self, data_, label_=None, filename_='./tmp_points.jpg', is_limit_axis = True):
        self.style_list = [('r', '.'), ('b', '.'), ('g', '.')]

        fig = plt.figure()

        if label_ is None:
            plt.scatter(data_[:, 0], data_[:, 1], data_[:, 2], c='b', marker='p')

        else:
            for i in np.unique(label_):
                idx = label_ == i
                plt.scatter(data_[idx, 0], data_[idx, 1], data_[idx, 2],
                            c=self.style_list[i][0], marker=self.style_list[i][1])

        if is_limit_axis == True:
            plt.axis([-2, 2, -2, 2])
        # plt.axis('square')
        plt.savefig(filename_)

        plt.close(fig)

    def drawPoints(self, X_, labels_, filename_):
        data = self.pca.transform(X_)
        self._drawScatter(data, labels_, filename_, False)

class Plot2D():
    def _computeLinePoints(self, bb_, x_):
        assert (np.sum(np.abs(bb_)) > 0)

        '''
        if bb_[0] == 0:
            bb_[0] += 1e-3

        if bb_[1] == 0:
            bb_[1] += 1e-3
        '''

        y = (-bb_[0]*x_ - bb_[2])/bb_[1]
        return y

    def _drawLines(self, bb_):
        x1 = -2
        x2 = 2
        y1 = self._computeLinePoints(bb_, x1)
        y2 = self._computeLinePoints(bb_, x2)

        plt.plot([x1, x2], [y1, y2])

        return False

    def drawPoints(self, data_, label_=None, filename_='./tmp_points.jpg', is_limit_axis = True):
        self.style_list = [('r', '.'), ('b', '.'), ('g', '.')]

        data_ = data_.squeeze()
        assert data_.shape.__len__() == 2 and data_.shape[1] == 2

        fig = plt.figure()

        if label_ is None:
            plt.scatter(data_[:, 0], data_[:, 1], c='b', marker='p')

        else:
            for i in np.unique(label_):
                idx = label_ == i
                plt.scatter(data_[idx, 0], data_[idx, 1],
                            c=self.style_list[i][0], marker=self.style_list[i][1])

        if is_limit_axis == True:
            plt.axis([-2, 2, -2, 2])
        #plt.axis('square')
        plt.savefig(filename_)

        plt.close(fig)

    def drawLines(self, bb_=None, filename_='./tmp_lines.jpg'):
        fig = plt.figure()
        for i in range(bb_.shape[1]):
            self._drawLines(bb_[:, i])

        plt.axis([-2, 2, -2, 2])
        plt.savefig(filename_)
        plt.close(fig)

    def drawPointsWithPivotAndLines(self, pivot, data_, label_=None, bb_=None, filename_='./tmp_piv_points.jpg'):

        self.style_list = [('r', '.'), ('b', '.'), ('g', '.')]

        data_ = data_.squeeze()
        if data_.shape.__len__() == 1:
            data_ = np.expand_dims(data_, axis=0)

        assert data_.shape.__len__() == 2 and data_.shape[1] == 2

        fig = plt.figure()

        if label_ is None:
            plt.scatter(data_[:, 0], data_[:, 1], c='b', marker='p')

        else:
            for i in np.unique(label_):
                idx = label_ == i
                plt.scatter(data_[idx, 0], data_[idx, 1],
                            c=self.style_list[i][0], marker=self.style_list[i][1])

        # plot pivot
        plt.scatter(pivot[0], pivot[1], c='c', marker='p')

        # plot lines of bb
        if bb_.shape.__len__() == 1:
            self._drawLines(bb_)
        else:
            for i in range(bb_.shape[1]):
                self._drawLines(bb_[:,i])

        plt.axis([-2, 2, -2, 2])
        #plt.axis('square')
        plt.savefig(filename_)

        plt.close(fig)

    def drawPointsWithPivot(self, pivot, data_, label_=None, filename_='./tmp_piv_points.jpg'):
        self.style_list = [('r', '.'), ('b', '.'), ('g', '.')]

        data_ = data_.squeeze()
        if data_.shape.__len__() == 1:
            data_ = np.expand_dims(data_, axis=0)

        assert data_.shape.__len__() == 2 and data_.shape[1] == 2

        fig = plt.figure()

        if label_ is None:
            plt.scatter(data_[:, 0], data_[:, 1], c='b', marker='p')

        else:
            for i in np.unique(label_):
                idx = label_ == i
                plt.scatter(data_[idx, 0], data_[idx, 1],
                            c=self.style_list[i][0], marker=self.style_list[i][1])

        # plot pivot
        plt.scatter(pivot[0], pivot[1], c='c', marker='p')

        plt.axis([-2, 2, -2, 2])
        #plt.axis('square')
        plt.savefig(filename_)

        plt.close(fig)

    def drawMiningStep(self, invariant_, match_mat_f_, match_mat_g_, glb_idx_f_, glb_idx_g_, filename='./debug.jpg'):
        covered_glb_idx_f = self._computeCoveredGlbIdx(invariant_, match_mat_f_, glb_idx_f_)
        covered_glb_idx_g = self._computeCoveredGlbIdx(invariant_, match_mat_g_, glb_idx_g_)

        feat = train_data._getArrayFeatures()

        fig = plt.figure()

        plt.scatter(feat[covered_glb_idx_f, 0], feat[covered_glb_idx_f, 1], c='r', marker='.')
        plt.scatter(feat[covered_glb_idx_g, 0], feat[covered_glb_idx_g, 1], c='b', marker='.')
        plt.axis([-2, 2, -2, 2])
        plt.savefig(filename)

        plt.close(fig)


    def _computeCoveredGlbIdx(self, invariant_, match_mat_, glb_idx_):
        num_boundaries = np.sum(invariant_)
        local_indi = (np.sum(match_mat_[:, invariant_], axis=1) == num_boundaries)
        return glb_idx_[local_indi]

    @staticmethod
    def testdraw():
        # draw points
        dataraw = torch.rand(100, 1, 1, 2).numpy()
        label   = np.random.rand(100)
        label[label > 0.5] = 1
        label[label < 0.5] = 0
        label = label.astype(np.int16)
        Plot2D().drawPoints(dataraw, label)

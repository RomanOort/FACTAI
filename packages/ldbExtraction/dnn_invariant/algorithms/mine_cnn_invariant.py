

from dnn_invariant.utilities.environ import *
from models.models4invariant import *
from queue import PriorityQueue
from dnn_invariant.utilities.visualization import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import cv2

np.random.seed(0)
torch.manual_seed(0)

class RuleMinerLargeCandiPool():
    def __init__(self, model_, train_mdata_, label_, layer_start_=-1):

        self._model = model_
        self._train_mdata = train_mdata_

        self._train_labels = train_mdata_._getArrayLabels()

        # The class of the seed image is considered positive
        # Rest of the classes are negative
        self._indices_list_pos = []
        self._indices_list_neg = []

        self._label = label_

        for idx in range(train_mdata_.__len__()):

            if self._train_labels[idx] == self._label:
                self._indices_list_pos.append(idx)
            else:
                self._indices_list_neg.append(idx)

        self._pos_sample = len(self._indices_list_pos)
        self._neg_sample = len(self._indices_list_neg)

        self._layer_start = layer_start_

    def CandidatePoolLabel(self, feature_, pool=50):

        self._bb = Struct_BB()

        boundary_list = []
        opposite_list = []
        initial_list = []

        self._model.eval()

        random_set = np.random.choice(self._neg_sample, pool, replace=False)

        for i in range(pool):

            if i % 100 == 0:
                print('Extracting Candidate Pool ', i)

            neg_index = self._indices_list_neg[random_set[i]]
            pos = feature_
            neg = self._train_mdata.getCNNFeature(neg_index).cuda()
            initial_list.append(neg_index)

            while True:
                # Adjusted binary search
                boundary_pt = 0.9 * pos + 0.1 * neg

                # Output of the boundary point
                vec = self._model._getOutputOfOneLayer(boundary_pt, self._model._layers_list.__len__() - 1, self._layer_start + 1).cpu().detach().numpy()

                vec_order = np.argsort(vec[0])
                out1 = vec_order[-1] # index of the largest element, which is the output
                out2 = vec_order[-2] # index of the second largest element

                if (vec[0][out1] - vec[0][out2]) ** 2 < 0.00001 and out1 == self._label:
                    break

                if out1 == self._label:
                    pos = boundary_pt
                else:
                    neg = boundary_pt

            boundary_list.append(boundary_pt)
            opposite_list.append(out2)

            bb_buf = self._model._getBBOfLastLayer(boundary_pt, self._layer_start + 1)
            self._bb.extendBB(bb_buf)

            self._train_configs = self._bb.computeConfigs(self._train_mdata._getArrayFeatures())

        self._boundary_list = boundary_list
        self._opposite_list = opposite_list
        self._initial_list = initial_list


    def _getCoveredIndi(self, invariant_, tgt_config_):
        num_boundaries = np.sum(invariant_)
        array_features_ = self._train_mdata._getArrayFeatures()

        configs = self._bb.computeSubConfigs(invariant_, array_features_)
        match_mat = (configs - tgt_config_ != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)

        return cover_indi

    def getInvariantClassifier(self, tgt_idx_, feature, label, org_train_labels_, delta_constr_=0, peel_mask_=None): # tgt_idx_ is probably not used
        train_configs = self._train_configs

        feature = feature.flatten()
        feature = feature[np.newaxis, :]
        tgt_config = self._bb.computeConfigs(feature)
        tgt_config = np.squeeze(tgt_config)

        train_labels = self._train_labels
        tgt_label = label

        indi_f = (train_labels == tgt_label)
        indi_g = ~indi_f

        if peel_mask_ != None:
            indi_f[peel_mask_] = False

        match_mat_f = (train_configs[indi_f, :] - tgt_config == 0)
        match_mat_g = (train_configs[indi_g, :] - tgt_config == 0)

        submodu_miner = SubmodularMinerAG(match_mat_f, match_mat_g, np.where(indi_f), np.where(indi_g), False)
        invariant, f_val, g_val = submodu_miner.mineInvariant(delta_constr_=delta_constr_)

        cover_indi = self._getCoveredIndi(invariant, tgt_config[invariant])
        org_train_subdata = self._train_mdata._getArrayFeatures()[cover_indi, :]
        org_train_sublabels = org_train_labels_[cover_indi]

        inv_classifier = InvariantClassifierGlb(self._bb, invariant, f_val, g_val, tgt_config, tgt_label,
                                                org_train_subdata, org_train_sublabels,
                                                self._train_mdata._getArrayFeatures()[tgt_idx_, :], self._layer_start, self._model)

        boundary_list_update = []
        opposite_list_update = []
        initial_list_update = []
        for i in range(len(self._boundary_list)):
            if invariant[i]:
                boundary_list_update.append(self._boundary_list[i])
                opposite_list_update.append(self._opposite_list[i])
                initial_list_update.append(self._initial_list[i])

        return inv_classifier, boundary_list_update, opposite_list_update, initial_list_update












class RuleMinerLargeCandiPool_peter():
    def __init__(self, model_, train_mdata_, sample_rate_=1):
        print('initializing RuleMinerDecisionFeatOnly ...')

        self._model = model_
        self._train_mdata = train_mdata_

        self._train_labels = train_mdata_._getArrayLabels()

        self._bb = Struct_BB()
        #labels_list = []
        indices_list_pos = []
        indices_list_neg = []

        for idx in range(train_mdata_.__len__()):

            if idx % 100 == 0:
                print("%d/%d, num candi: %d" % (idx, train_mdata_.__len__(), self._bb.getSizeOfBB()))


            cnn_feat = train_mdata_.getCNNFeature(idx).cuda()
            vec = self._model(cnn_feat).cpu().detach().numpy()

            if self._train_labels[idx] == 0:
                indices_list_neg.append(idx)
            else:
                indices_list_pos.append(idx)

            if self._bb.getSizeOfBB() > 1e6:
                break

            dice = np.random.rand(1)[0]
            if dice > sample_rate_:
                continue

            if (vec[0][0] - vec[0][1]) ** 2 > 0.001:
                continue

            #cnn_feat = train_mdata_.getCNNFeature(idx).cuda()
            bb_buf = model_._getBBOfLastLayer(cnn_feat)
            self._bb.extendBB(bb_buf)

            #labels_list.append(self._train_labels[idx])

        pos_sample = len(indices_list_pos)
        neg_sample = len(indices_list_neg)

        for i in range(1000):
            if i % 100 == 0:
                print('Boundary Creation: ', i)
            pos_index = indices_list_pos[np.random.randint(0, pos_sample)]
            neg_index = indices_list_neg[np.random.randint(0, neg_sample)]

            pos = train_mdata_.getCNNFeature(pos_index).cuda()
            neg = train_mdata_.getCNNFeature(neg_index).cuda()

            while True:
                boundary_pt = (pos + neg) / 2
                vec = self._model(boundary_pt).cpu().detach().numpy()
                if (vec[0][0] - vec[0][1]) ** 2 < 0.001:
                    break

                if vec[0][0] > vec[0][1]:
                    neg = boundary_pt
                else:
                    pos = boundary_pt

            bb_buf = model_._getBBOfLastLayer(boundary_pt)
            self._bb.extendBB(bb_buf)

        self._train_configs = self._bb.computeConfigs(train_mdata_._getArrayFeatures())

        print('Done! Num candidate bb: %d\n' % (self._train_configs.shape[1]))
        #print('Sampled classes: ', labels_list)

    def _getCoveredIndi(self, invariant_, tgt_config_):
        num_boundaries = np.sum(invariant_)

        array_features_ = self._train_mdata._getArrayFeatures()

        configs = self._bb.computeSubConfigs(invariant_, array_features_)
        match_mat = (configs - tgt_config_ != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)

        return cover_indi

    def getInvariantClassifier(self, tgt_idx_, org_train_labels_, delta_constr_=0, peel_mask_=None):
        train_configs = self._train_configs

        tgt_config = train_configs[tgt_idx_, :]

        train_labels = self._train_labels
        tgt_label = train_labels[tgt_idx_]

        indi_f = (train_labels == tgt_label)
        indi_g = ~indi_f

        # Disabled nearest neighbors
        #nn_indi, min_dist = self._train_mdata._getNNIndi(radius_=0, center_idx_=tgt_idx_)
        #print('num NN: %d' % (np.sum(nn_indi)))
        #indi_f[~nn_indi] = False
        #indi_g[~nn_indi] = False

        if peel_mask_ != None:
            indi_f[peel_mask_] = False

        match_mat_f = (train_configs[indi_f, :] - tgt_config == 0)
        match_mat_g = (train_configs[indi_g, :] - tgt_config == 0)

        submodu_miner = SubmodularMinerAG(match_mat_f, match_mat_g, np.where(indi_f), np.where(indi_g), False)
        # On and Off
        invariant, f_val, g_val = submodu_miner.mineInvariant(delta_constr_=delta_constr_)

        cover_indi = self._getCoveredIndi(invariant, tgt_config[invariant])
        org_train_subdata = self._train_mdata._getArrayFeatures()[cover_indi, :]
        org_train_sublabels = org_train_labels_[cover_indi]

        inv_classifier = InvariantClassifierGlb(self._bb, invariant, f_val, g_val, tgt_config, tgt_label,
                                                org_train_subdata, org_train_sublabels,
                                                self._train_mdata._getArrayFeatures()[tgt_idx_, :])

        return inv_classifier, tgt_idx_

class RuleMinerBoost():
    def __init__(self, model_list_, layer_ptr_lists_):
        self._model_list = model_list_  # a loaded model for mining
        self._layer_ptr_lists = layer_ptr_lists_  # the list of layers for configuration

    def _getBB4OneInput(self, cnn_input_):
        bb = Struct_BB()
        for i in range(self._model_list.__len__()):
            model = self._model_list[i]
            layer_ptr_list = self._layer_ptr_lists[i]
            bb_buf = model._getBBOfAllLayers(cnn_input_, layer_ptr_list)
            bb.extendBB(bb_buf)

        return bb

    def getInvariantClassifier(self, train_mdata_, tgt_idx_, delta_constr_=0):
        bb = self._getBB4OneInput(train_mdata_.getCNNFeature(tgt_idx_).cuda())

        train_configs = bb.computeConfigs(train_mdata_._getArrayFeatures())
        tgt_config = train_configs[tgt_idx_, :]

        train_labels = train_mdata_._getArrayLabels()
        tgt_label = train_labels[tgt_idx_]

        indi_f = (train_labels == tgt_label)
        indi_g = ~indi_f

        match_mat_f = (train_configs[indi_f, :] - tgt_config == 0)
        match_mat_g = (train_configs[indi_g, :] - tgt_config == 0)

        submodu_miner = SubmodularMinerAG(match_mat_f, match_mat_g, np.where(indi_f), np.where(indi_g), False)
        invariant, f_val, g_val = submodu_miner.mineInvariant(delta_constr_=delta_constr_)

        if invariant is None:
            return None

        inv_classifier = InvariantClassifier(bb, invariant, f_val, g_val, tgt_config, tgt_label)

        return inv_classifier












class InvariantClassifierGlb():
    def __init__(self, bb_, invariant_, f_val, g_val, tgt_config_, tgt_label_, org_train_subdata, org_train_sublabels, tgt_feature, layer_start_, model_):
        assert np.sum(invariant_) > 0

        #self._tgt_feature = tgt_feature
        #self._min_dist = min_dist

        self._layer_start = layer_start_
        self._model = model_

        # assign self._bb that stores all decision boundaries
        self._bb = bb_
        self._invariant = invariant_

        self._gt_config = tgt_config_[invariant_]

        # log the f_val and g_val for tihs rule
        self._f_val = f_val
        self._g_val = g_val

        # assign label for this convex polytope
        self._label = tgt_label_

        if np.unique(org_train_sublabels).__len__() <= 1:
            self._is_pure = True
        else:
            self._is_pure = False

        '''
        # train a logistic regression
        if self._is_pure == False:
            #self.classifier = LogisticRegression()
            self.classifier = SVC(kernel='rbf')
            self.classifier.fit(org_train_subdata, org_train_sublabels)
        '''

        #print('subtrain labels: ', [(lb, np.sum(org_train_sublabels == lb)) for lb in set(org_train_sublabels)], '\n')

    def updateSupportScore(self, spt_score_):
        self._spt_score = spt_score_

    def getNumBoundaries(self):
        return np.sum(self._invariant)

    def classify_one_boundary(self, array_features_, labels_):
        for j in range(len(self._invariant)):

            if self._invariant[j] == 0:
                continue

            else:
                invariant = np.zeros(len(self._invariant), dtype=bool)
                invariant[j] = 1

                num_boundaries = self.getNumBoundaries()
                configs = self._bb.computeSubConfigs(invariant, array_features_)
                match_mat = (configs - self._gt_config != 0)
                if num_boundaries > 1:
                    check_sum = np.sum(match_mat, axis=1)
                else:
                    check_sum = match_mat.squeeze()
                cover_indi = (check_sum == 0)

            print(j, np.count_nonzero(labels_[cover_indi] == 0), np.count_nonzero(labels_[cover_indi] == 1), np.count_nonzero(labels_[cover_indi] == 2))

    def classify_one_boundary_specific(self, array_features_, db):

        count = 0
        for j in range(len(self._invariant)):

            if self._invariant[j] == 0:
                continue

            else:
                if count != db:
                    count += 1
                    continue
                else:
                    invariant = np.zeros(len(self._invariant), dtype=bool)
                    invariant[j] = 1

                    pixels = []
                    for pixel in range(196):
                        array = array_features_.copy()
                        for variable in range(512 * 14 * 14):
                            if variable % 196 != pixel:
                                array[0, variable] = 0
                        val = self._bb.computeSubHashVal(invariant, array)
                        pixels.append(val)
                    count += 1
        pixels = np.array(pixels)
        pixels = np.reshape(pixels, (14, 14))
        #pixels = np.maximum(pixels, 0)
        pixels = cv2.resize(pixels, (H, W))
        pixels = pixels - np.min(pixels)
        if np.max(pixels) != 0:
            pixels = pixels / np.max(pixels)

        return pixels

    def classify(self, array_features_):
        num_boundaries = self.getNumBoundaries()

        #dist = np.sqrt(np.sum(np.power((array_features_ - self._tgt_feature), 2), axis=1))

        configs = self._bb.computeSubConfigs(self._invariant, array_features_)
        match_mat = (configs - self._gt_config != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)# & (dist <= self._min_dist)

        pred_labels = np.zeros(array_features_.shape[0]) - 1

        '''
        if np.sum(cover_indi) > 0:
            if self._is_pure == True:
                pred_labels[cover_indi] = self._label
            else:
                pred_labels[cover_indi] = self.classifier.predict(array_features_[cover_indi, :])

            pred_labels[cover_indi] = self._label
        '''

        pred_labels[cover_indi] = self._label

        return pred_labels, cover_indi


class InvariantClassifier():
    def __init__(self, bb_, invariant_, f_val, g_val, tgt_config_, tgt_label_):
        assert np.sum(invariant_) > 0

        # assign self._bb that stores all decision boundaries
        sub_bias = bb_.getBiasArray()[invariant_]
        sub_basis = bb_.getBasisArray()[:, invariant_]
        self._bb = Struct_BB(sub_basis, sub_bias)

        self._gt_config = tgt_config_[invariant_]

        # log the f_val and g_val for tihs rule
        self._f_val = f_val
        self._g_val = g_val

        # assign label for this convex polytope
        self._label = tgt_label_

    def updateSupportScore(self, spt_score_):
        self._spt_score = spt_score_

    def getNumBoundaries(self):
        return self._bb.getSizeOfBB()

    def classify(self, array_features_):
        num_boundaries = self.getNumBoundaries()

        configs = self._bb.computeConfigs(array_features_)
        match_mat = (configs - self._gt_config != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0)

        pred_labels = np.zeros(array_features_.shape[0]) - 1
        pred_labels[cover_indi] = self._label

        return pred_labels, cover_indi

    def exportConstraints(self, filename='tmp.cst'):
        # computes constraints in the form of wx + b <= 0
        bb_array = np.array(self._bb.getBBArray())
        for i in range(self._gt_config.__len__()):
            config_i = self._gt_config[i]
            assert config_i == 1 or config_i == 0
            if config_i == 1:
                bb_array[:, i] = -bb_array[:, i]

        import pickle as pkl
        with open(filename, 'wb') as fid:
            pkl.dump([bb_array], fid)




# Let OracleSP handle all the complicated computations
# SP stands for sparse
# such as maintaining precomputed statistics and computing marginal gains
class OracleSP():
    def __init__(self, match_mat_f_, match_mat_g_):
        self._match_mat_f = match_mat_f_
        self._match_mat_g = match_mat_g_

        assert match_mat_f_.shape[1] == match_mat_g_.shape[1]

        self._D = match_mat_f_.shape[1]
        self._f_N = match_mat_f_.shape[0]  # number of samples in all positive data
        self._g_N = match_mat_g_.shape[0]  # number of samples in all negative data
        self._N = self._f_N + self._g_N  # number of samples in all training data

        # Weighted
        self._f_N_weights = (np.sum(self._match_mat_f, axis=1) / self._D) ** 4
        self._g_N_weights = (np.sum(self._match_mat_g, axis=1) / self._D) ** 4
        self._f_N_weighted = np.sum(self._f_N_weights)
        self._g_N_weighted = np.sum(self._g_N_weights)

    # init the precomputed statistics
    def _init_precomp_stat(self):
        # init the buffer of merged cols for new_inv_Y=all zeros
        self._buf_ids_f = np.array(range(self._f_N))
        self._buf_ids_g = np.array(range(self._g_N))

    def _compute_nom_j(self, j_):
        if isinstance(self._match_mat_f, np.ndarray):
            matchmat_f_colj = self._match_mat_f[:, j_]
        else:
            matchmat_f_colj = np.asarray(self._match_mat_f[:, j_].todense()).squeeze()

        nom_j = np.sum(matchmat_f_colj[self._buf_ids_f] == False)

        assert nom_j >= 0

        return nom_j

    def _compute_denom_j(self, j_):
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, j_].todense()).squeeze()

        denom_j = np.sum(match_mat_g_colj[self._buf_ids_g] == False)

        assert denom_j >= 0

        return denom_j

    def _compute_ratio_vec(self):
        assert isinstance(self._match_mat_f, np.ndarray)
        assert isinstance(self._match_mat_g, np.ndarray)

        nom = np.sum(self._match_mat_f[self._buf_ids_f, :] == False, axis=0) + 1e-5
        denom = np.sum(self._match_mat_g[self._buf_ids_g, :] == False, axis=0) + 1e-10

        ratio_vec = nom/denom

        return ratio_vec

    def _update_by_j(self, sel_j_):
        # update precomputed statistics self._buf_ids_f
        if isinstance(self._match_mat_f, np.ndarray):
            match_mat_f_colj = self._match_mat_f[:, sel_j_]
        else:
            match_mat_f_colj = np.asarray(self._match_mat_f[:, sel_j_].todense()).squeeze()

        self._buf_ids_f = self._buf_ids_f[np.where(match_mat_f_colj[self._buf_ids_f])[0]]

        # update precomputed statistics self._buf_ids_g
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, sel_j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, sel_j_].todense()).squeeze()

        self._buf_ids_g = self._buf_ids_g[np.where(match_mat_g_colj[self._buf_ids_g])[0]]

    def compute_gval(self):
        gval_Y = self._g_N - self._buf_ids_g.__len__()
        return gval_Y

    def compute_fval(self):
        fval_Y = self._f_N - self._buf_ids_f.__len__()
        return fval_Y

    def compute_fval_idx(self):
        return self._buf_ids_f

    # init the precomputed statistics
    def _init_precomp_stat_peter(self, old_inv_X_):
        # initializes all kinds of precomputed statistics
        # such as self._buf_colsum_f, self._buf_colnum_f, self._fval_X and self._buf_vec_g

        # init the buffer of all matching cols of old_inv_X_
        if np.sum(old_inv_X_) > 0:
            self._buf_colsum_f = np.sum(self._match_mat_f[:, old_inv_X_], axis=1)
            self._buf_colnum_f = np.sum(old_inv_X_)
            self._fval_X = self._f_N_weighted - np.dot(self._buf_colsum_f == self._buf_colnum_f, self._f_N_weights)
        else:
            self._fval_X = self._f_N_weighted

        entire_set = np.ones(self._match_mat_f.shape[1], dtype=bool)
        self._entire_colsum_f = np.sum(self._match_mat_f[:, entire_set], axis=1)
        self._entire_colnum_f = np.sum(entire_set)
        self._fval_V = self._f_N_weighted - np.dot(self._entire_colnum_f == self._entire_colsum_f, self._f_N_weights)

        # init the buffer of merged cols for new_inv_Y=all zeros
        self._buf_vec_g  = np.ones(self._g_N, dtype=bool) # all True

    def _compute_nom_j_peter(self, j_, old_inv_X_):
        nom_j = -1

        if old_inv_X_[j_] == True:
            # j in old_inv_X_
            buf_colsum_f_woj = self._buf_colsum_f - self._match_mat_f[:, j_]
            buf_colnum_f_woj = self._buf_colnum_f - 1
            fval_V_wo_j = self._f_N_weighted - np.dot(buf_colsum_f_woj == buf_colnum_f_woj, self._f_N_weights)
            nom_j = self._fval_X - fval_V_wo_j
        else:
            # j not in old_inv_X_
            if np.sum(old_inv_X_) > 0:
                buf_colsum_f_wj = self._buf_colsum_f + self._match_mat_f[:, j_]
                buf_colnum_f_wj = self._buf_colnum_f + 1
                fval_X_wj = self._f_N_weighted - np.dot(buf_colsum_f_wj == buf_colnum_f_wj, self._f_N_weights)
                nom_j = fval_X_wj - self._fval_X
            else:
                nom_j = self._f_N_weighted - np.dot(self._match_mat_f[:, j_], self._f_N_weights) # Use it when X is empty

        assert nom_j >= -0.00001

        return nom_j

    def _compute_denom_j_peter(self, j_):
        gval_Y = self._g_N_weighted - np.sum(np.dot(self._buf_vec_g, self._g_N_weights))
        #gval_Y = self._g_N - np.sum(self._buf_vec_g)
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, j_].todense()).squeeze()

        gval_Y_w_j = self._g_N_weighted - np.sum(np.dot(self._buf_vec_g & match_mat_g_colj, self._g_N_weights))
        #gval_Y_w_j = self._g_N - np.sum(self._buf_vec_g & match_mat_g_colj)

        denom_j = (gval_Y_w_j - gval_Y) ** 1

        assert denom_j >= 0

        return denom_j

    def _compute_gval_mval_peter(self, old_inv_X_, new_inv_Y_):
        # compute g_val
        g_val = self._g_N - np.sum(self._buf_vec_g)  # the g_val of new_inv_Y (with sel_j)

        # compute m_val
        m_val = self._fval_X

        for _, j in np.ndenumerate(np.where(old_inv_X_ & ~new_inv_Y_)):
            if j.size > 0:
                m_val -= self._compute_nom_j_peter(j, old_inv_X_)

        for _, j in np.ndenumerate(np.where(new_inv_Y_ & ~old_inv_X_)):
            if j.size > 0:
                m_val += self._compute_nom_j_peter(j, old_inv_X_)

        return np.asscalar(g_val), np.asscalar(m_val)

    def _update_by_j_peter(self, sel_j_):
        # update precomputed statistics self._buf_vec_g
        if isinstance(self._match_mat_g, np.ndarray):
            match_mat_g_colj = self._match_mat_g[:, sel_j_]
        else:
            match_mat_g_colj = np.asarray(self._match_mat_g[:, sel_j_].todense()).squeeze()

        self._buf_vec_g = self._buf_vec_g & match_mat_g_colj

    def compute_fval_peter(self, invariant_):
        colsum_f    = np.sum(self._match_mat_f[:, invariant_], axis=1)
        colnum_f    = np.sum(invariant_)
        f_val       = self._f_N - np.sum(colsum_f == colnum_f)

        f_val_idx   = np.where(colsum_f == colnum_f)

        return f_val, f_val_idx

    def compute_gval_peter(self, invariant_):
        colsum_g    = np.sum(self._match_mat_g[:, invariant_], axis=1)
        colnum_g    = np.sum(invariant_)
        g_val       = self._g_N - np.sum(colsum_g == colnum_g)

        g_val_idx   = np.where(colsum_g == colnum_g)

        return g_val, g_val_idx


# This is the core of our invariant mining algorithm.
# This solves the submodular minimization with submodular constraint problem.
# AG means Accelerated Greedy version using priority queue.
class SubmodularMinerAG():
    def __init__(self, match_mat_f_, match_mat_g_, glb_idx_f_, glb_idx_g_, verbal_=False):
        # match_mat_f_: the true false match matrix for f (i.e., configurations of positive samples)
        # match_mat_g_: the true false match matrix for g (i.e., configurations of negative samples)
        self._oracle = OracleSP(match_mat_f_, match_mat_g_)
        self._glb_idx_f = np.array(glb_idx_f_).squeeze()
        self._glb_idx_g = np.array(glb_idx_g_).squeeze()
        self.mat_1 = match_mat_f_
        self.mat_2 = match_mat_g_
        self._verbal = verbal_

    def mineInvariant(self, delta_constr_=0):
        # this uses greedy submodular minimization with constraints to mine
        # the invariant w.r.t. the sample with idx_
        # the real constraint is self._g_N - delta_constr_, where self._g_N is the total number of training data for g.

        self._constr = self._oracle._g_N - delta_constr_
        if self._verbal: print('constraint: %d' % (self._constr))

        invariant, f_val, g_val = self._mineInvariantCore()

        #invariant = self._tightenInvariant(invariant)
        invariant = self._lossenInvariant(invariant)


        f_val_idx = self._oracle.compute_fval_idx()

        if self._verbal: print('FINAL ===> f_val: %d\tg_val: %d\tf_N: %d\tg_N: %d\n' % (
        f_val, g_val, self._oracle._f_N, self._oracle._g_N))
        if self._verbal: print('global:\n', self._glb_idx_f[f_val_idx])
        if self._verbal: print('local:\n', f_val_idx)

        return invariant, f_val, g_val

    def mineInvariant_peter(self, delta_constr_=0):
        # this uses greedy submodular minimization with constraints to mine
        # the invariant w.r.t. the sample with idx_
        # the real constraint is self._g_N - delta_constr_, where self._g_N is the total number of training data for g.
        self._constr = self._oracle._g_N - delta_constr_

        # start iteration
        old_m_val = self._oracle._f_N * 2  # double _f_N to make sure m_val is an upper bound
        invariant = np.zeros(self._oracle._D, dtype=bool)  # this is an indicator vector of selected dimensions
        total_invariant = np.zeros(self._oracle._D, dtype=bool)  # will use this to store all previous selected boundaries
        iter = 0
        while True and iter < 5: # number of iterations
            iter += 1
            new_invariant, g_val, m_val = self._mineInvariantCore_peter(invariant)

            total_invariant = np.maximum(total_invariant, new_invariant)  # keep track of boundaries that have been selected at some point

            if np.array_equal(new_invariant, invariant):
                invariant = new_invariant
                break
            else:
                invariant = new_invariant
                old_m_val = m_val



        #print('Total Boundaries: ', np.sum(total_invariant))
        removal_pool = total_invariant.copy()
        record = total_invariant.copy()
        min_f_val = 100000
        for turn in range(1):

            invariant = removal_pool.copy()
            while True:
                location = np.where(invariant == 1)[0]
                max = 0
                arg = -1
                for i in range(len(location)):
                    invariant_temp = invariant.copy()
                    invariant_temp[location[i]] = 0
                    self._buf_colsum_f = np.sum(self.mat_1[:, invariant_temp], axis=1)
                    self._buf_colnum_f = np.sum(invariant_temp)
                    self._fval_X = np.sum(self._buf_colsum_f == self._buf_colnum_f)
                    self._buf_colsum_g = np.sum(self.mat_2[:, invariant_temp], axis=1)
                    self._buf_colnum_g = np.sum(invariant_temp)
                    self._gval_X = self.mat_2.shape[0] - np.sum(self._buf_colsum_g == self._buf_colnum_g)
                    if self._gval_X > self.mat_2.shape[0] - 0.01 * self._fval_X and self._fval_X > max:  # Forgiving for misclassification
                        arg = location[i]
                        max = self._fval_X
                        max_save = self._fval_X

                if arg != -1:
                    invariant[arg] = 0

                if arg == -1:
                    break

            f_val, f_val_idx = self._oracle.compute_fval_peter(invariant)
            first = np.where(invariant == 1)[0][0:5]
            #print('Turn {}: {}'.format(turn, max_save))
            removal_pool[first] = 0
            if f_val < min_f_val:
                min_f_val = f_val
                record = invariant



        return record, f_val, g_val #m_val, f_val_idx # invariant


    def _init_sel_queue_peter(self, old_inv_X_):
        sel_queue = PriorityQueue()
        for j in range(self._oracle._D):
            # compute nom_vec_j
            nom_j = self._oracle._compute_nom_j_peter(j, old_inv_X_)

            # Reduce the advantage of previous selected boundaries
            if nom_j == 0:
                nom_j = 1

            # compute denom_vec_j
            denom_j = self._oracle._compute_denom_j_peter(j)

            # compute the ratio
            if denom_j > 0:
                # we do not put j into queue when denom_vec_j=0
                # because, we will never select such a j.
                lb_ratio = nom_j/denom_j # the lower bound of ratio
                sel_queue.put((lb_ratio, j))

        return sel_queue

    def _tightenInvariant(self, invariant_):
        new_bits = 0
        for j in range(self._oracle._D):
            if invariant_[j] == False:
                nom_j = self._oracle._compute_nom_j(j)
                if nom_j < 2: # nom_j == 0
                    invariant_[j] = True
                    new_bits += 1

        #print('new_bits: ', new_bits)

        return invariant_

    def _lossenInvariant(self, invariant_):
        new_bits = 0
        record_f = self._oracle.compute_fval_peter(invariant_)[0]
        record_g = self._oracle.compute_gval_peter(invariant_)[0]
        for j in range(self._oracle._D):
            if invariant_[j] == True:

                temp = invariant_.copy()
                temp[j] = 0
                if self._oracle.compute_fval_peter(temp)[0] == record_f and self._oracle.compute_gval_peter(temp)[0] == record_g:
                    invariant_[j] = False
                    new_bits += 1
        '''
        idd = 0
        for k in range(self._oracle._D):
            if invariant_[k] == True:
                print('Decision Boundary {}'.format(idd))
                idd += 1
                temp = np.repeat(False, len(invariant_))
                temp[k] = True
                print(self._oracle.compute_fval_peter(temp)[0], self._oracle.compute_gval_peter(temp)[0])
        '''

        #print('new_bits: ', new_bits)

        return invariant_

    def _mineInvariantCore(self):
        # init the precomputed statistics
        self._oracle._init_precomp_stat()

        # start iteration
        steps = 0
        new_inv_Y = np.zeros(self._oracle._D, dtype=bool)

        while True:
            steps += 1

            '''
            Plot2D().drawMiningStep(new_inv_Y, self._oracle._match_mat_f, self._oracle._match_mat_g,
                                    self._glb_idx_f, self._glb_idx_g, ('step_' + str(steps) + '.jpg'))
            '''

            sel_j = self._select_j_and_update(new_inv_Y)
            g_val = self._oracle.compute_gval()

            if self._verbal: print('steps: %d\tsel_j: %d\tg_val: %d\tf_val: %d' % (steps, sel_j, g_val, self._oracle.compute_fval()))

            if sel_j < 0 or g_val >= self._constr or steps > 110: # max no. of boundaries
                break

        return new_inv_Y, self._oracle.compute_fval(), self._oracle.compute_gval()

    def _mineInvariantCore_peter(self, old_inv_X_):
        # init the precomputed statistics
        self._oracle._init_precomp_stat_peter(old_inv_X_)

        # init selection queue storing the lower bound of each selection ratio
        sel_queue = self._init_sel_queue_peter(old_inv_X_)

        # start iteration
        g_val = 0
        m_val = self._oracle._f_N
        new_inv_Y = np.zeros(self._oracle._D, dtype=bool)
        #margin = min(margin, 500)
        while (g_val < self._constr) and (not sel_queue.empty()):
            g_val, m_val = self._select_j_and_update_peter(old_inv_X_, new_inv_Y, sel_queue)

        return new_inv_Y, g_val, m_val

    def _select_j_and_update(self, new_inv_Y_):
        ratio_vec = self._oracle._compute_ratio_vec()
        ratio_vec[new_inv_Y_] = 1e10

        sel_j = np.argmin(ratio_vec)

        new_inv_Y_[sel_j] = True
        self._oracle._update_by_j(sel_j)

        return sel_j

    def _select_j_and_update_peter(self, old_inv_X_, new_inv_Y_, sel_queue_):
        assert not sel_queue_.empty()
        # init sel_j
        sel_j = -1
        # we have more than one choices of sel_j
        while sel_queue_.queue.__len__() >= 2:

            top_item        = sel_queue_.get()
            candi_j         = top_item[1]

            if candi_j == 1127:
                leon = 1

            real_nom_j      = self._oracle._compute_nom_j_peter(candi_j, old_inv_X_)

            # Reduce the advantage of previous selected boundariesz
            if real_nom_j == 0:
                real_nom_j = 1

            real_denom_j    = self._oracle._compute_denom_j_peter(candi_j)

            assert real_denom_j >= 0
            if real_denom_j == 0:
                # the real_denom_j can be 0, because new_inv_Y_ is being updated
                continue

            real_ratio      = real_nom_j/real_denom_j

            peek_lb_ratio   = sel_queue_.queue[0][0]
            if real_ratio > peek_lb_ratio:
                sel_queue_.put((real_ratio, candi_j))
            else:
                sel_j = candi_j
                #print(candi_j, ' is selected, with ratio: ', real_nom_j, real_denom_j)
                break

        if sel_j == -1:
            assert sel_queue_.queue.__len__() == 1

            # only one choice of sel_j left
            top_item        = sel_queue_.get()
            candi_j         = top_item[1]

            real_denom_j    = self._oracle._compute_denom_j_peter(candi_j)

            assert real_denom_j >= 0
            if real_denom_j > 0:
                sel_j = candi_j

        assert sel_j >= 0 or sel_j == -1

        if sel_j >= 0:
            # we found a valid sel_j, thus do update by sel_j
            # update new_inv_Y_ by sel_j
            new_inv_Y_[sel_j] = True

            # update oracle by sel_j
            self._oracle._update_by_j_peter(sel_j)

        # compute g_val and m_val for return (Leon: this may waste time if sel_j = -1)
        g_val, m_val = self._oracle._compute_gval_mval_peter(old_inv_X_, new_inv_Y_)
        return g_val, m_val

    def OLD_select_j_and_update(self, new_inv_Y_):
        sel_j = -1  # init sel_j

        noloss_queue = PriorityQueue()
        sel_queue = PriorityQueue()

        for candi_j in range(self._oracle._D):
            if new_inv_Y_[candi_j] == False:
                nom_j   = self._oracle._compute_nom_j(candi_j)
                denom_j = self._oracle._compute_denom_j(candi_j)

                if denom_j > 0:
                    if nom_j == 0:
                        ratio_j = 1.0/denom_j
                        noloss_queue.put((ratio_j, candi_j))
                    else:
                        ratio_j = nom_j/denom_j
                        sel_queue.put((ratio_j, candi_j))

        if not noloss_queue.empty():
            top_item = noloss_queue.get()
            sel_j = top_item[1]
        elif not sel_queue.empty():
            top_item = sel_queue.get()
            sel_j = top_item[1]

        assert sel_j >= 0 or sel_j == -1

        if sel_j >= 0:
            new_inv_Y_[sel_j] = True
            self._oracle._update_by_j(sel_j)

        return sel_j

    def OLD_select_j_and_update(self, new_inv_Y_):
        buget = self._constr - self._oracle.compute_gval()
        assert buget >= 0

        sel_j = -1  # init sel_j

        purecut_queue = PriorityQueue()
        noloss_queue = PriorityQueue()
        sel_queue = PriorityQueue()

        for candi_j in range(self._oracle._D):
            if new_inv_Y_[candi_j] == False:
                nom_j   = self._oracle._compute_nom_j(candi_j)
                denom_j = self._oracle._compute_denom_j(candi_j)

                if denom_j >= buget:
                    ratio_j = nom_j / denom_j
                    purecut_queue.put((ratio_j, candi_j))
                elif denom_j > 0:
                    if nom_j == 0:
                        ratio_j = 1.0/denom_j
                        noloss_queue.put((ratio_j, candi_j))
                    else:
                        ratio_j = nom_j/denom_j
                        sel_queue.put((ratio_j, candi_j))

        if not purecut_queue.empty():
            top_item = purecut_queue.get()
            sel_j = top_item[1]
        elif not noloss_queue.empty():
            top_item = noloss_queue.get()
            sel_j = top_item[1]
        elif not sel_queue.empty():
            top_item = sel_queue.get()
            sel_j = top_item[1]

        assert sel_j >= 0 or sel_j == -1

        if sel_j >= 0:
            new_inv_Y_[sel_j] = True
            self._oracle._update_by_j(sel_j)

        return sel_j


    # applies mined invariant to do classification
    @staticmethod
    def classify(invariant_, configs_, labels_, gt_config_, gt_label_):
        match_mat = (configs_[:, invariant_] - gt_config_[invariant_] == 0)
        colsum = np.sum(match_mat, axis=1)
        colnum = np.sum(invariant_)
        match_row_indi = (colsum == colnum)

        num_total_samples = np.sum(match_row_indi)
        num_correct_samples = np.sum(labels_[match_row_indi] == gt_label_)
        accuracy = num_correct_samples / num_total_samples

        return accuracy, num_total_samples, match_row_indi

    # a simple test of submodularMiner
    @staticmethod
    def test_code():
        import time

        # generate data
        N = 25
        D = 8
        IDX = 0

        # seed = int(np.asscalar(np.random.rand(1)*10000))
        seed = 1699
        print('================================')
        print('seed: %d' % (seed))
        np.random.seed(seed)

        configs = np.zeros([N, D], dtype='bool')
        configs[np.random.rand(N, D) > 0.5] = True
        labels = np.zeros(N)
        labels[np.random.rand(N) > 0.5] = 1

        f_indi = (labels == labels[IDX])
        g_indi = ~f_indi

        match_mat_f = configs[f_indi]
        match_mat_g = configs[g_indi]

        print(match_mat_f)
        print('num_rows_f: %d\n' % (np.sum(f_indi)))
        print(match_mat_g)
        print('num_rows_g: %d\n' % (np.sum(g_indi)))

        m_subminer = SubmodularMinerAG(match_mat_f, match_mat_g)
        invariant = m_subminer.mineInvariant()

        print(invariant)


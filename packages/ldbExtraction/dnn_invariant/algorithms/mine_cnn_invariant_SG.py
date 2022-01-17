

from dnn_invariant.utilities.environ import *
from models.models4invariant import *
from queue import PriorityQueue
from dnn_invariant.utilities.visualization import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class RuleMinerLargeCandiPool():
    def __init__(self, model_, train_mdata_, sample_rate_=1):
        print('initializing RuleMinerDecisionFeatOnly ...')

        self._model = model_
        self._train_mdata = train_mdata_

        self._train_labels = train_mdata_._getArrayLabels()

        self._bb = Struct_BB()
        labels_list = []

        for idx in range(train_mdata_.__len__()):
            if idx % 100 == 0:
                print("%d/%d, num candi: %d" % (idx, train_mdata_.__len__(), self._bb.getSizeOfBB()))

            if self._bb.getSizeOfBB() > 1e6:
                break

            dice = np.random.rand(1)[0]
            if dice > sample_rate_:
                continue

            cnn_feat = train_mdata_.getCNNFeature(idx).cuda()
            bb_buf = model_._getBBOfLastLayer(cnn_feat)
            self._bb.extendBB(bb_buf)

            labels_list.append(self._train_labels[idx])

        self._train_configs = self._bb.computeConfigs(train_mdata_._getArrayFeatures())

        print('Done! Num candidate bb: %d\n' % (self._train_configs.shape[1]))
        print('Sampled classes: ', labels_list)

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

        nn_indi, min_dist = self._train_mdata._getNNIndi(radius_=0, center_idx_=tgt_idx_)
        print('num NN: %d' % (np.sum(nn_indi)))
        indi_f[~nn_indi] = False
        indi_g[~nn_indi] = False

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
                                                min_dist, self._train_mdata._getArrayFeatures()[tgt_idx_, :])

        return inv_classifier


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
    def __init__(self, bb_, invariant_, f_val, g_val, tgt_config_, tgt_label_, org_train_subdata, org_train_sublabels, min_dist, tgt_feature):
        assert np.sum(invariant_) > 0

        self._min_dist = min_dist
        self._tgt_feature = tgt_feature

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

        # train a logistic regression
        if self._is_pure == False:
            #self.classifier = LogisticRegression()
            self.classifier = SVC(kernel='rbf')
            self.classifier.fit(org_train_subdata, org_train_sublabels)

        print('subtrain labels: ', [(lb, np.sum(org_train_sublabels == lb)) for lb in set(org_train_sublabels)], '\n')

    def updateSupportScore(self, spt_score_):
        self._spt_score = spt_score_

    def getNumBoundaries(self):
        return np.sum(self._invariant)

    def classify(self, array_features_):
        num_boundaries = self.getNumBoundaries()

        dist = np.sqrt(np.sum(np.power((array_features_ - self._tgt_feature), 2), axis=1))

        configs = self._bb.computeSubConfigs(self._invariant, array_features_)
        match_mat = (configs - self._gt_config != 0)

        if num_boundaries > 1:
            check_sum = np.sum(match_mat, axis=1)
        else:
            check_sum = match_mat.squeeze()

        cover_indi = (check_sum == 0) & (dist <= self._min_dist)
        #cover_indi = (check_sum == 0)

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
        self._verbal = verbal_

    def mineInvariant(self, delta_constr_=0):
        # this uses greedy submodular minimization with constraints to mine
        # the invariant w.r.t. the sample with idx_
        # the real constraint is self._g_N - delta_constr_, where self._g_N is the total number of training data for g.

        self._constr = self._oracle._g_N - delta_constr_
        if self._verbal: print('constraint: %d' % (self._constr))

        invariant, f_val, g_val = self._mineInvariantCore()

        invariant = self._tightenInvariant(invariant)

        f_val_idx = self._oracle.compute_fval_idx()

        if self._verbal: print('FINAL ===> f_val: %d\tg_val: %d\tf_N: %d\tg_N: %d\n' % (
        f_val, g_val, self._oracle._f_N, self._oracle._g_N))
        if self._verbal: print('global:\n', self._glb_idx_f[f_val_idx])
        if self._verbal: print('local:\n', f_val_idx)

        return invariant, f_val, g_val

    def _tightenInvariant(self, invariant_):
        new_bits = 0
        for j in range(self._oracle._D):
            if invariant_[j] == False:
                nom_j = self._oracle._compute_nom_j(j)
                if nom_j == 0:
                    invariant_[j] = True
                    new_bits += 1

        print('new_bits: ', new_bits)

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

            if sel_j < 0 or g_val >= self._constr:
                break

        return new_inv_Y, self._oracle.compute_fval(), self._oracle.compute_gval()

    def _select_j_and_update(self, new_inv_Y_):
        ratio_vec = self._oracle._compute_ratio_vec()
        ratio_vec[new_inv_Y_] = 1e10

        sel_j = np.argmin(ratio_vec)

        new_inv_Y_[sel_j] = True
        self._oracle._update_by_j(sel_j)

        return sel_j


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


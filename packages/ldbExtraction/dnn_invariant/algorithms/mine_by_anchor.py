from __future__ import print_function
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
from anchor import utils
from anchor import anchor_tabular

'''
# make sure you have adult/adult.data inside dataset_folder
dataset_folder = '../data/'
dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder)


# initiate explainer
explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names,
                                                  dataset.feature_names,
                                                  dataset.data,
                                                  dataset.categorical_names)

explainer.fit(dataset.train,
              dataset.labels_train,
              dataset.validation,
              dataset.labels_validation)

# training a third party model and test its accuracy
c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
predict_fn = lambda x: c.predict(explainer.encoder.transform(x))
print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))


# extracting anchors
idx = 0
np.random.seed(1)
#print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())
'''

class mine_by_anchor():
    def __init__(self, raw_data, raw_labels):
        # raw_data is in matrix form, each row is a sample of data
        self._dataset = m_dataset()
        self._dataset.class_names = np.unique(raw_labels).astype(np.bytes_)
        self._dataset.feature_names = list(map(str, list(range(raw_data.shape[1]))))
        self._dataset.data = raw_data
        self._dataset.labels = raw_labels

        self._dataset.categorical_names = {}
        for j in range(raw_data.shape[1]):
            categorical_names_j = ['0', '1']
            self._dataset.categorical_names[j] = categorical_names_j

        self._explainer = anchor_tabular.AnchorTabularExplainer(self._dataset.class_names,
                                                                self._dataset.feature_names,
                                                                self._dataset.data,
                                                                self._dataset.categorical_names)

        self._explainer.fit(self._dataset.data,
                            self._dataset.labels,
                            self._dataset.data,
                            self._dataset.labels)

        self._model = m_classifier()
        self._model.fit(self._dataset.data, self._dataset.labels)

    def _hash(self, instance):
        key = np.packbits(instance).view('c').tostring()
        return key

    def explain(self, idx = 0):
        np.random.seed(1)
        exp = self._explainer.explain_instance(self._dataset.data[idx], self._model.predict, threshold=1.0)
        return exp

    def m_eval(self, exp, src_instance, src_label, tgt_instances, tgt_labels):
        # evaluate the real quality of explanation exp of src_instance upon tgt_intsance

        # show the explanation exp
        print('Anchor: %s' % (' AND '.join(exp.names())))

        # parsing exp
        parsed_indices = []
        parsed_pattern = []
        for str in exp.names():
            buf = str.split('=')
            parsed_indices.append(int(buf[0]))
            parsed_pattern.append(int(buf[1]))

        # check whether idx data has the pattern
        exp_pattern_key = self._hash(parsed_pattern)
        src_pattern_key = self._hash(src_instance[parsed_indices])
        if exp_pattern_key == src_pattern_key:
            print('exp contained by src_instance!')
        else:
            print('exp wrong! not contained by src_instance!')

        # extract patterns of all data and count coverage
        tgt_patterns = tgt_instances[:, parsed_indices]
        hash_all_data = {}
        for i in range(tgt_patterns.shape[0]):
            key_i = self._hash(tgt_patterns[i])
            if key_i in hash_all_data:
                hash_all_data[key_i].append(i)
            else:
                hash_all_data[key_i] = [i]

        coverage_absolute = 1.0*hash_all_data[exp_pattern_key].__len__()
        coverage_relative = coverage_absolute/tgt_patterns.shape[0]
        precision= 1.0*np.sum(tgt_labels[hash_all_data[exp_pattern_key]] == src_label)/coverage_absolute

        print('coverage_abs: %d\tcoverage_rel: %.5f\tprecision: %.2f' % (coverage_absolute, coverage_relative, precision))

        return hash_all_data[exp_pattern_key]

class m_dataset():
    def __init__(self):
        self.class_names = None
        self.feature_names = None
        self.data = None
        self.labels = None
        self.categorical_names = None

class m_classifier():
    def fit(self, data, labels):
        self._checktable = {}
        for i in range(data.shape[0]):
            key_i = self._hash(data[i])
            self._checktable[key_i] = labels[i]

    def _hash(self, instance):
        if(isinstance(instance, np.ndarray)):
            key = np.packbits(instance).view('c').tostring()
        else:
            instance = self.de_transform(instance)
            key = np.packbits(instance).view('c').tostring()
        return key

    def de_transform(self, instance):
        import scipy
        assert (isinstance(instance, scipy.sparse.csr.csr_matrix))
        trans_instance = []
        for idx in instance.indices:
            val_i = idx%2
            trans_instance.append(val_i)

        return trans_instance

    def predict(self, data):
        predicts = []
        for i in range(data.shape[0]):
            key_i = self._hash(data[i])
            if key_i in self._checktable:
                pred_i = self._checktable[key_i]
                predicts.append(pred_i)
            else:
                #print('WRONG m_classifier !')
                return 0

        return predicts








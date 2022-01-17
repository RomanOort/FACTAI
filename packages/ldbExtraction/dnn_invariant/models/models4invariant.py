

from collections import OrderedDict
import math
from torch.utils.data import DataLoader
from dnn_invariant.utilities.environ import *
from dnn_invariant.utilities.datasets import *

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._model_rootpath = './mdls/' # the root path to save trained models
        #self._model_rootpath = 's3://vbdai-share/Peter/dnn_interpretation/code/dnn_invariant/mdls/'
        import os
        if not os.path.exists(self._model_rootpath):
            os.makedirs(self._model_rootpath)

    def forward(self, input):
        # loop over all modules in _layers_list
        out = input
        for i in range(self._layers_list.__len__()):


            if self._layers_list[i].__class__.__name__ == 'BatchNorm1d':
                o_shape = out.shape
                out = out.reshape(o_shape[0],-1)
                out = self._layers_list[i](out)
                out = out.reshape(o_shape[0],o_shape[1],o_shape[2],o_shape[3])
            elif self._layers_list[i].__class__.__name__ != 'Linear':
                out = self._layers_list[i](out)
            else:
                out = self._layers_list[i](out.reshape(out.size(0), -1))



        out = out.reshape(out.size(0), -1)

        return out

    def _setLayerList(self):
        raise NotImplementedError

    # gets the output of a layer pointed by layer_ptr
    # modified to allow input from a different layer
    def _getOutputOfOneLayer(self, input, layer_ptr, layer_start=0):
        assert layer_ptr < self._layers_list.__len__()
        self.eval()

        out = input

        if layer_ptr == -1:
            return out

        for i in range(layer_start, self._layers_list.__len__()):

            if self._layers_list[i].__class__.__name__ == 'BatchNorm1d':
                o_shape = out.shape
                out = out.reshape(o_shape[0], -1)
                out = self._layers_list[i](out)
                out = out.reshape(o_shape[0], o_shape[1], o_shape[2], o_shape[3])

            elif self._layers_list[i].__class__.__name__ != 'Linear' :
                out = self._layers_list[i](out)
            else:
                out = self._layers_list[i](out.reshape(out.size(0), -1))

            if (i >= layer_ptr):
                break

        return out

    def _getOutputOfOneLayer_Group(self, input_group, layer_ptr, layer_start=0):
        assert layer_ptr < self._layers_list.__len__()
        self.eval()

        group_size = input_group.shape[0]

        for i in range(group_size):
            input = input_group[i]
            input = input.view(1, C, H, W)
            output = self._getOutputOfOneLayer(input, layer_ptr, layer_start)



            if i == 0:
                output_group = torch.zeros(input_group.shape[0], output.shape[1], output.shape[2], output.shape[3])

            output_group[i] = output

        return output_group

    def getLayerPtrList(self, tgt_modus):
        layer_ptr_list = []
        for i, modu in enumerate(self._layers_list):
            for tgt_modu in tgt_modus:
                if isinstance(modu, tgt_modu):
                    layer_ptr_list.append(i)
                    break

        assert layer_ptr_list.__len__() > 0
        return layer_ptr_list

    def _printLayerOutputSize(self):

        out = torch.zeros((1, C, H, W)).cuda()
        for i in range(0, self._layers_list.__len__()):

            print('Layer ' + str(i-1) + ':', out.shape)
            if i < self._layers_list.__len__() - 1:
                out = self._layers_list[i](out)
            else:
                out = self._layers_list[i](out.reshape(out.size(0), -1))

    # ======================================================
    # Following code gets the PCA of the output of a layer
    # ======================================================
    def _getOutPCAOfOneLayer(self, inputs, layer_ptr, batch_size = 30):
        out_of_layer = torch.tensor([], dtype = torch.float)
        for i in range (0, inputs.size(0), batch_size):
            buf_out_of_layer = self._getOutputOfOneLayer(inputs[i:i + batch_size], layer_ptr).detach().clone()
            buf_out_of_layer = buf_out_of_layer.reshape(buf_out_of_layer.size(0), -1).cpu()
            out_of_layer = torch.cat((out_of_layer, buf_out_of_layer), dim=0)

        # start doing PCA
        mean_vec = out_of_layer.mean(dim=0)
        std_vec      = out_of_layer.std(dim=0) + 0.001

        normed_out_of_layer = (out_of_layer - mean_vec)

        cov_mat = torch.mm(normed_out_of_layer.permute(1, 0), normed_out_of_layer).numpy()

        print(cov_mat.shape)
        print(cov_mat)
        np.savetxt('cov_mat.txt', cov_mat, fmt="%.2f", delimiter=' ')

        from numpy import linalg as la
        U, sigma, VT = la.svd(cov_mat)
        out_PCA = normed_out_of_layer.numpy().dot(U[:, 0:2])

        print(np.sqrt(sigma))



        from sklearn.decomposition import PCA
        pca = PCA(n_components=36, whiten=True)
        newData = pca.fit_transform(out_of_layer.numpy())
        print(newData.shape)

        pca.fit(out_of_layer.numpy())

        print(pca.singular_values_)

        return newData[:, 0:3]

    # ======================================================
    # Following code gets the configuration of layers
    # ======================================================

    # this method can take inputs tensor containing multiple samples
    def _getConfOfOneLayer(self, inputs, layer_ptr, batch_size = 30):
        conf_of_layer = torch.tensor([], dtype=torch.uint8)
        for i in range(0, inputs.size(0), batch_size):
            buf_out_of_layer  = self._getOutputOfOneLayer(inputs[i:i+batch_size], layer_ptr).detach().clone()
            buf_conf_of_layer = buf_out_of_layer.reshape(buf_out_of_layer.size(0), -1) > 0
            conf_of_layer = torch.cat((conf_of_layer, buf_conf_of_layer.cpu().type(torch.uint8)), dim=0)

        return conf_of_layer

    # ======================================================
    # Following code gets the basis and bias of units
    # ======================================================
    def _getBBOfAllLayers(self, input, layer_ptr_list):
        # prepare the buffer
        bb_all_layers = Struct_BB()

        for layer_ptr in layer_ptr_list:
            num_layers = self._layers_list.__len__()
            assert layer_ptr < num_layers

            if layer_ptr < num_layers - 1:
                if isinstance(self._layers_list[layer_ptr], nn.MaxPool2d):
                    bb_one_layer = self._getBBofMaxPool2DLayer(input, layer_ptr)
                elif isinstance(self._layers_list[layer_ptr], nn.MaxPool1d):
                    bb_one_layer = self._getBBofMaxPool1DLayer(input, layer_ptr)
                else:
                    bb_one_layer = self._getBBOfOneLayer(input, layer_ptr)
            else:
                bb_one_layer = self._getBBOfLastLayer(input)

            bb_all_layers.extendBB(bb_one_layer)

        return bb_all_layers

    def _getBBOfLastLayer(self, input, layer_start=-1):
        # get the bb of the logits
        layer_ptr = self._layers_list.__len__() - 1
        bb_of_logits = self._getBBOfOneLayer(input, layer_ptr, layer_start)

        # identify the idx of the pivot logit
        logits = bb_of_logits.computeHashVal(input.reshape(input.size(0), -1).cpu().numpy())
        assert(logits.shape[0] == 1)
        logits = logits.squeeze()

        logits_order = np.argsort(logits)
        pivot_id1 = logits_order[-1]
        pivot_id2 = logits_order[-2]

        #pivot_id = np.argmax(logits)

        # subtract between the logits to get BB_of_last_layer
        bb_of_logits.subPivotOverOthers(pivot_id1, pivot_id2)

        return bb_of_logits


    def _getBBofMaxPool1DLayer(self, input, layer_ptr):
        assert (layer_ptr > 0)

        assert(False)

    def _getBBofMaxPool2DLayer(self, input, layer_ptr):
        assert(layer_ptr > 0)
        assert(isinstance(self._layers_list[layer_ptr], nn.MaxPool2d))

        # get kernalsize, stride and padding
        kernel_size = self._layers_list[layer_ptr].kernel_size
        stride      = self._layers_list[layer_ptr].stride
        padding     = self._layers_list[layer_ptr].padding

        # start to extract BB for previous layer of Maxpool2d
        # prepare the buffer
        basis_list = []
        bias_list = []

        # set true grad flag for input
        input.requires_grad_(True)

        out_of_layer = self._getOutputOfOneLayer(input, layer_ptr - 1)

        org_out_size = out_of_layer.size()

        out_of_layer = out_of_layer.reshape(out_of_layer.size(0), -1)

        self.zero_grad()
        self.eval()
        for idx in range(out_of_layer.size(1)):
            unit_mask = torch.zeros(out_of_layer.size())
            unit_mask[:, idx] = 1
            unit_mask = unit_mask.cuda()

            # compute basis of this unit
            out_of_layer.backward(unit_mask, retain_graph=True)
            basis = input.grad.clone().detach().reshape(input.size(0), -1)
            basis_list.append(basis)

            # do substraction to get bias
            basis_mul_x = torch.mul(input.clone().detach(), input.grad.clone().detach())
            basis_mul_x = torch.sum(basis_mul_x, dim=(1, 2, 3)).cuda()
            bias = out_of_layer[:, idx].clone().detach() - basis_mul_x
            bias_list.append(bias)

            # clean up
            self.zero_grad()
            input.grad.data.zero_()

        # set false grad flag for input
        input.requires_grad_(False)

        # reshape basis to tensor shape
        stacked_basis   = torch.stack(basis_list, dim=2)
        array_basis    = stacked_basis.detach().cpu().numpy()
        array_basis    = array_basis.reshape(array_basis.shape[0],
                                               array_basis.shape[1],
                                               org_out_size[1],
                                               org_out_size[2],
                                               org_out_size[3])

        # reshape bias to tensor shape
        stacked_bias    = torch.stack(bias_list, dim=1)
        array_bias     = stacked_bias.detach().cpu().numpy()
        array_bias     = array_bias.reshape(array_bias.shape[0],
                                              org_out_size[1],
                                              org_out_size[2],
                                              org_out_size[3])

        out_of_layer = out_of_layer.reshape(org_out_size)

        BB_maxpool = Struct_BB()
        for c in range(org_out_size[-3]):
            conv_coord = ConvCoordinates((org_out_size[-2], org_out_size[-1]), kernel_size, stride, padding)
            while True:
                ulc_pos, pos_list = conv_coord.getNextULCPos()

                if ulc_pos is None:
                    break

                val_list = []
                BB_buf = Struct_BB()
                for pos in pos_list:
                    val = out_of_layer[0:, c:c+1, pos[0], pos[1]]
                    val_list.append(val)

                    tmp_bb = Struct_BB(array_basis[0, 0:, c:c+1, pos[0], pos[1]],
                                       array_bias[0:, c, pos[0], pos[1]])

                    BB_buf.extendBB(tmp_bb)

                val_tensor = torch.cat(val_list, dim=1)
                max_tensor, max_index = torch.max(val_tensor, dim=1, keepdim=True)

                BB_buf.subPivotOverOthers(max_index[0,0].cpu().numpy())
                BB_maxpool.extendBB(BB_buf)

        return BB_maxpool

    def _getBBOfOneLayer(self, input, layer_ptr, layer_start=-1):
        # prepare the buffer
        basis_list = []
        bias_list = []

        # set true grad flag for input
        input.requires_grad_(True)

        out_of_layer = self._getOutputOfOneLayer(input, layer_ptr, layer_start)

        out_of_layer = out_of_layer.reshape(out_of_layer.size(0), -1)

        self.zero_grad()
        self.eval()
        for idx in range(out_of_layer.size(1)):
            unit_mask = torch.zeros(out_of_layer.size())
            unit_mask[:, idx] = 1
            unit_mask = unit_mask.cuda()

            # compute basis of this unit
            out_of_layer.backward(unit_mask, retain_graph=True)
            basis = input.grad.clone().detach().reshape(input.size(0), -1)
            basis_list.append(basis)

            # do substraction to get bias
            basis_mul_x = torch.mul(input.clone().detach(), input.grad.clone().detach())
            # print("basis shape: ", input.shape, basis_mul_x.shape)

            basis_mul_x = torch.sum(basis_mul_x, dim=(1, 2, 3)).cuda()
            bias = out_of_layer[:, idx].clone().detach() - basis_mul_x
            bias_list.append(bias)

            # clean up
            self.zero_grad()
            input.grad.data.zero_()

        # set false grad flag for input
        input.requires_grad_(False)

        # reshape basis to tensor shape
        stacked_basis = torch.stack(basis_list, dim=2)
        array_basis = stacked_basis.detach().squeeze().cpu().numpy()

        # reshape bias to tensor shape
        stacked_bias = torch.stack(bias_list, dim=1)
        array_bias = stacked_bias.detach().squeeze().cpu().numpy()

        return Struct_BB(array_basis, array_bias)

    # ================================================================
    # Following is for saving and loading models
    # ================================================================
    def saveModel(self, model_savepath = None):
        if model_savepath != None:
            print('Saving model to {}'.format(model_savepath))
            torch.save(self.state_dict(), model_savepath)
        else:
            print('Saving model to {}'.format(self.model_savepath))
            torch.save(self.state_dict(), self.model_savepath)

    def loadModel(self, model_savepath = None):
        if model_savepath != None:
            print('Loading model from {}'.format(model_savepath))
            self.load_state_dict(torch.load(model_savepath, map_location='cuda:0'))
        else:
            print('Loading model from {}'.format(self.model_savepath))
            self.load_state_dict(torch.load(self.model_savepath, map_location='cuda:0'))###

    # ================================================================
    # Following is for sanity check of this model4invariant
    # ================================================================
    def _sanityCheck4CNN4Invariant(self, input_, layer_ptr_):
        BB_one_layer        = self._getBBOfOneLayer(input_, layer_ptr_)
        bias_one_layer      = BB_one_layer.getBiasTensor().reshape(1, 2, 28, 28)
        basis_one_layer     = BB_one_layer.getBasisTensor().reshape(784, 2, 28, 28)
        out_one_layer       = self._getOutputOfOneLayer(input_, layer_ptr_)
        cmp_out_one_layer   = torch.sum(torch.mul(basis_one_layer.permute(1,2,3,0), input_.reshape(784)), dim=(3))
        diff_out_one_layer  = torch.sum(torch.abs(out_one_layer - cmp_out_one_layer))
        print('diff_out_one_layer: %.4f' % (diff_out_one_layer)) # do not use this when conv2d has bias

        BB_next_layer       = self._getBBOfOneLayer(input_, layer_ptr_+2)
        bias_next_layer     = BB_next_layer.getBiasTensor().reshape(1, 3, 28, 28)
        basis_next_layer    = BB_next_layer.getBasisTensor().reshape(784, 3, 28, 28)
        out_next_layer      = self._getOutputOfOneLayer(input_, layer_ptr_+2)
        cmp_out_next_layer  = torch.sum(torch.mul(basis_next_layer.permute(1,2,3,0), input_.reshape(784)), dim=(3))
        diff_out_next_layer = torch.sum(torch.abs(out_next_layer - cmp_out_next_layer))
        print('diff_out_next_layer: %.4f' % (diff_out_next_layer)) # do not use this when conv2d has bias

        conv_weight         = self._layers_list[layer_ptr_+1].weight.data.clone().detach()
        conv_bias           = self._layers_list[layer_ptr_+1].bias.data.clone().detach()

        [num_input, fm_dep, fm_row, fm_col] = out_next_layer.size()
        assert(num_input == 1)

        diff_basis = 0
        diff_bias  = 0

        for dep in range(fm_dep):
            for row in range(1, fm_row-1):
                for col in range(1, fm_col-1):
                    if(torch.abs(out_next_layer[0, dep, row, col]) > 0):

                        loc_basis_one_layer     = basis_one_layer[:, :, row-1:row+2, col-1:col+2]
                        cmp_basis_next_layer    = torch.sum(torch.mul(loc_basis_one_layer, conv_weight[dep, :, :, :]), dim=(1,2,3))
                        gt_basis_next_layer     = basis_next_layer[:, dep, row, col]
                        tmp_diff_basis          = torch.sum(torch.abs(cmp_basis_next_layer - gt_basis_next_layer))
                        diff_basis              += tmp_diff_basis
                        #print('tmp diff basis: %.4f' % (tmp_diff_basis))

                        loc_bias_one_layer      = bias_one_layer[:, :, row-1:row+2, col-1:col+2]
                        cmp_bias_next_layer     = torch.sum(torch.mul(loc_bias_one_layer, conv_weight[dep, :, :, :]), dim=(1,2,3)) + conv_bias[dep]
                        gt_bias_next_layer      = bias_next_layer[:, dep, row, col]
                        tmp_diff_bias           = torch.abs(cmp_bias_next_layer - gt_bias_next_layer)
                        diff_bias               += tmp_diff_bias
                        #print('tmp diff bias: %.4f' % (tmp_diff_bias))

        print('==> diff_basis: %.4f' % (diff_basis))
        print('==> diff_bias: %.4f' % (diff_bias))


# ================================================================
# Following are utilities
# ================================================================
class ConvCoordinates():
    def __init__(self, featmap_size_, kernel_size_, stride_, padding_):
        assert(isinstance(featmap_size_, tuple) and featmap_size_.__len__() == 2)
        assert(isinstance(kernel_size_, int))
        assert(isinstance(stride_, int))
        assert(isinstance(padding_, int))

        self._featmap_size = featmap_size_
        self._kernel_size  = kernel_size_
        self._stride       = stride_
        self._padding      = padding_

        self._row_limit     = (-padding_, featmap_size_[0] + padding_ - kernel_size_)
        self._col_limit     = (-padding_, featmap_size_[1] + padding_ - kernel_size_)

        self.initULCPos()

    # get the initial upper left corner
    def initULCPos(self):
        self.ulc_pos = np.array([0,0]) - self._padding

    def getNextULCPos(self):
        if self.ulc_pos is None:
            return None, None

        # log current value
        cur_ulc_pos     = np.array((self.ulc_pos[0], self.ulc_pos[1]))
        cur_pos_list    = self._computeValidPosList()

        # update to next ulc_pos
        self.ulc_pos[1] += self._stride
        if self.ulc_pos[1] > self._col_limit[1]:
            self.ulc_pos[1] = -self._padding
            self.ulc_pos[0] += self._stride
            if self.ulc_pos[0] > self._row_limit[1]:
                self.ulc_pos = None

        return cur_ulc_pos, cur_pos_list

    def _computeValidPosList(self):
        pos_list = []
        for r in range(self.ulc_pos[0], self.ulc_pos[0] + self._kernel_size):
            if r < 0 or r > self._featmap_size[0] - 1:
                continue
            for c in range(self.ulc_pos[1], self.ulc_pos[1] + self._kernel_size):
                if c >= 0 and c <= self._featmap_size[1] - 1:
                    pos_list.append([r, c])

        return pos_list

# this stores the the composed basis and bias of a certain juntion of the neural network.
# BasisBias is 'BB' for short.
class Struct_BB():
    def __init__(self, basis_=None, bias_=None):
        self.bb = None

        if(bias_ is None):
            assert(basis_ is None)
        else:
            assert(basis_ is not None)
            self.importBB(basis_, bias_)

    def importBB(self, basis_, bias_):
        assert(isinstance(basis_, np.ndarray))
        assert(isinstance(bias_, np.ndarray))
        assert(basis_.shape.__len__() == 2)
        assert(bias_.shape.__len__() == 1)

        if self.bb is None:
            self.bb = np.concatenate((basis_, bias_.reshape(1, -1)), axis=0)
        else:
            new_bb = np.concatenate((basis_, bias_.reshape(1, -1)), axis=0)
            self.bb = np.concatenate((self.bb, new_bb), axis=1)

    def extendBB(self, struct_bb_):
        assert(isinstance(struct_bb_, Struct_BB))
        if self.bb is None:
            self.bb = struct_bb_.bb
        else:
            self.bb = np.concatenate((self.bb, struct_bb_.bb), axis=1)

    def getBiasArray(self):
        return self.bb[-1, :]

    def getBasisArray(self):
        return self.bb[:-1, :]

    def getBBArray(self):
        return self.bb

    # userful for extracting bb for pooling layers and last layer
    def subPivotOverOthers(self, pivot_id1_, pivot_id2_):
        #self.bb = self.bb[:, pivot_id_:pivot_id_+1] - self.bb
        #self.bb = np.delete(self.bb, pivot_id_, axis=1)
        self.bb = self.bb[:, pivot_id1_:pivot_id1_+1] - self.bb[:, pivot_id2_:pivot_id2_+1]

    def getSizeOfBB(self):
        if self.bb is None:
            return 0
        else:
            return self.getBiasArray().shape[0]

    def computeHashVal(self, array_features_):
        assert(isinstance(array_features_, np.ndarray))

        expanded_feature = np.concatenate((array_features_, np.ones((array_features_.shape[0], 1))), axis=1)
        hashvals = np.matmul(expanded_feature, self.bb)

        return hashvals

    def computeConfigs(self, array_features_):
        assert (isinstance(array_features_, np.ndarray))

        hashvals = self.computeHashVal(array_features_)
        configs = np.zeros(hashvals.shape)
        configs[hashvals > 0] = 1

        return configs

    def computeSubHashVal(self, invariant_, array_features_):
        assert (isinstance(array_features_, np.ndarray))
        assert (np.sum(invariant_) > 0)

        expanded_feature = np.concatenate((array_features_, np.ones((array_features_.shape[0], 1))), axis=1)

        hashvals = np.matmul(expanded_feature, self.bb[:, invariant_])

        return hashvals

    def computeSubConfigs(self, invariant_, array_features_):
        assert (isinstance(array_features_, np.ndarray))
        assert (np.sum(invariant_) > 0)

        hashvals = self.computeSubHashVal(invariant_, array_features_)
        configs = np.zeros(hashvals.shape)
        configs[hashvals > 0] = 1

        return configs

# the module that changes view of input tensors
class Modu_View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Modu_Linearize(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1).unsqueeze(1)

# ================================================================
# Following are specific models
# ================================================================
# a CNN model for cnn_invariant test
class CNN4Invariant(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 2

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 3
            nn.ReLU(),  # layer 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 5

            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 6
            nn.ReLU(),  # layer 7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 8

            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 9
            nn.ReLU(),  # layer 10
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 11

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 12
            nn.ReLU(),  # layer 13
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),   # layer 14

            # use a conv2d layer to equivalently replace the fc layer
            nn.Conv2d(4, self._num_classes, kernel_size=3, stride=1, padding=0, bias=self.has_bias) # layer 15
        ])

class CNN4CIFAR(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 2

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 3
            nn.ReLU(),  # layer 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 5

            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 6
            nn.ReLU(),  # layer 7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 8

            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 9
            nn.ReLU(),  # layer 10
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 11

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 12
            nn.ReLU(),  # layer 13
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),   # layer 14

            # use a conv2d layer to equivalently replace the fc layer
            nn.Conv2d(4, self._num_classes, kernel_size=3, stride=1, padding=0, bias=self.has_bias) # layer 15
        ])

class CNN4Caltech(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 2

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 3
            nn.ReLU(),  # layer 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 5

            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 6
            nn.ReLU(),  # layer 7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 8

            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 9
            nn.ReLU(),  # layer 10
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 11

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 12
            nn.ReLU(),  # layer 13
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),   # layer 14

            nn.Linear(4 * 9 * 17, 2) # layer 15
        ])

class CNN4Kaggle(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 2

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 3
            nn.ReLU(),  # layer 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 5

            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 6
            nn.ReLU(),  # layer 7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 8

            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 9
            nn.ReLU(),  # layer 10
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 11

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 12
            nn.ReLU(),  # layer 13
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),   # layer 14

            nn.Linear(1156, 2) # layer 15
        ])

class CNN_AvgPool(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),

            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 2
            nn.ReLU(),  # layer 3
            nn.MaxPool2d(kernel_size=7, stride=1, padding=0),   # layer 4

            nn.Conv2d(5, 5, kernel_size=1, stride=1, padding=0, bias=self.has_bias),  # layer 5
            nn.ReLU(),  # layer 6

            # use a conv2d layer to equivalently replace the fc layer
            nn.Conv2d(5, self._num_classes, kernel_size=1, stride=1, padding=0, bias=self.has_bias) # layer 7
        ])

class CNN_AvgPool_Small(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(1, 10, kernel_size=2, stride=2, padding=0, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 2
            nn.ReLU(),  # layer 3
            nn.MaxPool2d(kernel_size=7, stride=1, padding=0),   # layer 4

            #nn.Conv2d(10, 10, kernel_size=1, stride=1, padding=0, bias=self.has_bias),  # layer 5
            #nn.ReLU(),  # layer 6

            # use a conv2d layer to equivalently replace the fc layer
            nn.Conv2d(5, self._num_classes, kernel_size=1, stride=1, padding=0, bias=self.has_bias) # layer 7
        ])

class CNN_AvgPool_Small2(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(1, 2, kernel_size=4, stride=4, padding=0, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=0, bias=self.has_bias),    # layer 2
            nn.ReLU(),  # layer 3
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),   # layer 4

            #nn.Conv2d(10, 10, kernel_size=1, stride=1, padding=0, bias=self.has_bias),  # layer 5
            #nn.ReLU(),  # layer 6

            # use a conv2d layer to equivalently replace the fc layer
            nn.Conv2d(2, self._num_classes, kernel_size=1, stride=1, padding=0, bias=self.has_bias) # layer 7
        ])

# a MLP model for mlp_invariant test
class MLP4Invariant(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl' # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        self._layers_list = nn.ModuleList([
            Modu_Linearize(),       # layer 0

            nn.Linear(784, 20),    # layer 1
            nn.ReLU(),              # layer 2
            #nn.AvgPool1d(kernel_size=10, stride=10, padding=0),

            nn.Linear(20, 8),     # layer 3
            nn.ReLU(),              # layer 4
            #nn.AvgPool1d(kernel_size=4, stride=4, padding=0),

            nn.Linear(8, self._num_classes)    # layer 5
        ])

# a MLP model for mlp_invariant test
class MLP_Large(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl' # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        self._layers_list = nn.ModuleList([
            Modu_Linearize(),       # layer 0

            nn.Linear(784, 1000),    # layer 1
            nn.ReLU(),              # layer 2

            nn.Linear(1000, 200),     # layer 3
            nn.ReLU(),              # layer 4

            nn.Linear(200, 30),  # layer 5
            nn.ReLU(),          # layer 6

            nn.Linear(30, self._num_classes)    # layer 7
        ])

# a MLP model to 2D input
class MLP2D(BaseModel):
    def __init__(self, num_classes_=2, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl' # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        self._layers_list = nn.ModuleList([
            Modu_Linearize(),   # layer 0

            nn.Linear(2, 100),   # layer 1
            nn.ReLU(),          # layer 2

            nn.Linear(100, 50),   # layer 3
            nn.ReLU(),          # layer 4

            nn.Linear(50, 10),    # layer 5
            nn.ReLU(),          # layer 6

            nn.Linear(10, self._num_classes) # layer 7
        ])

class VGG19(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 2
            nn.ReLU(),  # layer 3
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 5
            nn.ReLU(),  # layer 6

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 7
            nn.ReLU(),  # layer 8
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 9

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 10
            nn.ReLU(),  # layer 11

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 12
            nn.ReLU(),  # layer 13

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 14
            nn.ReLU(),  # layer 15

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 16
            nn.ReLU(),  # layer 17
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 18

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 19
            nn.ReLU(),  # layer 20

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 21
            nn.ReLU(),  # layer 22

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 23
            nn.ReLU(),  # layer 24

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 25
            nn.ReLU(),  # layer 26
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 27

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 28
            nn.ReLU(),  # layer 29

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 30
            nn.ReLU(),  # layer 31

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 32
            nn.ReLU(),  # layer 33

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 34
            nn.ReLU(),  # layer 35

            nn.BatchNorm1d(100352),

            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 36

            nn.AdaptiveAvgPool2d(output_size=(7,7)),

            nn.Linear(25088, 4096), # layer 0
            nn.ReLU(), # layer 1
            nn.Dropout(p=0.5), # layer 2
            nn.Linear(4096, 4096), # layer 3
            nn.ReLU(), # layer 4
            nn.Dropout(p=0.5), # layer 5
            nn.Linear(4096, self._num_classes) # layer 6

        ])

class CNN4MNIST(BaseModel):
    def __init__(self, num_classes_, has_bias_=True):
        super().__init__()

        self.has_bias = has_bias_    # the bool flag for bias
        self.model_savepath = self._model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes_
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 2

            nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 3
            nn.ReLU(),  # layer 4
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 5

            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 6
            nn.ReLU(),  # layer 7
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 8

            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 9
            nn.ReLU(),  # layer 10
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # layer 11

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=self.has_bias),    # layer 12
            nn.ReLU(),  # layer 13
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),   # layer 14

            nn.Linear(36, self._num_classes) # layer 15
        ])


import torch
import torch.nn as nn
from collections import OrderedDict
import math
from torch.utils.data import DataLoader
from dnn_invariant.utilities.environ import *

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        # parameters for getBasisOfNextPatch
        self.is_patch_getter_ready = False

    def forward(self, input):
        # loop over all modules in _layers_list
        out = input
        for i in range(self._layers_list.__len__()):
            out = self._layers_list[i](out)
        out = out.reshape(out.size(0), -1)

        return out

    # ======================================================
    # Utilities
    # ======================================================
    def _setLayerList(self):
        raise NotImplementedError

    # get the number of ReLU layers
    def getNumPoolingLayers(self):
        num_Pooling = 0
        for idx, m in enumerate(self.modules()):
            if(isinstance(m, torch.nn.modules.pooling.MaxPool2d)):
                num_Pooling += 1

        return num_Pooling

    def _checkPoolingLayerPtr(self, layer_ptr):
        # layer_ptr = 0 points to the first Pooling layer
        # layer_ptr = num_Pooling -1 points to the last Pooling layer
        assert (layer_ptr >= 0)
        assert (layer_ptr <= self.getNumPoolingLayers()-1)

    # return the kernel weights and paddings of the conv layer after the Pooling of layer_ptr
    # layer_ptr only points to pooling layer
    def _getKernelModule(self, layer_ptr):
        m_module = None
        num_Pooling = 0

        for idx, m in enumerate(self.modules()):
            if (isinstance(m, torch.nn.modules.pooling.MaxPool2d)):
                num_Pooling += 1
            if (num_Pooling > layer_ptr):
                if (isinstance(m, torch.nn.modules.conv.Conv2d)):
                    m_module = m
                    break

        return m_module

    def _getKernelWeightsPaddingsBias(self, layer_ptr):
        m_module = self._getKernelModule(layer_ptr)
        if(m_module is None):
            return [None, None, None]
        else:
            return [m_module.weight, m_module.padding, m_module.bias]

    def getKernelWeights(self, layer_ptr):
        [kernel_weights, _, _] = self._getKernelWeightsPaddingsBias(layer_ptr)
        return kernel_weights

    def getKernelWeightsSize(self, layer_ptr):
        [kernel_weights, _, _] = self._getKernelWeightsPaddingsBias(layer_ptr)
        return kernel_weights.size()

    def setKernelWeights(self, layer_ptr, new_ker_weight):
        m_module = self._getKernelModule(layer_ptr)
        m_module.weight.data.copy_(new_ker_weight.detach().clone())

    def getKernelBias(self, layer_ptr):
        [_, _, kernel_bias] = self._getKernelWeightsPaddingsBias(layer_ptr)
        return kernel_bias

    # ======================================================
    # Following is for getting the basis vectors of units
    # ======================================================

    # This method do initialization for the getBasisOfNextPatch
    # It must be called before calling getBasisOfNextPatch
    def initPatchGetter(self, input, layer_ptr, stride=1):
        self._clearMemo()

        self.input = input
        self.input.requires_grad_(True)

        self.stride = stride
        self.kernel_loc = [0, 0]  # this is the starting location of the kernel patch

        self.out_pooling = self._getOutputOfPoolingLayer(input, layer_ptr)
        self.out_conv = self._getOutputOfConvLayer(input, layer_ptr)

        # configure kernel and padding info
        [kernel_weights, kernel_padding, _] = self._getKernelWeightsPaddingsBias(layer_ptr)
        self.kernel_weights = kernel_weights.clone().detach()
        self.kernel_padding = kernel_padding

        assert(kernel_weights.size(2) == kernel_weights.size(3)) # assert symmetric kernel
        self.ker_offset = math.floor(kernel_weights.size(2) / 2.0)

        assert (kernel_padding[0] == kernel_padding[1]) # assert symmetric padding
        self.padding_size = kernel_padding[0]

        # mark basis getter ready
        self.is_patch_getter_ready = True

    # release some large objects to free memory
    # use in the begining of initPatchGetter
    def _clearMemo(self):
        if(hasattr(self, 'input') and self.input is not None):
            del self.input
            self.input = None

        if(hasattr(self, 'stride') and self.stride is not None):
            del self.stride
            self.stride = None

        if(hasattr(self, 'kernel_loc') and self.kernel_loc is not None):
            del self.kernel_loc
            self.kernel_loc = None

        if(hasattr(self, 'out_pooling') and self.out_pooling is not None):
            del self.out_pooling
            self.out_pooling = None

        if(hasattr(self, 'out_conv') and self.out_conv is not None):
            del self.out_conv
            self.out_conv = None

        if(hasattr(self, 'kernel_weights') and self.kernel_weights is not None):
            del self.kernel_weights
            self.kernel_weights = None

        if(hasattr(self, 'kernel_padding') and self.kernel_padding is not None):
            del self.kernel_padding
            self.kernel_padding = None

        if(hasattr(self, 'ker_offest') and self.ker_offset is not None):
            del self.ker_offset
            self.ker_offset = None

        if(hasattr(self, 'padding_size') and self.padding_size is not None):
            del self.padding_size
            self.padding_size = None

        if(hasattr(self, 'is_patch_getter_ready') and self.is_patch_getter_ready is not None):
            del self.is_patch_getter_ready
            self.is_patch_getter_ready = None


    # update the self.kernel_loc for the next patch in a convlutional manner
    def _updateKernelLoc(self):
        fm_row = self.out_pooling.size(2)
        fm_col = self.out_pooling.size(3)

        # check validity of current kernel_loc
        if(self.kernel_loc[0] < 0 or self.kernel_loc[0] >= fm_row):
            return False
        if(self.kernel_loc[1] < 0 or self.kernel_loc[1] >= fm_col):
            return False

        # update kernel_loc
        self.kernel_loc[1] += self.stride

        if(self.kernel_loc[1] >= fm_col):
            self.kernel_loc[0] += self.stride
            self.kernel_loc[1] = 0

        # check validity of updated kernel_loc
        if(self.kernel_loc[0] < 0 or self.kernel_loc[0] >= fm_row):
            return False
        else:
            return True # update is sucessful

    # this method returns the tensors of pooling_basis and conv_basis of the next patch
    def getBasisOfNextPatch(self):
        assert(self.is_patch_getter_ready == True)

        pooling_basis = self._getBasisesPoolingOut()
        conv_basis = self._getBasisesConvOut()

        # move to the next patch
        has_next = self._updateKernelLoc()

        # check ending of the convlution
        if(has_next == False):
            # turn off input grad buffer
            self.input.requires_grad_(False)

            # mark basis getter ready
            self.is_patch_getter_ready = False

        return [pooling_basis, conv_basis, has_next]

    def skipBasisOfNextPatch(self):
        assert (self.is_patch_getter_ready == True)

        # move to the next patch
        has_next = self._updateKernelLoc()

        # check ending of the convlution
        if (has_next == False):
            # turn off input grad buffer
            self.input.requires_grad_(False)

            # mark basis getter ready
            self.is_patch_getter_ready = False

        return has_next

    # get the output of the pooling layer pointed by layer_ptr
    def _getOutputOfPoolingLayer(self, input, layer_ptr):
        self.eval()
        self._checkPoolingLayerPtr(layer_ptr)

        num_Pooling = 0
        out = input
        for i in range(self._layers_list.__len__()):
            out = self._layers_list[i](out)
            if(isinstance(self._layers_list[i], torch.nn.modules.pooling.MaxPool2d)):
                num_Pooling += 1
            if(num_Pooling > layer_ptr):
                break
        return out

    # kernel_loc points to the pixel of feature map that aligns with center of the kernel
    # the index of pixels starts from zero.
    # gets the basis w.r.t. the output of the pooling layer pointed by layer_ptr
    def _getBasisesPoolingOut(self):
        basis_list = [] # this is a list of tensors

        # compute the range of pixels
        range_row = range(self.kernel_loc[0] - self.ker_offset, self.kernel_loc[0] + self.ker_offset + 1)
        range_col = range(self.kernel_loc[1] - self.ker_offset, self.kernel_loc[1] + self.ker_offset + 1)

        fm_dep = self.out_pooling.size(1)
        fm_row = self.out_pooling.size(2)
        fm_col = self.out_pooling.size(3)

        for dep_idx in range(fm_dep):
            for row in range_row:
                for col in range_col:
                    assert(row >= -self.padding_size and row < fm_row + self.padding_size)
                    assert(col >= -self.padding_size and col < fm_col + self.padding_size)

                    if(row >= 0 and row < fm_row and col >= 0 and col < fm_col):
                        unit_mask = torch.zeros(self.out_pooling.size())
                        unit_mask[:, dep_idx, row, col] = 1
                        unit_mask = unit_mask.cuda()

                        self.out_pooling.backward(unit_mask, retain_graph=True)
                        basis = self.input.grad.clone().detach().reshape(self.input.size(0), -1)
                        basis_list.append(basis)

                        # clean up
                        self.zero_grad()
                        self.input.grad.data.zero_()
                    else:
                        basis = torch.zeros(self.input.reshape(self.input.size(0), -1).size())
                        basis = basis.cuda()
                        basis_list.append(basis)

                        # clean up
                        self.zero_grad()

        # reshape basis to tensor shape
        stacked_basis = torch.stack(basis_list, dim=2)
        tensor_basis = stacked_basis.reshape((stacked_basis.size(0), stacked_basis.size(1), fm_dep, range_row.__len__(), range_col.__len__()))
        tensor_basis = tensor_basis.detach()

        return tensor_basis

    # gets the output of the conv layer behind the pooling layer pointed by layer_ptr
    def _getOutputOfConvLayer(self, input, layer_ptr):
        self.eval()
        self._checkPoolingLayerPtr(layer_ptr)

        num_Pooling = 0
        out = input
        for i in range(self._layers_list.__len__()):
            out = self._layers_list[i](out)
            if (isinstance(self._layers_list[i], torch.nn.modules.pooling.MaxPool2d)):
                num_Pooling += 1
            if (num_Pooling > layer_ptr and isinstance(self._layers_list[i], torch.nn.modules.conv.Conv2d)):
                break
        return out

    # kernel_loc points to the pixel of feature map that aligns with center of the kernel
    # the index of pixels starts from zero.
    # gets the basis w.r.t. the output of the conv layer behind the pooling layer pointed by layer_ptr
    def _getBasisesConvOut(self):
        basis_list = []  # this is a list of tensors

        fm_dep = self.out_conv.size(1)

        for dep_idx in range(fm_dep):
            unit_mask = torch.zeros(self.out_conv.size())
            unit_mask[:, dep_idx, self.kernel_loc[0], self.kernel_loc[1]] = 1
            unit_mask = unit_mask.cuda()

            self.out_conv.backward(unit_mask, retain_graph=True)
            basis_list.append(self.input.grad.clone().detach().reshape(self.input.size(0), -1))

            # clean up
            self.zero_grad()
            self.input.grad.data.zero_()

        tensor_basis = torch.stack(basis_list, dim=2)
        tensor_basis = tensor_basis.detach()

        return tensor_basis

    # evaluates the prediction accuracy of this model on the input "data"
    # LEON POSSIBLE BUG: cannot call "to(device)" in this method, otherwise the optimizer will act strangely.
    def evalAccuracy(self, data, batch_size=100):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        self.cuda(device=CUDA_DEVICE)
        with torch.no_grad():
            total = correct = 0
            for instances, labels in data_loader:
                outputs = self(instances.cuda(device=CUDA_DEVICE))
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels.cuda(device=CUDA_DEVICE)).sum().item()

        self.train()
        torch.enable_grad()

        return 1.0*correct/total

    # ================================================================
    # Utilities for baseline CP
    # ================================================================
    # this method returns the feature maps of pooling_basis and conv_basis of the next patch
    def getFeatMapOfNextPatch(self):
        assert(self.is_patch_getter_ready == True)

        pooling_featmap = self._getFeatMapPoolingOut()
        conv_featmap = self._getFeatMapConvOut()

        # move to the next patch
        has_next = self._updateKernelLoc()

        # check ending of the convlution
        if(has_next == False):
            # turn off input grad buffer
            self.input.requires_grad_(False)

            # mark basis getter ready
            self.is_patch_getter_ready = False

        return [pooling_featmap, conv_featmap, has_next]

    def skipFeatMapOfNextPatch(self):
        assert (self.is_patch_getter_ready == True)

        # move to the next patch
        has_next = self._updateKernelLoc()

        # check ending of the convlution
        if (has_next == False):
            # turn off input grad buffer
            self.input.requires_grad_(False)

            # mark basis getter ready
            self.is_patch_getter_ready = False

        return has_next

    # kernel_loc points to the pixel of feature map that aligns with center of the kernel
    # the index of pixels starts from zero.
    # gets the feature map of the pooling layer pointed by layer_ptr
    def _getFeatMapPoolingOut(self):
        # compute the range of pixels
        min_row = self.kernel_loc[0] - self.ker_offset
        max_row = self.kernel_loc[0] + self.ker_offset

        min_col = self.kernel_loc[1] - self.ker_offset
        max_col = self.kernel_loc[1] + self.ker_offset

        range_row = range(min_row, max_row + 1)
        range_col = range(min_col, max_col + 1)

        num_instances   = self.out_pooling.size(0)
        fm_dep          = self.out_pooling.size(1)
        fm_row          = self.out_pooling.size(2)
        fm_col          = self.out_pooling.size(3)

        feat_map = torch.zeros(num_instances, fm_dep, range_row.__len__(), range_col.__len__())

        for dep_idx in range(fm_dep):
            for row in range_row:
                for col in range_col:
                    assert (row >= -self.padding_size and row < fm_row + self.padding_size)
                    assert (col >= -self.padding_size and col < fm_col + self.padding_size)

                    if (row >= 0 and row < fm_row and col >= 0 and col < fm_col):
                        feat_map[:, dep_idx, row - min_row, col - min_col] = self.out_pooling[:, dep_idx, row, col].clone().detach()

        # reshape basis to tensor shape
        return feat_map

    # kernel_loc points to the pixel of feature map that aligns with center of the kernel
    # the index of pixels starts from zero.
    # gets the feature map of the conv layer behind the pooling layer pointed by layer_ptr
    def _getFeatMapConvOut(self):
        basis_list = []  # this is a list of tensors

        num_instances   = self.out_conv.size(0)
        fm_dep          = self.out_conv.size(1)

        feat_map = torch.zeros(num_instances, fm_dep)

        for dep_idx in range(fm_dep):
            feat_map[:, dep_idx] = self.out_conv[:, dep_idx, self.kernel_loc[0], self.kernel_loc[1]].clone().detach()

        return feat_map

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
            self.load_state_dict(torch.load(self.model_savepath, map_location='cuda:0'))


# ================================================================
# Following are specific models
# ================================================================
class MLP_FMNIST(BaseModel):
    def __init__(self, num_classes):
        super().__init__()

        self.model_savepath = './' + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(784, 1000)),
                ('relu', nn.ReLU())
            ])),

            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(1000, 200)),
                ('relu', nn.ReLU())
            ])),

            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(200, 40)),
                ('relu', nn.ReLU())
            ])),

            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(40, 20)),
                ('relu', nn.ReLU())
            ])),

            nn.Linear(20, self._num_classes)]
        )

class TEST_MLP(BaseModel):
    def __init__(self, num_classes):
        super().__init__()

        self.model_savepath = './' + self.__class__.__name__ + '.mdl'  # default path to save the model
        self._num_classes = num_classes
        self._setLayerList()

    def _setLayerList(self):
        # set the layers of the model in a list
        self._layers_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(784, 7)),
                ('relu', nn.ReLU())
            ])),

            nn.Sequential(OrderedDict([
                ('fc', nn.Linear(7, 3)),
                ('relu', nn.ReLU())
            ])),

            nn.Linear(3, self._num_classes)]
        )

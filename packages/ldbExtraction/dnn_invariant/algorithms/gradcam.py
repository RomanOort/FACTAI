import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from dnn_invariant.utilities.datasets import *

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Denormalize a 3 x M x N numpy image for visualization
def process_img(img, ImageNetNormalize):
    # the input of this is either a 3xMxN image, or MxN image
    # the output should be a 3xMxN image or 1xMxN image
    # if there was an ImageNet normalization, it will be undone

    if img.ndim == 2:
        img = img[np.newaxis, :]

    if len(img[:, 0, 0]) == 1:
        return img

    if ImageNetNormalize:
        for i in range(3):
            img[i, :, :] = img[i, :, :] * std[i] + mean[i]

    return img

'''
# Normalize a 3 x M x N numpy image for classification
def deprocess_img(img):
    if img.ndim == 2:
        img = img[np.newaxis, :]

    if len(img[:, 0, 0]) == 1:
        return img

    for i in range(3):
        img[i, :, :] = (img[i, :, :] - mean[i]) / std[i]

    return img

# Classify a 3 x M x N RGB numpy image. Input pixel values are between 0 and 1. The image needs to be normalized before passing it into the model.
def model_output_numpy(model, img):
    img = deprocess_img(img)
    t = torch.from_numpy(img).cuda()
    t = t.view(1, C, H, W)
    t = t.float()
    return nn.Softmax(dim=1)(model(t).double())
'''

def find_last_conv(model):
    last_conv = -1
    for i in range(len(model._layers_list)):
        layer = model._layers_list.__getitem__(i)
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            last_conv = i

    assert last_conv > -1
    return last_conv

def find_first_linear(model):
    first_linear = -1
    for i in range(len(model._layers_list)):
        layer = model._layers_list.__getitem__(i)
        if isinstance(layer, torch.nn.modules.linear.Linear):
            first_linear = i
            break

    assert first_linear > -1
    return first_linear

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, layer_str, target_layers, is_graph=False):
        self.model = model
        self.layer_str = layer_str
        self.target_layers = target_layers
        self.gradients = []
        self.is_graph=is_graph

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        x.requires_grad = True

        if self.layer_str == self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]

        # print("target: ", self.target_layers)
        if self.is_graph:
            return outputs,x

        for name in range(self.layer_str + 1, find_first_linear(self.model)):
            module = self.model._modules["_layers_list"].__getitem__(name)

            if module.__class__.__name__ == 'BatchNorm1d':
                o_shape = x.shape
                x = x.reshape(o_shape[0], -1)
                x = module(x)
                x = x.reshape(o_shape[0], o_shape[1], o_shape[2], o_shape[3])
            else:
                x = module(x)

            if name == self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x # outputs is a list of one element, containing the output of the targeted layer [1, 512, 14, 14]. x is the output of all layers except FC

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, layer_str, target_layers, is_graph=False):
        self.model = model
        self.is_graph = is_graph
        self.feature_extractor = FeatureExtractor(self.model, layer_str, target_layers,is_graph=self.is_graph)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x) #output is feature map just before 1st fc
        #target_activations and output are same for graph but different for vgg

        output = output.view(output.size(0), -1) # flatten to a 512 * 7 * 7 dim vector

        self.model.eval()
        if self.is_graph:
            output = self.model._getOutputOfOneLayer(output) #get prediction
        else:
            for name in range(find_first_linear(self.model), len(self.model._modules["_layers_list"])):
                module = self.model._modules["_layers_list"].__getitem__(name)
                output = module(output)

        return target_activations, output # targeted_activiations is a list the output of targeted layer, output is the probability vector?

class GradCam:
    def __init__(self, model, layer_str, target_layer_names, is_graph=False):
        self.model = model.cuda()
        self.model.eval()
        self.is_graph = is_graph

        self.extractor = ModelOutputs(self.model, layer_str, target_layer_names, is_graph=self.is_graph) #I is 2nd and 3rd argument

        # Only needed when intend to use Guided Grad-CAM
        '''
        for i in range(45):
            if self.model._modules["_layers_list"].__getitem__(i).__class__.__name__ == 'GuidedBackpropReLU_mod':
                self.model._layers_list[i] = nn.ReLU()
        '''

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, input2, inv=1, input_exp=None):
        _, output = self.extractor(input.cuda()) #output is final prediction

        # features is a list of one tensor
        # this tensor has size [1, 512, 14, 14], which is the output of the targeted layer

        # output is a tensor of size [1, 1000] containing "prediction probability", without softmax

        output_numpy = output.cpu().data.numpy()
        index = np.argmax(output_numpy) #pred class
        #index_opp = np.argmin(output.cpu().data.numpy())
        index_opp = output_numpy[0].argsort()[-2] #2nd likely class for pivot

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  #pred class is made 1

        if not torch.all(torch.eq(input, input2)):
            one_hot[0][index_opp] = -1

        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output) #1st score - 2nd score

        # one_hot is just one number (tensor), the "prediction probability" for the class
        if self.is_graph:
            self.model.zero_grad()
        else:

            for i in range(len(self.model._modules["_layers_list"])):
                self.model._modules["_layers_list"].__getitem__(i).zero_grad()

        one_hot.backward(retain_graph=True) #gradient of p1-p2 w.r.t features
        grads_l = self.extractor.get_gradients()
        grads_val = grads_l[-1].cpu().data.numpy()  #1*20
        # print("grads val: ", grads_val)


        features, _ = self.extractor(input2.cuda())

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        # target is the output of targeted layer

        if self.is_graph:
            weights = grads_val #1*20 dim vector
            weights = np.expand_dims(weights, axis=1)
            weights = np.repeat(weights,input_exp.shape[1],axis=1) #1*nodes*20
            node_weights = input_exp*weights
            node_weights = np.maximum(np.sum(node_weights,axis=2), 0.0) #1*nodes
            return node_weights[0]




        weights = np.mean(grads_val, axis=(2, 3))[0, :] # 512 dimensional vector
        cam = np.zeros(target.shape[1:], dtype=np.float32) # 14x14 vector
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam * inv
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (H, W))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam


def Rule_GradCam(model, img, layer_str = -1):

    grad_cam = GradCam(model=model, layer_str = layer_str, target_layer_names=find_last_conv(model))
    mask = grad_cam(img, img, 1)
    return mask, show_cam_on_image(img, mask)

def Boundary_Visualization(model, img, apply, inv = 1, layer_str = -1):

    grad_cam = GradCam(model=model, layer_str = layer_str, target_layer_names = layer_str)
    mask = grad_cam(img, apply, inv)
    return mask

def Boundary_Visualization_Graph(model, img, apply, input_exp, inv = 1, layer_str = -1):

    grad_cam = GradCam(model=model, layer_str = layer_str, target_layer_names = layer_str, is_graph=True)
    mask = grad_cam(img, apply, inv, input_exp)
    return mask

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img = img.cpu().data.numpy() # this is 1x3xMxN for RGB or 1xMxN for BW image
    img = np.squeeze(img) # 3xMxN or MxN now
    img = process_img(img, ImageNetNormalize) # 3xMxN or 1xMxN now, without ImageNet normalization
    img = np.moveaxis(img, 0, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.max(img) > 100:
        img = img / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (output_size, output_size))
    return cam

def show_heatmap_on_image(img, heatmap):
    img = img.cpu().data.numpy() # this is 1x3xMxN for RGB or 1xMxN for BW image
    img = np.squeeze(img) # 3xMxN or MxN now
    img = process_img(img, ImageNetNormalize) # 3xMxN or 1xMxN now, without ImageNet normalization
    img = np.moveaxis(img, 0, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.max(img) > 100:
        img = img / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (output_size, output_size))
    return cam

def show_box_on_image(img, mask):
    cutoff = np.quantile(mask, 0.85)
    Range = mask > cutoff
    left, right = np.min(np.sum(Range, axis=0).nonzero()), np.max(np.sum(Range, axis=0).nonzero())
    top, bottom = np.min(np.sum(Range, axis=1).nonzero()), np.max(np.sum(Range, axis=1).nonzero())

    img = img.cpu().data.numpy()
    img = np.squeeze(img)
    img = process_img(img)

    yellow = np.array([1, 1, 0])

    for i in range(3):
        for w in range(left, right + 1):
            img[i, top, w] = yellow[i]
            img[i, bottom, w] = yellow[i]
        for h in range(top, bottom + 1):
            img[i, h, left] = yellow[i]
            img[i, h, right] = yellow[i]

    #img = np.moveaxis(img, 0, -1)
    #img = cv2.resize(img, (600, 600))
    return img



















'''
def Rule_GradCam_Mask_Layer(model, img, id, layer_str = -1):

    grad_cam = GradCam(model=model, layer_str = layer_str, target_layer_names=id, use_cuda=True)
    target_index = None
    mask = grad_cam(img, img, target_index)
    return mask
'''





'''
class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLU_mod(nn.Module):
    def forward(self, input):
        return GuidedBackpropReLU.apply(input)



class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for i in range(45):
            if self.model._modules["_layers_list"].__getitem__(i).__class__.__name__ == 'ReLU':
                self.model._layers_list[i] = GuidedBackpropReLU_mod()


    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        input.requires_grad_(True)

        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        for i in range(45):
            self.model._modules["_layers_list"].__getitem__(i).zero_grad()

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def Rule_GuidedGradCam_Mask(model, mask, img):

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(img, index=None)

    #gb = gb.transpose((1, 2, 0))
    gb = np.moveaxis(gb, 0, -1)

    gb = gb - np.mean(gb)
    gb = gb / (np.std(gb) + 1e-5)
    gb = gb * 0.1
    gb = gb + 0.5
    gb = np.clip(gb, 0, 1)

    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = cam_mask * gb

    gb = cv2.resize(gb, (600, 600))
    gb = cv2.cvtColor(gb, cv2.COLOR_BGR2RGB)

    #cam_gb = np.moveaxis(cam_gb, 0, -1)
    cam_gb = cv2.resize(cam_gb, (600, 600))
    cam_gb = cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB)

    return gb, cam_gb
'''
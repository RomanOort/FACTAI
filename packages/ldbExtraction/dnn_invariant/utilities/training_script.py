

import sys
print(sys.path)
sys.path.insert(0, "/home/mohit/Mohit/model_interpretation/ai-adversarial-detection")

from dnn_invariant.models.models4invariant import *
from dnn_invariant.utilities.trainer import *
from dnn_invariant.utilities.datasets import *
from dnn_invariant.utilities.environ import *
import numpy as np

np.set_printoptions(threshold=np.inf, precision=20)
np.random.seed(0)
torch.set_printoptions(precision=6)
torch.manual_seed(0)

'''
Choose the model you want and the name for the saved model. You can create your own model in models/models4invariant.py
'''
#model = VGG19(num_classes_=4)
model = VGG19(num_classes_=2)
# model = CNN4MNIST(num_classes_=2)

#model_name = 'ZhangLab_3epochs.mdl'
model_name = 'Assira_3epochs.mdl'
#model_name = 'FMNIST_24.mdl'
# model_name = 'MNIST_24.mdl'



'''
Loading weights: If your model has already been trained by training_script.py and you want to continue training (rarely used)
'''
#model.loadModel()



'''
Loading weights: If you haven't done any training, and want to begin with pretrained ImageNet weights
We have to transfer the weights in such a silly way because the pretrained model by PyTorch and our model are two very different Python classes
'''
#load_pretrained_ImageNet = True
load_pretrained_ImageNet = True

load_pretrained_ImageNet_path = 'dnn_invariant/models/VGG19.mdl'



'''
number of epochs
'''
#num_epoch = 3
num_epoch = 30











'''
Training about to begin
'''

if load_pretrained_ImageNet:
    # model2 = torchvision.models.vgg19(pretrained=True).cuda()
    model2 = torch.load(load_pretrained_ImageNet_path, map_location='cuda:0')
    with torch.no_grad():
        model._layers_list[0].weight = torch.nn.parameter.Parameter(model2['features.0.weight'])
        model._layers_list[0].bias = torch.nn.parameter.Parameter(model2['features.0.bias'])
        model._layers_list[2].weight = torch.nn.parameter.Parameter(model2['features.2.weight'])
        model._layers_list[2].bias = torch.nn.parameter.Parameter(model2['features.2.bias'])
        model._layers_list[5].weight = torch.nn.parameter.Parameter(model2['features.5.weight'])
        model._layers_list[5].bias = torch.nn.parameter.Parameter(model2['features.5.bias'])
        model._layers_list[7].weight = torch.nn.parameter.Parameter(model2['features.7.weight'])
        model._layers_list[7].bias = torch.nn.parameter.Parameter(model2['features.7.bias'])
        model._layers_list[10].weight = torch.nn.parameter.Parameter(model2['features.10.weight'])
        model._layers_list[10].bias = torch.nn.parameter.Parameter(model2['features.10.bias'])
        model._layers_list[12].weight = torch.nn.parameter.Parameter(model2['features.12.weight'])
        model._layers_list[12].bias = torch.nn.parameter.Parameter(model2['features.12.bias'])
        model._layers_list[14].weight = torch.nn.parameter.Parameter(model2['features.14.weight'])
        model._layers_list[14].bias = torch.nn.parameter.Parameter(model2['features.14.bias'])
        model._layers_list[16].weight = torch.nn.parameter.Parameter(model2['features.16.weight'])
        model._layers_list[16].bias = torch.nn.parameter.Parameter(model2['features.16.bias'])
        model._layers_list[19].weight = torch.nn.parameter.Parameter(model2['features.19.weight'])
        model._layers_list[19].bias = torch.nn.parameter.Parameter(model2['features.19.bias'])
        model._layers_list[21].weight = torch.nn.parameter.Parameter(model2['features.21.weight'])
        model._layers_list[21].bias = torch.nn.parameter.Parameter(model2['features.21.bias'])
        model._layers_list[23].weight = torch.nn.parameter.Parameter(model2['features.23.weight'])
        model._layers_list[23].bias = torch.nn.parameter.Parameter(model2['features.23.bias'])
        model._layers_list[25].weight = torch.nn.parameter.Parameter(model2['features.25.weight'])
        model._layers_list[25].bias = torch.nn.parameter.Parameter(model2['features.25.bias'])
        model._layers_list[28].weight = torch.nn.parameter.Parameter(model2['features.28.weight'])
        model._layers_list[28].bias = torch.nn.parameter.Parameter(model2['features.28.bias'])
        model._layers_list[30].weight = torch.nn.parameter.Parameter(model2['features.30.weight'])
        model._layers_list[30].bias = torch.nn.parameter.Parameter(model2['features.30.bias'])
        model._layers_list[32].weight = torch.nn.parameter.Parameter(model2['features.32.weight'])
        model._layers_list[32].bias = torch.nn.parameter.Parameter(model2['features.32.bias'])
        model._layers_list[34].weight = torch.nn.parameter.Parameter(model2['features.34.weight'])
        model._layers_list[34].bias = torch.nn.parameter.Parameter(model2['features.34.bias'])

        # model._layers_list[38].weight = torch.nn.parameter.Parameter(model2['classifier.0.weight'])
        # model._layers_list[38].bias = torch.nn.parameter.Parameter(model2['classifier.0.bias'])
        # model._layers_list[41].weight = torch.nn.parameter.Parameter(model2['classifier.3.weight'])
        # model._layers_list[41].bias = torch.nn.parameter.Parameter(model2['classifier.3.bias'])

        model._layers_list[39].weight = torch.nn.parameter.Parameter(model2['classifier.0.weight'])
        model._layers_list[39].bias = torch.nn.parameter.Parameter(model2['classifier.0.bias'])
        model._layers_list[42].weight = torch.nn.parameter.Parameter(model2['classifier.3.weight'])
        model._layers_list[42].bias = torch.nn.parameter.Parameter(model2['classifier.3.bias'])

print(model._layers_list)

print('Num training: %d\tNum testing: %d\n' % (train_data.__len__(), test_data.__len__()))

train_labels = train_data._getTensorLabels().numpy()
#valid_labels = valid_data._getTensorLabels().numpy()
test_labels = test_data._getTensorLabels().numpy()

print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
#print('Validation labels: ', [(lb, np.sum(valid_labels == lb)) for lb in set(valid_labels)], '\n')
print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')

trainer = Trainer(model, train_data, 0, test_data, num_epochs=num_epoch, batch_size=32)
trainer.trainModel()
trainer.getTrainACU()
trainer.getTestACU()
model.saveModel(model._model_rootpath + model_name)

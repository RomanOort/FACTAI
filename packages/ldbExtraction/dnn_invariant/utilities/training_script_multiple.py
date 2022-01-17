

from dnn_invariant.models.models4invariant import *
from dnn_invariant.utilities.trainer import *
from dnn_invariant.utilities.datasets import *
import numpy as np

import moxing as mox
mox.file.shift('os', 'mox')

np.set_printoptions(threshold=np.inf, precision=20)
np.random.seed(0)
torch.set_printoptions(precision=6)
torch.manual_seed(0)

for i in range(5):
    model = VGG19Assira(num_classes_=2)
    model2 = torch.load('mdls/VGG19Kaggle.mdl', map_location='cuda:0')
    #model2 = torch.load('s3://vbadai-share/Peter/dnn_interpretation/code/dnn_invariant/mdls/VGG19Kaggle.mdl', map_location='cuda:0')

    with torch.no_grad():
        mean = torch.tensor([0.0])
        std = torch.tensor([0.0001])
        
        model._layers_list[0].weight = torch.nn.parameter.Parameter(model2['features.0.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[0].bias = torch.nn.parameter.Parameter(model2['features.0.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[2].weight = torch.nn.parameter.Parameter(model2['features.2.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[2].bias = torch.nn.parameter.Parameter(model2['features.2.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[5].weight = torch.nn.parameter.Parameter(model2['features.5.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[5].bias = torch.nn.parameter.Parameter(model2['features.5.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[7].weight = torch.nn.parameter.Parameter(model2['features.7.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[7].bias = torch.nn.parameter.Parameter(model2['features.7.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[10].weight = torch.nn.parameter.Parameter(model2['features.10.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[10].bias = torch.nn.parameter.Parameter(model2['features.10.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[12].weight = torch.nn.parameter.Parameter(model2['features.12.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[12].bias = torch.nn.parameter.Parameter(model2['features.12.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[14].weight = torch.nn.parameter.Parameter(model2['features.14.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[14].bias = torch.nn.parameter.Parameter(model2['features.14.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[16].weight = torch.nn.parameter.Parameter(model2['features.16.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[16].bias = torch.nn.parameter.Parameter(model2['features.16.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[19].weight = torch.nn.parameter.Parameter(model2['features.19.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[19].bias = torch.nn.parameter.Parameter(model2['features.19.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[21].weight = torch.nn.parameter.Parameter(model2['features.21.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[21].bias = torch.nn.parameter.Parameter(model2['features.21.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[23].weight = torch.nn.parameter.Parameter(model2['features.23.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[23].bias = torch.nn.parameter.Parameter(model2['features.23.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[25].weight = torch.nn.parameter.Parameter(model2['features.25.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[25].bias = torch.nn.parameter.Parameter(model2['features.25.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[28].weight = torch.nn.parameter.Parameter(model2['features.28.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[28].bias = torch.nn.parameter.Parameter(model2['features.28.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[30].weight = torch.nn.parameter.Parameter(model2['features.30.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[30].bias = torch.nn.parameter.Parameter(model2['features.30.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[32].weight = torch.nn.parameter.Parameter(model2['features.32.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[32].bias = torch.nn.parameter.Parameter(model2['features.32.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[34].weight = torch.nn.parameter.Parameter(model2['features.34.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[34].bias = torch.nn.parameter.Parameter(model2['features.34.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[38].weight = torch.nn.parameter.Parameter(model2['classifier.0.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[38].bias = torch.nn.parameter.Parameter(model2['classifier.0.bias'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[41].weight = torch.nn.parameter.Parameter(model2['classifier.3.weight'] + torch.normal(mean=mean, std=std).cuda())
        model._layers_list[41].bias = torch.nn.parameter.Parameter(model2['classifier.3.bias'] + torch.normal(mean=mean, std=std).cuda())


    print('Num training: %d\tNum testing: %d\n' % (train_data.__len__(), test_data.__len__()))

    train_labels = train_data._getTensorLabels().numpy()
    test_labels  = test_data._getTensorLabels().numpy()

    print('Trainining labels: ', [(lb, np.sum(train_labels == lb)) for lb in set(train_labels)], '\n')
    print('Testing labels: ', [(lb, np.sum(test_labels == lb)) for lb in set(test_labels)], '\n')


    trainer = Trainer(model, train_data, 0, test_data, num_epochs=20, learning_rate=0.001, batch_size=32)
    trainer.trainModel()
    trainer.getTrainACU()
    trainer.getTestACU()

    model.saveModel(model._model_rootpath + model.__class__.__name__ + str(i) + '.mdl')

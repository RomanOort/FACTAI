

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dnn_invariant.utilities.environ import *

class Trainer():
    def __init__(self, model, train_data, valid_data, test_data,
                 num_epochs=100, learning_rate=0.001, batch_size=100):

        # hyperparamters for training
        self.num_epochs     = num_epochs
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size

        self.criterion      = nn.CrossEntropyLoss()
        #self.optimizer      = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # other variables
        self.model      = model
        self.model = self.model.cuda()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)


        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data  = test_data

    def trainModel(self):
        self.model.train()

        print('Start training ...')
        print('num_epochs: {}, lr: {}, batch_size: {}'.format(self.num_epochs, self.learning_rate, self.batch_size))
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        total_steps = len(train_loader)

        for epoch in range(self.num_epochs):
            for step, (instances, labels) in enumerate(train_loader):
                instances = instances.cuda()
                labels = labels.cuda()

                # Forward pass
                outputs = self.model(instances)


                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (step + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f}'
                          .format(epoch + 1, self.num_epochs, step + 1, total_steps, loss.item()))

            #self.getTrainACU()
            #self.getTestACU()

    def getTrainACU(self):
        train_acu = self.evalAccuracy(self.train_data)
        print('Training Accuracy of the model: %.8f %%' % (100.0*train_acu))
        return train_acu

    def getValidACU(self):
        valid_acu = self.evalAccuracy(self.valid_data)
        print('Validation Accuracy of the model: %.8f %%' % (100.0*valid_acu))
        return valid_acu

    def getTestACU(self):
        test_acu = self.evalAccuracy(self.test_data)
        print('Test Accuracy of the model: %.8f %%' % (100.0*test_acu))
        return test_acu

    def evalAccuracy(self, data):
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)

        self.model.cuda()
        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            total = correct = 0
            total_loss = 0
            for instances, labels in data_loader:
                instances = instances.cuda()
                labels = labels.cuda()
                outputs = self.model(instances)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                total_loss += self.criterion(outputs, labels).item()
                correct += (predicted == labels).sum().item()

            print('Total Loss: %.10f' % total_loss)

        return 1.0*correct/total

    @staticmethod
    def evalAccuracyOfModel(model, data):
        data_loader = DataLoader(data, batch_size=30, shuffle=False)

        gt_labels = data._getTensorLabels().cpu().numpy()

        pred_labels_list = []
        pred_probs_list  = []

        model.cuda()
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            total = correct = 0
            for adjs, instances, labels, num_nodes in data_loader:

                instances = instances.cuda()
                labels = labels.cuda()
                adjs = adjs.cuda()
                outputs, _ = model(instances, adjs, batch_num_nodes = num_nodes.tolist())

                _, predicted = torch.max(outputs.data, 1)

                probs = torch.nn.Softmax(dim=1)(outputs.data)
                probs, _ = torch.max(probs, 1)
                pred_probs_list.append(probs.cpu().numpy())

                total += labels.size(0)
                pred_labels_list.append(predicted.cpu().numpy())

        import numpy as np

        pred_probs  = np.concatenate(pred_probs_list)
        pred_labels = np.concatenate(pred_labels_list)

        correct = np.sum(pred_labels == gt_labels)

        return 1.0*correct/total, pred_labels, pred_probs

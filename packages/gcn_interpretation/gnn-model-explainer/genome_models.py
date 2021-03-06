import logging
import time
import itertools
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from scipy import sparse
# from models.utils import *
from genome_utils import *


class Model(nn.Module):

    def __init__(self, name=None, column_names=None, num_epochs=10, channels=16, num_layer=2, embedding=8, gating=0., attention_head=0,
                 omic_num=2, dropout=False, cuda=False, seed=0, adj=None, graph_name=None, aggregation=None, prepool_extralayers=0,
                 lr=0.001, patience=1000, agg_reduce=2, scheduler=True, metric=sklearn.metrics.accuracy_score,
                 optimizer=torch.optim.Adam, weight_decay=0.0001, batch_size=200, train_valid_split=0.8, 
                 evaluate_train=True, verbose=True,verbose_epoch=True, verbose_batch=False,full_data_cuda=False,score='All'):
        self.name = name
        self.column_names = column_names
        self.num_layer = num_layer
        self.channels_num = channels
        self.channels = [channels] * self.num_layer
        self.embedding = embedding
        self.gating = gating
        self.omic_num = omic_num
        self.dropout = dropout
        self.on_cuda = cuda
        self.num_epochs = num_epochs
        self.seed = seed
        self.adj = adj
        self.graph_name = graph_name
        self.prepool_extralayers = prepool_extralayers
        self.aggregation = aggregation
        self.lr = lr
        self.scheduler = scheduler
        self.agg_reduce = agg_reduce
        self.batch_size = batch_size
        self.start_patience = patience
        self.attention_head = attention_head
        self.train_valid_split = train_valid_split
        self.best_model = None
        self.metric = metric
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.verbose_epoch = verbose_epoch
        self.verbose_batch = verbose_batch
        self.evaluate_train = evaluate_train
        self.full_data_cuda = full_data_cuda
        self.score = str(score)
        if self.verbose:
            print("Early stopping metric is " + self.metric.__name__)
        self.att = False
        super(Model, self).__init__()
   
    def setup_model(self, X, y, adj=None):
        self.adj = adj
        self.X = X
        self.y = y
        self.setup_layers()
        self.eval()

    def fit(self, X, y, adj=None):
        print('Begin')
        self.adj = adj
        self.X = X
        self.y = y
        self.setup_layers()
        x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, stratify=y, train_size=self.train_valid_split, test_size=1-self.train_valid_split, random_state=self.seed)
        
        y_true = y_train # Save copy on CPU for evaluation
        print("X train:",x_train.shape)
#         print("X train:",x_train)
        x_train = torch.FloatTensor(np.expand_dims(x_train, axis=2))
        x_valid = torch.FloatTensor(np.expand_dims(x_valid, axis=2))
        # x_train = torch.FloatTensor(x_train)
        #L x_valid = torch.FloatTensor(x_valid)
        y_train = torch.FloatTensor(y_train)
        y_valid = torch.FloatTensor(y_valid)
        if self.on_cuda and self.full_data_cuda:
            try:
                x_train = x_train.cuda()
                x_valid = x_valid.cuda()
                y_train = y_train.cuda()
                y_valid = y_valid.cuda()
            except:
                # Move data to GPU batch by batch
                self.full_data_cuda = False
                
        
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

        max_valid = 0
        patience = self.start_patience
        #self.best_model = self.state_dict().copy()
        all_time = time.time()
        epoch = 10 # when num_epoch is set to 0 for testing
        epoch_loss_all_train = []
        metric_train = []
        metric_valid = []

        for epoch in range(0, self.num_epochs):
            start = time.time()
            epoch_loss = 0
            batch_times = 0
            for i in range(0, x_train.shape[0], self.batch_size):
                self.train()
                
                inputs, labels = x_train[i:i + self.batch_size], y_train[i:i + self.batch_size]
#                 print(inputs.shape, inputs[0,0])
                inputs = Variable(inputs, requires_grad=False).float()
                if self.on_cuda and not self.full_data_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                #self = torch.nn.DataParallel(self) 
                y_pred = self(inputs)
                targets = Variable(labels, requires_grad=False).long()
                loss = criterion(y_pred, targets)
                
                batch_times = batch_times + 1
                epoch_loss = epoch_loss + loss
                
                if self.verbose_batch:
                    print("  batch ({}/{})".format(i, x_train.shape[0]) + ", train loss:" + "{0:.4f}".format(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            epoch_loss_ave = epoch_loss/batch_times
            epoch_loss_all_train.append(epoch_loss_ave.cpu().detach().numpy())
            
            self.eval()
            start = time.time()

            auc = {'train': 0., 'valid': 0.}
            if self.evaluate_train:
                res = []
                for i in range(0, x_train.shape[0], self.batch_size):
                    inputs = Variable(x_train[i:i + self.batch_size]).float()
                    if self.on_cuda and not self.full_data_cuda:
                        inputs = inputs.cuda()
                    res.append(self(inputs).data.cpu().numpy())
                y_hat = np.concatenate(res)
                auc['train'] = self.metric(y_true, np.argmax(y_hat, axis=1))
                metric_train .append(auc['train'])

            res = []
            epoch_loss = 0
            batch_times = 0
            for i in range(0, x_valid.shape[0], self.batch_size):
                inputs = Variable(x_valid[i:i + self.batch_size]).float()
                if self.on_cuda and not self.full_data_cuda:
                    inputs = inputs.cuda()
                res.append(self(inputs).data.cpu().numpy())
            
            y_hat = np.concatenate(res)
            auc['valid'] = self.metric(y_valid.cpu(), np.argmax(y_hat, axis=1))
            metric_valid .append(auc['valid'])
            patience = patience - 1
            if patience == 0:
                break
            if (auc['valid'] > max_valid ) and epoch > 5:
                #print('max_valid: {},auc: {} '.format(max_valid,auc['valid']))
                max_valid = auc['valid']
                patience = self.start_patience
                #self.best_model = self.state_dict().copy()
            if self.verbose_epoch and epoch%20==0:
                print("epoch: {}, time: {:.2f}, train_loss: {:.4f}, train_metric: {:.4f}, valid_metric: {:.4f}".format(\
               str(epoch), (time.time() - start)*10, epoch_loss_ave, auc['train'], auc['valid'] ))
            if self.scheduler:
                scheduler.step()
                
        if self.verbose:
            print("total train time:" + "{0:.2f}".format((time.time() - all_time)*10) + " for epochs: " + str(epoch))
            
        epoch_loss_metric_all = {}
        epoch_loss_metric_all['Train_loss'] = epoch_loss_all_train
        epoch_loss_metric_all['metric_train'] = metric_train
        epoch_loss_metric_all['metric_valid'] = metric_valid
        #epoch_loss_all['Valid'] = epoch_loss_all_val
        epoch_loss_metric_all = pd.DataFrame(epoch_loss_metric_all)
        epoch_loss_metric_files = "./outputs/score{}_epoch{}_bs{}_numlayer{}_cha{}_emb{}_loss_train_validx.csv".format(self.score,self.num_epochs,\
                  self.batch_size,self.num_layer, self.channels_num, self.embedding )
        epoch_loss_metric_all.to_csv(epoch_loss_metric_files)
        print('Write loss of train and validation into {}'.format(epoch_loss_metric_files))
        
        #self.load_state_dict(self.best_model)
        #self.best_model = None

    def predict(self, inputs, probs=True, adj=None):
        """
        Run the trained model on the inputs

        Args:
        inputs: Input to the model
        probs (bool): Get probability estimates
        """
        # inputs = inputs.unsqueeze(2)
        # inputs = torch.FloatTensor(np.expand_dims(inputs, axis=2))
        # if self.on_cuda:
        #     inputs = inputs.cuda()

        if adj is None:
            out = self.forward(inputs)
        else:
            out = self.forward_adj(inputs, adj)
        if probs:
            # m = nn.Softmax(dim=1)
            # out = m(out)
            out = F.softmax(out, dim=1)
        return out
        # return out.cpu().detach()

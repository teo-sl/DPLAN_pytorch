import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(42)
torch.manual_seed(42)


# deep sad hyperparameters
eta = 0.2
eps = 1e-6
train_labels = None



### Neural Network definition ### 


class DQN(nn.Module):
    """
    Deep Q Network
    """
    def __init__(self, n_observations,hidden_size, n_actions, device='cpu'):
        super(DQN, self).__init__()
        self.device = device
        bias = True
        self.latent = nn.Sequential(
            nn.Linear(n_observations,hidden_size,bias=bias),
        )
        self.output_layer = nn.Linear(hidden_size,n_actions,bias=bias)

    def forward(self, x):
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,device=self.device)
        x = F.relu(self.latent(x))
        return self.output_layer(x)
    def get_latent(self,x):
        """
        Get the latent representation of the input using the latent layer
        """
        self.eval()
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,device=self.device)
        
        with torch.no_grad():
            latent_embs = F.relu(self.latent(x))
        self.train()
        return latent_embs
    def predict_label(self,x):
        self.eval()
        """
        Predict the label of the input as the argmax of the output layer
        """
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,device=self.device)

        with torch.no_grad():
            ret = torch.argmax(self.forward(x),axis = 1)
            self.train()
            return ret
        
    def _initialize_weights(self,):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.constant_(m.bias, 0.0)

    def forward_latent(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,device=self.device)
        latent = F.relu(self.latent(x))
        out = self.output_layer(latent)
        return out,latent
    
    def get_latent_grad(self,x):
        if not isinstance(x,torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32,device=self.device)
        latent_embs = F.relu(self.latent(x))
        return latent_embs
    

### Utility functions ###

def DQN_iforest(x, model):
    # iforest function on the penuli-layer space of DQN
    # get the output of penulti-layer
    latent_x=model.get_latent(x)
    latent_x=latent_x.cpu().detach().numpy()
    # calculate anomaly scores in the latent space
    iforest=IsolationForest().fit(latent_x)
    scores = -iforest.decision_function(latent_x)
    # normalize the scores
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    #norm_scores = np.array([-1*s+0.5 for s in scores])
    return norm_scores


def distance_from_c(x,net,c):
    with torch.no_grad():
        latent_x=net.get_latent(x)
        #dist = torch.abs(**2)
        dist = torch.sum((latent_x - c)**2, dim=1)
        dist = (dist - torch.min(dist)) / (torch.max(dist) - torch.min(dist))
        dist = dist.cpu().detach().numpy()
    return dist


def loss_sad(x,labels,c,eta,eps):
    labels = labels*(-1)
    dist = torch.sum((x - c) ** 2, dim=1)
    losses = torch.where(labels == 0, dist, eta * ((dist + eps) ** labels.float()))
    loss = torch.mean(losses)
    return loss

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

### DRAN class ###

class DRAN:
    def __init__(self,train_set,test_set,config,c,device='cpu'):
        """
        c : the hypersphere's center according to Deep-SAD
        """

        self.device = device

        self.x = train_set[:,:-1]
        self.x_tensor = torch.tensor(self.x,dtype=torch.float32,device=self.device)
        self.y = train_set[:,-1]
        self.y[self.y==2] = -1
        self.test_set = test_set
        
        self.relabeling_accuracy = []
        self.changed = []
        self.iter = -1

        self.model = DQN(self.x.shape[1],20,1).to(self.device)
        self.model._initialize_weights()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr']) #-3 for thyroid
        self.criterion = torch.nn.SmoothL1Loss()

        self.batch_size = config['batch_size']
        self.validation_step = config['validation_step']
        self.update_step = config['update_step']

        self.validation_history = []
        self.steps_per_epoch = None


        self.index_a = np.argwhere(self.y==1).reshape(-1)
        self.index_n = np.argwhere(self.y==-1).reshape(-1)
        self.index_u = np.argwhere(self.y==0).reshape(-1)
        
        
        self.c = c

        self.sad_lr = config['sad_lr']
        self.scores = None
        


    def train_sad(self,num_epochs=30,logs=True):
        sad_optimizer = torch.optim.Adam(self.model.latent.parameters(), lr=self.sad_lr, weight_decay=1e-6)
        y = torch.tensor(self.y, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(self.x_tensor,y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            total_loss = 0
            for x,y in data_loader:
                sad_optimizer.zero_grad()
                latent = self.model.get_latent_grad(x)
                loss = loss_sad(latent,y,self.c,eta,eps)
                loss.backward()
                sad_optimizer.step()
                total_loss += loss.cpu().detach().item()
            loss_epoch = total_loss/len(data_loader)
            if logs and (epoch+1) % 5 == 0:
                print(f'epoch: {epoch} loss: {loss_epoch}')
            total_loss = 0
            

    def train(self,n_epochs=100,n_epochs_sad=10):
        # optional deep sad pretraining
        self.train_sad(n_epochs_sad)
    
        self.scores = distance_from_c(self.x_tensor,self.model,self.c)

        self.validation_history = []
        n_steps = 0
        num_normals = int(self.batch_size*0.4)
        num_anomalies = int(self.batch_size*0.4)
        num_unlabeled = int(self.batch_size*0.2) # or 0.1?

        for epoch in range(n_epochs):
            for _ in range(self.x.shape[0]//self.batch_size):
                idx_normal = np.random.choice(self.index_n,num_normals)
                idx_anomaly = np.random.choice(self.index_a,num_anomalies)
                idx_unlabeled = np.random.choice(self.index_u,num_unlabeled) #self.get_lowest_scores(num_unlabeled) 
                x_normal = self.x[idx_normal]
                x_anomaly = self.x[idx_anomaly]
                x_unlabeled = self.x[idx_unlabeled]
                x_batch = np.concatenate((x_normal,x_anomaly,x_unlabeled),axis=0)
    
                y = np.concatenate((
                    self.y[idx_normal]+self.scores[idx_normal],
                    self.y[idx_anomaly]+self.scores[idx_anomaly],
                    np.full(num_unlabeled,-1)+self.scores[idx_unlabeled],
                )).reshape(-1,1)


                x_batch = torch.tensor(x_batch,dtype=torch.float32,device=self.device)
                y = torch.tensor(y,dtype=torch.float32,device=self.device)
                
                # forward pass
                y_pred = self.model(x_batch)
                # compute loss
                loss = self.criterion(y_pred,y)
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                n_steps += 1
                if (n_steps) % self.validation_step == 0:
                    self.validation_history.append(self.test())
            if epoch==0:
                self.steps_per_epoch = n_steps                
            if (epoch+1) % 1 == 0:
                self.update_labels()
        self.plot_results()
    def plot_results(self):
        x_axis = np.arange(0,len(self.validation_history))*self.validation_step/self.steps_per_epoch
        plt.plot(x_axis,self.validation_history)
        plt.show()


    def update_labels(self):   
        self.y[self.changed] = 0
        self.changed = []
        self.iter+=1

        self.index_a = np.argwhere(self.y==1).reshape(-1)
        self.index_n = np.argwhere(self.y==-1).reshape(-1)
        self.index_u = np.argwhere(self.y==0).reshape(-1)

        scores = self.model(self.x_tensor).cpu().detach().numpy().reshape(-1)
        arg_scores =  np.argsort(scores)
        arg_scores = arg_scores[np.isin(arg_scores,self.index_u)]
        P = 100
        k = int(P+self.iter*P) # 100 for unsbs
        top_k = arg_scores[-k:]
        # add only if the score of the top k is distant enough from 0
        self.relabeling_accuracy = []
        for i in top_k:
            if scores[i] >= 0.8:
                #if train_labels[i]!=3:
                #    self.relabeling_accuracy.append(1)
                #else: self.relabeling_accuracy.append(0)

                self.y[i]= 1
                self.changed.append(i)

        
        #bottom_k = arg_scores[:k]
       
        #self.y[bottom_k] = -1

        # update indeces
        self.index_a = np.argwhere(self.y==1).reshape(-1)
        self.index_n = np.argwhere(self.y==-1).reshape(-1)
        self.index_u = np.argwhere(self.y==0).reshape(-1)
        
        
    def test(self):
        dataset = self.test_set
        self.model.eval()
        with torch.no_grad():
            test_X, test_y=dataset[:,:-1], dataset[:,-1]
            pred_y=self.model(test_X).cpu().detach().numpy()

        roc = roc_auc_score(test_y, pred_y)
        pr = average_precision_score(test_y, pred_y)

        self.model.train()
        return pr,roc
        
import numpy as np
import math

import torch 
import torch.nn as nn

    
class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0.
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                       nn.ReLU(),
                                       nn.Linear(int(hidden_size), y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 0, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)




class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(MINE, self).__init__()
        if hidden_size is None:
            self.T_func = nn.Linear(x_dim+y_dim, 1)
        else:
            self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, int(hidden_size)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_size), 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(NWJ, self).__init__()
        if hidden_size is None:
            self.F_func = nn.Linear(x_dim + y_dim, 1)
        else:
            self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, int(hidden_size)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_size), 1))
                                    
    def forward(self, x_samples, y_samples): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


    
class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=None):
        super(InfoNCE, self).__init__()

        if hidden_size is None:
            self.F_func = nn.Linear(x_dim + y_dim, 1)
        else:
            self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, int(hidden_size)),
                                    nn.ReLU(),
                                    nn.Linear(int(hidden_size), 1))
                                    #nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


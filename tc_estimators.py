import numpy as np
import math

import torch 
import torch.nn as nn

from mi_estimators import *

class TotalCorEstimator(nn.Module):
    def __init__(self, num_var, dim, hidden_size, mi_est_name = "CLUB"):  
        '''
        Calculate Total Correlation Estimation for variable X1, X2,..., Xn, each Xi dimension = dim, n = num_var
        '''
        super(TotalCorEstimator, self).__init__()
        self.mi_estimators = nn.ModuleList([eval(mi_est_name)(dim * (i+1), dim, hidden_size * np.sqrt(i+1)) for i in range(num_var-1)])
    
    def forward(self, samples): # samples is a tensor with shape [batch, num_var, dim]
        batch_size, num_var, dim = samples.size()
        outputs = []
        for i in range(1, num_var):
            outputs.append(self.mi_estimators[i-1](samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        batch_size, num_var, dim = smaples.size()
        outputs = []
        for i in range(1, num_var):
            outputs.append(self.mi_estimators[i-1].learning_loss(samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).mean()
        



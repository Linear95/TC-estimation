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
        self.mi_estimators = nn.ModuleList([eval(mi_est_name)(dim * (i+1), dim, int(hidden_size * np.sqrt(i+1))) for i in range(num_var-1)])
    
    def forward(self, samples): # samples is a tensor with shape [batch, num_var, dim]
        batch_size, num_var, dim = samples.size()
        outputs = []
        for i in range(1, num_var):
            outputs.append(self.mi_estimators[i-1](samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        batch_size, num_var, dim = samples.size()
        outputs = []
        for i in range(1, num_var):
            outputs.append(self.mi_estimators[i-1].learning_loss(samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).mean()
        


class TreeTC4Estimator(nn.Module):
    def __init__(self, num_var, dim, hidden_size, mi_est_name = "CLUB"):
        super(TreeTC4Estimator, self).__init__()
        estimator_list = [eval(mi_est_name)(dim, dim, hidden_size) for i in range(2)]
        estimator_list.append(eval(mi_est_name)(dim*2, dim*2, int(hidden_size * np.sqrt(2))))
        self.mi_estimators = nn.ModuleList(estimator_list)

    def forward(self, samples):
        mi_1 = self.mi_estimators[2](samples[:,2:].flatten(start_dim=1), samples[:,:2].flatten(start_dim=1))
        mi_2 = self.mi_estimators[0](samples[:,0], samples[:,1])
        mi_3 = self.mi_estimators[1](samples[:,2], samples[:,3])
        return mi_1 + mi_2 + mi_3

    def learning_loss(self, samples):
        loss_1 = self.mi_estimators[2](samples[:,2:].flatten(start_dim=1), samples[:, :2].flatten(start_dim=1))
        loss_2 = self.mi_estimators[0](samples[:,0], samples[:,1])
        loss_3 = self.mi_estimators[1](samples[:,2], samples[:,3])
        return (loss_1+loss_2+loss_3)/3.
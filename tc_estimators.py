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


class Tree_TC_Estimator(nn.Module):
    def __init__(self, num_var, dim, hidden_size, mi_est_name = "CLUB"):
        super(Tree_TC_Estimator, self).__init__()
        self.num_var = num_var
        self.mi_est_name = mi_est_name
        self.hidden_size = hidden_size
        self.est_index = []
        self.root = self._build_tree(0, num_var-1)
        print(self.est_index)
        estimator_list = [eval(mi_est_name)((mid-l+1)*dim, (r-mid)*dim, int(hidden_size*np.sqrt(r-l))) for (l, mid, r) in self.est_index]
        #print(estimator_list)
        # while 
        #estimator_list.append(eval(mi_est_name)(dim*2, dim*2, int(hidden_size * np.sqrt(2))))
        self.mi_estimators = nn.ModuleList(estimator_list)

    def _build_tree(self, l, r):
        if r-l <= 0:
            return
        else:
            mid = (r+l)//2
            self.est_index.append((l, mid, r))
            self._build_tree(l, mid)
            self._build_tree(mid+1, r)
            return 

    def forward(self, samples):
        output_list = []
        for i in range(len(self.est_index)):
            l, mid, r = self.est_index[i]
            output_list.append(self.mi_estimators[i](
                samples[:,l:mid+1].flatten(start_dim=1), 
                samples[:,mid+1:r+1].flatten(start_dim=1)))
        #print(mi_output_list)
        return torch.stack(output_list).sum()

    def learning_loss(self, samples):
        output_list = []
        for i in range(len(self.est_index)):
            l, mid, r = self.est_index[i]
            output_list.append(self.mi_estimators[i].learning_loss(
                samples[:,l:mid+1].flatten(start_dim=1), 
                samples[:,mid+1:r+1].flatten(start_dim=1)))
        return torch.stack(output_list).sum()


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
        loss_1 = self.mi_estimators[2].learning_loss(samples[:,2:].flatten(start_dim=1), samples[:, :2].flatten(start_dim=1))
        loss_2 = self.mi_estimators[0].learning_loss(samples[:,0], samples[:,1])
        loss_3 = self.mi_estimators[1].learning_loss(samples[:,2], samples[:,3])
        return (loss_1+loss_2+loss_3)/3.
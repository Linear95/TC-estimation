import numpy as np
import math

import torch 
import torch.nn as nn

from mi_estimators import CLUB, CLUBSample, MINE, NWJ, InfoNCE, L1OutUB, VarUB

MI_CLASS={
    'CLUB': CLUB,
    'CLUBSample': CLUBSample,
    'MINE': MINE,
    'NWJ': NWJ,
    'InfoNCE': InfoNCE,
    'L1OutUB': L1OutUB,
    'VarUB': VarUB
    }

class TCLineEstimator(nn.Module):
    def __init__(self, num_var, dim, hidden_size, mi_est_name = 'CLUB'):  
        '''
        Calculate Total Correlation Estimation for variable X1, X2,..., Xn, each Xi dimension = dim, n = num_var.
        '''
        super().__init__()
        self.num_var = num_var
        self.mi_est_name = mi_est_name
        self.mi_class = MI_CLASS[mi_est_name]
        
        self.mi_estimators = nn.ModuleList(
            [self.mi_class(
                x_dim=dim * (i+1),
                y_dim=dim,
                hidden_size=int(hidden_size * np.sqrt(i+1))
                )
             for i in range(num_var-1)
            ]
        )
    
    def forward(self, samples): # samples is a tensor with shape [batch, num_var, dim]
        '''
        forward the estimated total correlation value with the given samples.
        '''
        outputs = []
        for i in range(1, self.num_var):
            outputs.append(self.mi_estimators[i-1](samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        '''
        return the learning loss to train the parameters of mi estimators.
        '''
        outputs = []
        for i in range(1, self.num_var):
            outputs.append(self.mi_estimators[i-1].learning_loss(samples[:,:i].flatten(start_dim = 1), samples[:,i]))
        return torch.stack(outputs).mean()


class TCTreeEstimator(nn.Module):
    def __init__(self, num_var, dim, hidden_size, mi_est_name = 'CLUB'):
        super().__init__()
        self.num_var = num_var
        self.mi_est_name = mi_est_name
        self.mi_class = MI_CLASS[mi_est_class]
        
        self.est_index = []
        self.root = self._build_tree(0, num_var-1)
        
        estimator_list = [mi_est_class((mid-l+1)*dim, (r-mid)*dim, int(hidden_size*np.sqrt(r-l))) for (l, mid, r) in self.est_index]

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
        for i, (l, mid, r) in enumerate(self.est_index):
            output_list.append(
                self.mi_estimators[i](
                    samples[:,l:mid+1].flatten(start_dim=1), 
                    samples[:,mid+1:r+1].flatten(start_dim=1)
                )
            )
            
        #print(mi_output_list)
        return torch.stack(output_list).sum()

    def learning_loss(self, samples):
        output_list = []
        for i, (l, mid, r) in enumerate(self.est_index):
            output_list.append(
                self.mi_estimators[i].learning_loss(
                    samples[:,l:mid+1].flatten(start_dim=1), 
                    samples[:,mid+1:r+1].flatten(start_dim=1)
                )
            )
            
        return torch.stack(output_list).sum()

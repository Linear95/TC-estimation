import numpy as np
import math

import torch 
import torch.nn as nn

from mi_estimators import  CLUBMean, MINE, NWJ, InfoNCE

MI_CLASS={
    'CLUB': CLUBMean,
    'MINE': MINE,
    'NWJ': NWJ,
    'InfoNCE': InfoNCE,
    }

class TCLineEstimator(nn.Module):
    def __init__(self, dims, hidden_size=None, mi_estimator='CLUB'):  
        '''
        Calculate Total Correlation Estimation for variable X1, X2,..., Xn, each Xi dimension = dim_i
        args:
            dims: a list of variable dimensions, [dim_1, dim_2,..., dim_n]
            hidden_size: hidden_size of vairiational MI estimators
            mi_estimator: the used MI estimator, selected from MI_CLASS
        '''
        super().__init__()
        self.dims = dims
        self.mi_est_type = MI_CLASS[mi_estimator]

        mi_estimator_list = [
            self.mi_est_type(
                x_dim=sum(dims[:i+1]),
                y_dim=dim,
                hidden_size=(None if hidden_size is None else hidden_size * np.sqrt(i+1))
            )
            for i, dim in enumerate(dims[:-1])
        ]
            
        self.mi_estimators = nn.ModuleList(mi_estimator_list)
        
    
    def forward(self, samples): # samples is a list of tensors with shape [Tensor([batch, dim_i])]
        '''
        forward the estimated total correlation value with the given samples.
        '''
        outputs = []
        concat_samples = [samples[0]]
        for i, dim in enumerate(self.dims[1:]):
            cat_sample = torch.cat(concat_samples, dim=1)
            outputs.append(self.mi_estimators[i](cat_sample, samples[i+1]))
            concat_samples.append(samples[i+1])
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        '''
        return the learning loss to train the parameters of mi estimators.
        '''
        outputs = []
        concat_samples = [samples[0]]
        for i, dim in enumerate(self.dims[1:]):
            cat_sample = torch.cat(concat_samples, dim=1)
            outputs.append(self.mi_estimators[i].learning_loss(cat_sample, samples[i+1]))
            concat_samples.append(samples[i+1])

        return torch.stack(outputs).mean()


class TCTreeEstimator(nn.Module):
    def __init__(self, dims, hidden_size=None, mi_estimator='CLUB'):
        super().__init__()
        self.dims = dims
        self.mi_est_type = MI_CLASS[mi_estimator]
        
        self.idx_scheme = []  # dims[left_idx: mid_idx] and dims[mid_idx: right_idx]
        self._build_idx_scheme(0, len(dims))
        
        mi_estimator_list = [
            self.mi_est_type(
                x_dim=sum(dims[l:m]),
                y_dim=sum(dims[m:r]),
                hidden_size=(None if hidden_size is None else hidden_size * np.sqrt(r-l-1))
            )
            for l, m, r in self.idx_scheme
        ]                                                        
        
        self.mi_estimators = nn.ModuleList(mi_estimator_list)

    def _build_idx_scheme(self, left_idx, right_idx):
        # split variables [left_idx : right_idx] to variables [left_idx: mid_idx] and [mid_idx, right_idx]
        if right_idx-left_idx > 1:
            mid_idx = (right_idx + left_idx) // 2
            self.idx_scheme.append((left_idx, mid_idx, right_idx))
            self._build_idx_scheme(left_idx, mid_idx)
            self._build_idx_scheme(mid_idx, right_idx)


    def forward(self, samples):
        outputs = []
        for i, (l, m, r) in enumerate(self.idx_scheme):
            outputs.append(
                self.mi_estimators[i](
                    x_samples=torch.cat(samples[l:m], dim=1),
                    y_samples=torch.cat(samples[m:r], dim=1)                    
                )
            )            
        return torch.stack(outputs).sum()

    def learning_loss(self, samples):
        outputs = []
        for i, (l, m, r) in enumerate(self.idx_scheme):
            outputs.append(
                self.mi_estimators[i].learning_loss(
                    x_samples=torch.cat(samples[l:m], dim=1),
                    y_samples=torch.cat(samples[m:r], dim=1)                    
                )
            )                        
        return torch.stack(outputs).sum()

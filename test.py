import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False, cubic = False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0,0]
    cov = [[1.0, rho],[rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if cubic:
        y = y ** 3

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        #x = torch.cat([x, torch.randn_like(x).cuda() * 0.3], dim=-1)
        y = torch.from_numpy(y).float().cuda()
    return x, y


def rho_to_mi(rho, dim):
    result = -dim / 2 * np.log(1 - rho **2)
    return result


def mi_to_rho(mi, dim):
    result = np.sqrt(1 - np.exp(-2 * mi / dim))
    return result


sample_dim = 20
batch_size = 64
hidden_size = 15
learning_rate = 0.005
training_steps = 4000

cubic = False 
#model_list = ["NWJ", "MINE", "InfoNCE","L1OutUB","CLUB","CLUBSample"]
model_list = ["CLUB"]

mi_list = [2.0, 4.0, 6.0, 8.0, 10.0]

total_steps = training_steps*len(mi_list)


from mi_estimators import *
from tc_estimators import *

mi_results = dict()
for i, model_name in enumerate(model_list):
    
    model = TotalCorEstimator(2, sample_dim, hidden_size, model_list[i])
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    mi_est_values = []

    start_time = time.time()
    for i, mi_value in enumerate(mi_list):
        rho = mi_to_rho(mi_value, sample_dim)

        for step in range(training_steps):
            batch_x, batch_y = sample_correlated_gaussian(rho, dim=sample_dim, batch_size = batch_size, to_cuda = True, cubic = cubic)

            model.eval()
            mi_est_values.append(model(batch_x, batch_y).item())
            
            model.train() 
            model_loss = model.learning_loss(batch_x, batch_y)
           
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            
            del batch_x, batch_y
            torch.cuda.empty_cache()

        print("finish training for %s with true MI value = %f" % (model.__class__.__name__, mi_value))
        # torch.save(model.state_dict(), "./model/%s_%d.pt" % (model.__class__.__name__, int(mi_value)))
        torch.cuda.empty_cache()
    end_time = time.time()
    time_cost = end_time - start_time
    print("model %s average time cost is %f s" % (model_name, time_cost/total_steps))
    mi_results[model_name] = mi_est_values




# bias_dict = dict()
# var_dict = dict()
# mse_dict = dict()
# for i, model_name in enumerate(model_list):
#     bias_list = []
#     var_list = []
#     mse_list = []
#     for j in range(len(mi_list)):
#         mi_est_values = mi_results[model_name][training_steps*(j+1)- 500:training_steps*(j+1)]
#         est_mean = np.mean(mi_est_values)
#         bias_list.append(np.abs(mi_list[j] - est_mean))
#         var_list.append(np.var(mi_est_values))
#         mse_list.append(bias_list[j]**2+ var_list[j])
#     bias_dict[model_name] = bias_list
#     var_dict[model_name] = var_list
#     mse_dict[model_name] = mse_list

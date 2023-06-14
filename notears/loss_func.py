import torch
import torch.nn as nn
import numpy as np


# fangfu: loss function 
def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def single_loss(output, target):
    n = target.shape[0]
    single_loss = 0.5 / n * ((output - target) ** 2)
    return single_loss

def reweighted_loss(output, target, reweight_list, beta):
    n = target.shape[0]
    re_matrix = torch.eye(n,n)
    for idx in reweight_list:
        re_matrix[idx,idx] = beta
    R = output-target
    loss = 0.5 / n * torch.sum(torch.matmul(re_matrix, R) ** 2)
    return loss

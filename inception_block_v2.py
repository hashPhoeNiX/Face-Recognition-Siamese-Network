import torch
from torch import nn
from torch.nn import functional as F

def inception_block_1a(X):
    '''
    Implementation of an inception block
    '''

    X_3x3 = nn.Conv2d(96, 128, kernel_size=(1, 1))
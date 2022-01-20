#!/usr/bin/env python3

import torch
from torch import nn 

import numpy as np
import math
from torch.distributions.normal import Normal


class DQNModel(nn.Module):

    def __init__(self, input_size, output_size, layers_sizes=[32,32], activation=nn.ReLU()):
        super(DQNModel, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.layers = []
        self.layers.append(nn.Flatten())
        first_layer = nn.Linear(in_features=input_size, out_features=layers_sizes[0], bias=False)
        self.layers.append(first_layer)
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(layers_sizes[-1],output_size))
        
        self.net = nn.Sequential(*self.layers)
        
        self.outputs = dict()
        
    def forward(self, input):
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.clone().detach().float()
            
        out = self.net(X)
        return out

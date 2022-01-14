#!/usr/bin/env python3

import torch
from torch import nn 

import numpy as np

class CriticModel(nn.Module):
    '''
    It is used to predict the distance given in input the state of the environment 
    The input is state, the output is the distance to the goal state
    '''
    def __init__(self, input_size, output_size=1, layers_sizes=[128,256,128,64,32], activation=nn.ReLU()):
        super(CriticModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        self.layers.append(nn.Linear(input_size, layers_sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers_sizes[-1],output_size))
        # self.layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.layers)

        # print("Critic Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.clone().detach().float()    
        #X = torch.tensor(input, dtype=torch.float, device=self.device)
        X = X.clone().detach()
        # print(f"X shape: {X.shape}")
        result = self.net(X)
        # for l in self.layers:
        #     X = l(X)
            
        return result

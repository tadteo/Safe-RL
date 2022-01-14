#!/usr/bin/env python3

import torch
from torch import nn 

import numpy as np

class StateModel(nn.Module):
    '''
    It is used to predict the next state given in input the state and the action taken 
    The input is state and the action, the output is the next state
    '''
    def __init__(self, state_size, action_size, action_layers_sizes=[8,16,32], state_layers_sizes=[32,128,64,32], layers_sizes=[32,64,128, 512,128, 64,32], activation=nn.Tanh(), has_continuous_action_space=True):
        super(StateModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.has_continuous_action_space = has_continuous_action_space
        
        self.layers = []
        if has_continuous_action_space:
            self.layers.append(nn.Linear(state_size+action_size, layers_sizes[0]))
        else:
            self.layers.append(nn.Linear(state_size+1, layers_sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers_sizes[-1],state_size))
        

        self.features_layers = nn.Sequential(*self.layers)

        # print(f"State Model structure:\n After concat layers: {self.features_layers}\n\n")

    def forward(self, state_input, action_input):
        if(type(state_input) == np.ndarray):
            state_input = torch.from_numpy(state_input).float().to(self.device)
        if(type(action_input) == np.ndarray):
            action_input = torch.from_numpy(action_input).float().to(self.device)
        
        X_state = state_input.clone().detach().float()
        
        if self.has_continuous_action_space:
            X_action = action_input.clone().detach().float()
        else:
            X_action= action_input.unsqueeze(-1)
            X_action = X_action.clone().detach().float()

        X = torch.cat((X_state,X_action), dim=-1)

        for l in self.layers:
            X = l(X)
        
        return X

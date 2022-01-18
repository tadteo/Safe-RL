#!/usr/bin/env python3

from collections import deque

import torch

from utils import *

def fill_deque_with_zeros(deque_to_fill, element_size):
        for i in range(deque_to_fill.maxlen):
                deque_to_fill.append(torch.zeros(element_size))

class Agent(object):
    def __init__(self,state_size, 
                 action_size,
                 state_memory_size = 1,
                 batch_size=32,
                 replay_memory_size=10000,
                 has_continuous_action_space=True,
                 path_for_trained_models=None,
                 id_nr=0):
        
        self.has_continuous_action_space = has_continuous_action_space

        #Device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        #Memory used as observation --> the observation can contatin multiple states
        self.state_memory_size = state_memory_size
        self.state_memory = deque(maxlen=state_memory_size)
        fill_deque_with_zeros(self.state_memory, state_size)
        
        self.previous_state_memory = deque(maxlen=state_memory_size)
        fill_deque_with_zeros(self.previous_state_memory, state_size)
        
        self.batch_size = batch_size
                
        #Replay memory buffer
        self.replay_memory_size = replay_memory_size
        self.replay_memory_buffer = ReplayMemory(capacity=self.replay_memory_size)
        
        #Create or load networks
        if path_for_trained_models == None:
            #Create networks
            self.create_networks_from_scratch()
        else:
            import json
            with open(path_for_trained_models+"/model_sizes.json", 'r') as f:
                data = json.load(f)
            #Create networks
            self.create_networks_from_weights(path_for_trained_models,data)

        
    def observe(self, state):
        self.previous_state_memory.append(self.state_memory[-1])
        state = torch.from_numpy(state).float().to(self.device)
        self.state_memory.append(state)
    
    def reset(self):
        self.state_memory = deque(maxlen=self.state_memory_size)
        fill_deque_with_zeros(self.state_memory, self.state_size)
        
        self.previous_state_memory = deque(maxlen=self.state_memory_size)
        fill_deque_with_zeros(self.previous_state_memory, self.state_size)

    def select_action(self, exploration_on=True):
        raise NotImplementedError
    
    def update_replay_memory(self, action, distance):
        predictions = []
        self.replay_memory_buffer.push(self.state_memory, 
                                       self.previous_state_memory,
                                       action, #action
                                       0, #distance_critic
                                       0, #distance_with_state_prediction
                                       distance, #optimal_distance is the distance to goal when you are enough far away from the obstacles
                                       predictions
                                       )
    
    #Neural Networks methods
    
    def create_networks_from_scratch(self):
        raise NotImplementedError
    
    def create_networks_from_weights(self, path_for_trained_models,data):
        raise NotImplementedError
    
    def save_models(self, experiment_name, path_for_trained_models, epoch):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    #utils
    
        
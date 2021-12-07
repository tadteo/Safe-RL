#!/usr/bin/env python3

from collections import namedtuple, deque
import random

def generate_exp_name(exp_name):
    """
    Generate a unique experiment name.
    """
    import datetime
    
    # Get the current time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Generate a unique experiment name
    exp_name = exp_name + '_' + now
    
    return exp_name

Prediction = namedtuple('Prediction', ("action", 
                                       "distance", 
                                       "state"))

Transition = namedtuple('Transition', ('state', 
                                       'previous_state', 
                                       'action', 
                                       'distance_critic', 
                                       'distance_with_state_prediction',  
                                       'optimal_distance', 
                                       'predictions'))

## adding a comment

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))  #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

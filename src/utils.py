#!/usr/bin/env python3

from collections import namedtuple, deque
import random

def calculate_distance(previous_state,state):
    #calculate the distance
    previous_goal,previous_hazards = previous_state[:16],previous_state[16:]
    goal,hazards = state[:16],state[16:]
    
    distance_to_goal = max(goal)
    distance_to_goal_previous = max(previous_goal)
    
    distance = (distance_to_goal - distance_to_goal_previous) + 0.1*distance_to_goal #TODO Add a positive factor to the closeness to the hazard
    
    # if max(goal) >= 0.8:
    #     distance_to_goal = 0
    # else:
    #     distance_to_goal = 1/max(max(goal),0.0001) #(1-max(observation[(-3*16):(-2*16)]))*
    #     for i in range(len(hazards)):
    #         if hazards[i] > 0.8:
    #             distance_to_goal += 1/(1-hazards[i])
    
    return distance

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
                                       'distance',
                                       'distance_critic', 
                                       'distance_with_state_prediction',  
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

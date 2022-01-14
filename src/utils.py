#!/usr/bin/env python3

from collections import namedtuple, deque
import random
import math

def calculate_distance(state,goal_start,goal_end):
    #calculate the distance
    # print(state)
    # goal,hazards = state[:16],state[16:]
    if goal_start == 0 and goal_end==0:
        goal = state
    else:
        goal = state[goal_start:goal_end]


    # print(f"Goal: {goal}")
    
    # print(max(goal))
    distance = max(max(goal),0.00001)
    # if(max(hazards)>0.8):
    #     distance -=max(max(hazards),0.00001) #min(10/(max(goal)),100)
    # print(f"Distance to goal: {distance_to_goal}")
    # distance = (1-(distance_to_goal/10)**0.5) #(distance_to_goal - distance_to_goal_previous) + 0.1*distance_to_goal #TODO Add a positive factor to the closeness to the hazard
    
    return distance

def calculate_distance_2(previous_state,state):
    #calculate the distance
    previous_goal,previous_hazards = previous_state[:16],previous_state[16:]
    goal,hazards = state[:16],state[16:]
    
    distance_to_goal = -math.log(max(goal)) #min(10/(max(goal)),100)
    distance_to_goal_previous = max(previous_goal)
    # print(f"Distance to goal: {distance_to_goal}")
    distance = (1-(distance_to_goal/10)**0.5) #(distance_to_goal - distance_to_goal_previous) + 0.1*distance_to_goal #TODO Add a positive factor to the closeness to the hazard
    
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


#Logging and Tensorboard
def write_losses_log(loss_actor, loss_critic, loss_state, total_steps, episode, writer, agent):
    writer.add_scalar("Loss/Loss: actor_off_policy", loss_actor, total_steps)
    writer.add_scalar("Loss/Loss: critic_off_policy", loss_critic, total_steps)
    writer.add_scalar("Loss/Loss: state_off_policy", loss_state, total_steps)

    # if(episode%1==0):
    for name,weight in agent.actor_model.named_parameters():
        writer.add_histogram(name,weight,episode)
        writer.add_histogram("actor/"+name+"/weight",weight,episode)
        writer.add_histogram("actor/"+name+"/grad",weight.grad,episode)
        
    for name,weight in agent.critic_model.named_parameters():
        writer.add_histogram(name,weight,episode)
        writer.add_histogram("critic/"+name+"/weight",weight,episode)
        writer.add_histogram("critic/"+name+"/grad",weight.grad,episode)
    
    for name,weight in agent.state_model.named_parameters():
        writer.add_histogram(name,weight,episode)
        writer.add_histogram("state/"+name+"/weight",weight,episode)
        writer.add_histogram("critic/"+name+"/grad",weight.grad,episode)


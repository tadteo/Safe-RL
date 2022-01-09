#!/usr/bin/env python3

from asyncore import write
import queue
from turtle import distance
import numpy as np
import torch
import yaml
import os
import logging, sys
    
import socket
from datetime import datetime
    
from safety_gym.envs.engine import Engine 

from utils import calculate_distance

actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard_random.yaml'))
config = yaml.load(config_file, Loader=yaml.FullLoader)

ENV_DICT = config.get('environment')
RENDER = config.get('render')
HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')
EPISODES = config.get('episodes')
STEPS_PER_EPOCH = config.get('steps_per_epoch')
UPDATE_FREQUENCY = config.get('update_frequency')

STEPS_IN_FUTURE = config.get('steps_in_future')

#Agent parameters
MIN_EPSILON = config.get('min_epsilon')

ON_POLICY = config.get('on_policy')
OFF_POLICY = config.get('off_policy')

INITIAL_VARIANCE = config.get('initial_variance')
FINAL_VARIANCE = config.get('final_variance')

DISCOUNT_FACTOR = config.get('discount_factor')
BATCH_SIZE = config.get('batch_size')

ACTOR_MODEL_WEIGHTS = config.get('actor_model_weights')
CRITIC_MODEL_WEIGHTS = config.get('critic_model_weights')
STATE_MODEL_WEIGHTS = config.get('state_model_weights')

STATE_DIVERGENCE_TRESHOLD = config.get('state_divergence_treshold')

TRAINED_MODEL_PATH = config.get('trained_model_path')


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def generate_future(agent, observation, steps_in_future):
    """This function generates the future actions and states 
    from the given observation for n = steps in future, steps.
    
    Args:
        agent: the agent
        observation: the current observation
        steps_in_future: the number of steps in the future
        
    Returns:
        future_actions: the future actions
        future_states: the future states
    """
    
    future_actions = []
    future_states = []
    current_state = observation
    
    future_actions.append(agent.select_action(current_state))
    future_states.append(agent.state_model(current_state, future_actions[-1]))
    
    for i in range(steps_in_future):
        future_actions.append(agent.select_action(future_states[-1]))
        future_states.append(agent.state_model(future_states[-1], future_actions[-1]))
    
    print("Future actions: ", future_actions)
    # print("Future states: ", future_states)
    
    tmp = future_actions 
    future_actions = queue.Queue()
    [future_actions.put(i) for i in tmp]
    
    tmp = future_states
    future_states = queue.Queue()
    [future_states.put(i) for i in tmp]
    
    return future_actions, future_states

def check_prediction(future_states: queue.Queue) -> bool:
    """This function checks if the future states are safe or not.
    
    Args:
        prediction: the prediction
        future_states: the future states
        steps_in_future: the number of steps in the future
        
    Returns:
        correct: if the prediction is correct
    """
    import copy
    
    is_safe = True
    states = copy.copy(future_states)
    while not future_states.empty():
        future_state = states.get()
        hazards_vector = future_state[16:]
        
        for i in hazards_vector:
            if i > 0.5:
                is_safe = False
                logging.info("Hazard detected")
                break    
    logging.info("Safe")
    return is_safe

def main():
    
    #Create the environment
    env = Engine(config=ENV_DICT)
    
    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    
    # action space dimension
    if HAS_CONTINUOUS_ACTION_SPACE:
        action_size = env.action_space.shape
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    #TODO: Load the shape of the model and use it to create the models of the agent.

    #Setting up the logger
    total_number_of_steps = 0
    episodes = 0 
    while True:        
        logging.info(f"Starting episode {episodes}\n\n")
        
        observation, episode_steps, episode_distance_traveled = env.reset(), 0, 0        
        while not env.done:
            if RENDER:
                env.render()
            
            #increase counters
            total_number_of_steps += 1
            episode_steps += 1            
            
            previous_observation = observation
            action = env.action_space.sample() #random action
            observation, reward, done, info = env.step(action) # Reward not used            
            
            # print(f"Observation: {observation}")
            goal_vector = observation[:16]
            hazards_vector = observation[16:]
            
            distance_to_goal = calculate_distance(previous_observation, observation)
            
            logging.debug(f"{episode_steps}. The distance to goal is: {distance_to_goal}")
        
        episodes += 1
        
if __name__ == '__main__':
    main()

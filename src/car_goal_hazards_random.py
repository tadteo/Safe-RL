#!/usr/bin/env python3

import yaml
import os
import logging, sys
    
from safety_gym.envs.engine import Engine 

from utils import calculate_distance

actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard_random.yaml'))
config = yaml.load(config_file, Loader=yaml.FullLoader)

ENV_DICT = config.get('environment')
RENDER = config.get('render')
HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

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

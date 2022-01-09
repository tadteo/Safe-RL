#!/usr/bin/env python3

from asyncore import write
import queue
import numpy as np
import torch
import yaml
import os
import logging, sys
    
import socket
from datetime import datetime
    
from safety_gym.envs.engine import Engine 

from acs_pytorch import ACSAgent
from sac_pytorch import SACAgent

from torch.utils.tensorboard import SummaryWriter

from src.utils import calculate_distance

actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard_test_policy.yaml'))
config = yaml.load(config_file, Loader=yaml.FullLoader)

ENV_DICT = config.get('environment')
RENDER = config.get('render')
HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')
EPISODES = config.get('episodes')
STEPS_PER_EPOCH = config.get('steps_per_epoch')


ACTOR_MODEL_WEIGHTS = config.get('actor_model_weights')
CRITIC_MODEL_WEIGHTS = config.get('critic_model_weights')
STATE_MODEL_WEIGHTS = config.get('state_model_weights')

TRAINED_MODEL_PATH = config.get('trained_model_path')


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def main():
    
    #Create the environment
    env = Engine(config=ENV_DICT)
    
    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    
    observation = env.reset()
    
    print(observation)
    
    # action space dimension
    if HAS_CONTINUOUS_ACTION_SPACE:
        action_size = env.action_space.shape
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    #TODO: Load the shape of the model and use it to create the models of the agent.
    
    
    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, 
                     action_size=action_size, 
                     path_for_trained_models= os.path.join(actual_path,TRAINED_MODEL_PATH),
                     actor_model_weights=ACTOR_MODEL_WEIGHTS,
                     critic_model_weights=CRITIC_MODEL_WEIGHTS,
                     state_model_weights=STATE_MODEL_WEIGHTS
                     )
    logging.info(f"Agent created")

    #Setting up the logger
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=os.path.join(actual_path,"../runs", "Inference"+current_time + '_' + socket.gethostname()))
    logging.info(f'Starting evaluation')
    total_number_of_steps = 0
    total_number_of_episodes = 0 
    for episode in range(EPISODES):        
        logging.info(f"Starting episode {episode}\n\n")
        
        observation, episode_steps, episode_distance_traveled = env.reset(), 0, 0
        
        while not env.done:
            if RENDER:
                env.render()
            
            #increase counters
            total_number_of_steps += 1
            episode_steps += 1
            
            # print("Action: ", actions.get())
            observation, reward, done, info = env.step(agent.select_action(observation,exploration_on=False)) # Reward not used
            
            distance_to_goal = calculate_distance(observation)
            logging.debug(f"{episode_steps}. The distance to goal is: {distance_to_goal}")
            
            
            # episode_distance_traveled += np.linalg.norm(observation-previous_observation)
            
            
            
            
            writer.add_scalar("Distances/Distance: distance_to_goal", distance_to_goal, total_number_of_steps)
            # writer.add_scalar("Distances/Distance: distance_to_goal_predicted", agent.critic_model(observation), total_number_of_steps)
            # #Reset the environment and save data
            # if distance_to_goal >= 99:
            #     done = True
            if done or (total_number_of_steps == STEPS_PER_EPOCH-1):
                
                writer.add_scalar("Performances/Episode steps", episode_steps, total_number_of_episodes)
                writer.add_scalar("Performances/Episode distance", episode_distance_traveled, total_number_of_episodes)
                total_number_of_episodes +=1
                observation, episode_distance_traveled, episode_steps = env.reset(), 0, 0
                
                logging.info(f"Starting episode {total_number_of_episodes}")
                
            
    
    writer.close()
        
if __name__ == '__main__':
    main()

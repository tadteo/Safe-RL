#!/usr/bin/env python3

from asyncore import write
import numpy as np
import torch
import yaml
import os
import logging, sys
    
import socket
from datetime import datetime
    
from safety_gym.envs.engine import Engine 

from src.agents.sac_pytorch import SACAgent

from torch.utils.tensorboard import SummaryWriter

from utils import *


actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard_sac.yaml'))
config = yaml.load(config_file, Loader=yaml.FullLoader)

EXPERIMENT_NAME = config['experiment_name']

ENV_DICT = config.get('environment')
RENDER = config.get('render')
HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')
EPOCHS = config.get('epochs')
STEPS_PER_EPOCH = config.get('steps_per_epoch')
UPDATE_FREQUENCY = config.get('update_frequency')

STEPS_IN_FUTURE = config.get('steps_in_future')

#Agent parameters
MIN_EPSILON = config.get('min_epsilon')

ON_POLICY = config.get('on_policy')
OFF_POLICY = config.get('off_policy')

DISCOUNT_FACTOR = config.get('discount_factor')
BATCH_SIZE = config.get('batch_size')

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main():
    
    experiment_name = generate_exp_name(EXPERIMENT_NAME)
    
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

    logging.debug(f'Creating agent')
    agent = SACAgent(state_size=state_size, action_size=action_size, batch_size=BATCH_SIZE, discount_factor = DISCOUNT_FACTOR)
    logging.info(f"Agent created")

    writer = SummaryWriter(log_dir=os.path.join(actual_path,"../runs", experiment_name + '_' + socket.gethostname()))

    logging.info(f'Starting training')
    total_steps = 0
    episode = 0 
    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch}\n\n")
        
        observation, episode_steps, episode_cumulative_distance = env.reset(), 0,0
        distance = calculate_distance(observation)
        for t in range(STEPS_PER_EPOCH):
            if RENDER:
                env.render()
            
            #increase counters
            total_steps += 1
            episode_steps += 1
                            
            action = agent.select_action(env, observation, exploration_on=True) #equivalent to line below
            previous_observation = observation
            observation, reward, done, info = env.step(action) # Reward not used
            
            previous_distance = distance
            distance = calculate_distance(observation)
            
            if episode_steps%100 == 0:
                logging.debug(f"{episode_steps}. The distance to goal is: {distance}")
                        
            # Memory buffer components
            #'state', 
            #'previous_state', 
            #'action',
            #'distance'
            #'distance_critic', 
            #'distance_with_state_prediction',  
            #'predictions'
            agent.memory_buffer.push(observation, #state
                                     previous_observation, #previous_state 
                                     action, #action
                                     0, #distance_critic
                                     0, #distance_with_state_prediction
                                     distance, #distance to goal
                                     0) #predictions)
            
            episode_cumulative_distance += distance
            #Reset the environment and save data
            if distance >= 99:
                done = True
            if done or (t == STEPS_PER_EPOCH-1) or distance >= 99:
                writer.add_scalar("Performances/Episode steps", episode_steps, episode)
                writer.add_scalar("Performances/Episode final distance to goal", previous_distance, episode) # Printing previous distance not to have random location if goal is reached
                writer.add_scalar("Performances/Episode cumulative distance", episode_cumulative_distance, episode)

                episode +=1
                observation, episode_cumulative_distance, episode_steps = env.reset(),0, 0                
                logging.info(f"Starting episode {episode}") 
            
            #Perform training steps
            if total_steps % UPDATE_FREQUENCY == 0:
                loss_actor, loss_critic, loss_state = agent.train()
                write_losses_log(loss_actor, loss_critic, loss_state, total_steps, episode, writer, agent)
                                    
        if epoch % 5 == 0 and epoch != 0:
            agent.save_models(experiment_name, epoch)
    writer.close()
        
if __name__ == '__main__':
    main()

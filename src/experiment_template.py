#!/usr/bin/env python3

import configparser
import yaml
import logging, sys
    
    
from safety_gym.envs.engine import Engine 

from acs_pytorch import EPOCHS, STEPS_PER_EPOCH, ACSAgent

from torch.utils.tensorboard import SummaryWriter


config_file =  open('../config/car_goal_hazard.yaml')
config = yaml.load(config_file, Loader=yaml.FullLoader)

ENV_DICT = config.get('environment')
render = config.get('render')
has_continuous_action_space = config.get('has_continuous_action_space')
EPOCHS = config.get('epochs')
STEPS_PER_EPOCH = config.get('steps_per_epoch')

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main():
    
    #Create the environment
    env = Engine(config=ENV_DICT)

    
    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    
    
    observation_dict = env.reset()
    
    print(observation_dict)
    
    # action space dimension
    if has_continuous_action_space:
        action_size = env.action_space.shape[0]
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, action_size=action_size)
    logging.info(f"Agent created")

    writer = SummaryWriter()

    logging.info(f'Starting training')
    steps = 0
    total_num_episodes = 0
    number_of_episodes = 0 

    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch}\n\n")
        
        observation = env.reset()
            
        
    for t in range(STEPS_PER_EPOCH):
        if render:
            env.render()
        
        env.step(env.action_space.sample())
    
    writer.close()
    
        
if __name__ == '__main__':
    main()

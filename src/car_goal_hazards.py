#!/usr/bin/env python3

import configparser
import yaml
import logging, sys
    
    
from safety_gym.envs.engine import Engine 

from acs_pytorch import EPOCHS, MIN_EPSILON, STEPS_PER_EPOCH, ACSAgent

from torch.utils.tensorboard import SummaryWriter


config_file =  open('../config/car_goal_hazard.yaml')
config = yaml.load(config_file, Loader=yaml.FullLoader)

ENV_DICT = config.get('environment')
RENDER = config.get('render')
HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')
EPOCHS = config.get('epochs')
STEPS_PER_EPOCH = config.get('steps_per_epoch')

#Agent parameters
MIN_EPSILON = config.get('min_epsilon')


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def main():
    
    #Create the environment
    env = Engine(config=ENV_DICT)

    
    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    
    
    observation_dict = env.reset()
    
    print(observation_dict)
    
    # action space dimension
    if HAS_CONTINUOUS_ACTION_SPACE:
        action_size = env.action_space.shape[0]
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, action_size=action_size)
    logging.info(f"Agent created")

    writer = SummaryWriter()

    logging.info(f'Starting training')
    total_number_of_steps = 0
    total_number_of_episodes = 0 
    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch}\n\n")
        
        observation, episode_steps, episode_distance_traveled = env.reset(), 0, 0
        
        for t in range(STEPS_PER_EPOCH):
            if RENDER:
                env.render()
            
            #increase counters
            total_number_of_steps += 1
            episode_steps += 1
            
            
            action = agent.select_action(env, observation)
            
            observation, reward, done, info = env.step(action) # Reward not used
            
            goal_vector = observation[:16]
            hazards_vector = observation[16:]
            
            #calculate the distance
            if max(goal_vector) >= 0.99:
                distance_to_goal = 0
                optimal_distance = 0
            else:
                distance_to_goal = 1/max(max(goal_vector),0.001) #(1-max(observation[(-3*16):(-2*16)]))*
                optimal_distance = 1/max(max(goal_vector),0.001)
                # if max(hazards_vector) > 0.8: #observation hazard
                #     distance_to_goal += 1/(1-max(hazards_vector))
            
            distance_to_goal = 1-(1/(1+distance_to_goal))
            logging.debug(f"The distance to goal is: {distance_to_goal}")
            
            agent.memory_buffer.push(observation, 
                                     0, 
                                     action, 
                                     0, 
                                     0, 
                                     optimal_distance, #optimal_distance is the distance to goal when you are enough far away from the obstacles
                                     [])
            
            
            
            writer.add_scalar("Distances/Distance: distance_to_goal", distance_to_goal, total_number_of_steps)
            
            #Reset the environment and save data
            if done or (t == STEPS_PER_EPOCH-1):
                
                writer.add_scalar("Performances/Episode steps", episode_steps, total_number_of_episodes)
                writer.add_scalar("Performances/Episode distance", episode_distance_traveled, total_number_of_episodes)
                total_number_of_episodes +=1
                observation, episode_distance_traveled, episode_steps = env.reset(), 0, 0
                
                logging.info(f"Starting episode {total_number_of_episodes}")
                agent.exploration_epsilon = max(MIN_EPSILON, agent.exploration_epsilon/total_number_of_episodes)
    writer.close()
    
        
if __name__ == '__main__':
    main()

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

actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard_inference.yaml'))
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
                     batch_size=BATCH_SIZE, 
                     initial_variance=INITIAL_VARIANCE, 
                     final_variance=FINAL_VARIANCE, 
                     discount_factor = DISCOUNT_FACTOR,
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
        
        actions, states = generate_future(agent,observation,STEPS_IN_FUTURE) 
        
        print("actions: ", list(actions.queue))
        print("states: ", list(states.queue))
        
        while not env.done:
            if RENDER:
                env.render()
            
            #increase counters
            total_number_of_steps += 1
            episode_steps += 1
            
            # print("Action: ", actions.get())
            observation, reward, done, info = env.step(actions.get()) # Reward not used
            
            if states.empty():
                raise Exception("States queue is empty")
            state_predicted = states.get()
            observation = torch.from_numpy(observation).float().to(agent.device)
            print("State predicted: ", state_predicted.size())
            print("State actual: ", observation.size())
            divergence = ((observation-state_predicted)**2).sum()/state_size
            print("Divergence: ", divergence)
            if  divergence < STATE_DIVERGENCE_TRESHOLD or states.qsize() <= 5:
                # logging.info(f"State divergence: {divergence}")
                # logging.info(f"States queue size: {states.qsize()}")
                
                logging.info(f"Recalculating future")
                
                actions, states = generate_future(agent,observation,STEPS_IN_FUTURE)
                # print("actions: ", list(actions.queue))
                # print("states: ", list(states.queue))
                # logging.info(f"Future recalculated")
                
                logging.info(f"Checking prediction")
                if check_prediction(states):
                    print("Predictions safe")
                else:
                    print("Predictions unsafe")
                    
                                   
                print("actions: ", list(actions.queue))
                # print("states: ", list(states.queue))
            # print(f"Observation: {observation}")
            goal_vector = observation[:16]
            hazards_vector = observation[16:]
            
            #calculate the distance
            if max(goal_vector) >= 0.8:
                distance_to_goal = 0
                optimal_distance = 0
            else:
                distance_to_goal = 1/max(max(goal_vector),0.0001) #(1-max(observation[(-3*16):(-2*16)]))*
                optimal_distance = 1/max(max(goal_vector),0.0001)
                if max(hazards_vector) >= 0.5: #observation hazard
                    distance_to_goal += 1/(1-max(hazards_vector))
            
            # distance_to_goal = 1-(1/(1+distance_to_goal))
            logging.debug(f"{episode_steps}. The distance to goal is: {distance_to_goal}")
            
            
            # episode_distance_traveled += np.linalg.norm(observation-previous_observation)
            
            
            
            
            # writer.add_scalar("Distances/Distance: distance_to_goal", distance_to_goal, total_number_of_steps)
            # writer.add_scalar("Distances/Distance: distance_to_goal_predicted", agent.critic_model(observation), total_number_of_steps)
            # #Reset the environment and save data
            # if distance_to_goal >= 99:
            #     done = True
            # if done or (t == STEPS_PER_EPOCH-1):
                
            #     writer.add_scalar("Performances/Episode steps", episode_steps, total_number_of_episodes)
            #     writer.add_scalar("Performances/Episode distance", episode_distance_traveled, total_number_of_episodes)
            #     total_number_of_episodes +=1
            #     observation, episode_distance_traveled, episode_steps = env.reset(), 0, 0
                
            #     logging.info(f"Starting episode {total_number_of_episodes}")
                
            
    
    writer.close()
        
if __name__ == '__main__':
    main()

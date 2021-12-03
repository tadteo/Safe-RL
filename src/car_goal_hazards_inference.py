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

from acs_pytorch import OFF_POLICY, ON_POLICY, ACSAgent, Prediction
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

    future_actions = queue.Queue(future_actions)
    future_states = queue.Queue(future_states)
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
    
    is_safe = True
    while not future_states.empty():
        future_state = future_states.get()
        hazards_vector = future_state[16:]
        
        for i in hazards_vector:
            if i > 0.5:
                is_safe = False
                break    
    
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
    agent = SACAgent(state_size=state_size, action_size=action_size, batch_size=BATCH_SIZE, initial_variance=INITIAL_VARIANCE, final_variance=FINAL_VARIANCE, discount_factor = DISCOUNT_FACTOR)
    logging.info(f"Agent created")
    
    
    
    #Loading the weights of the models
    agent.actor_model.load_state_dict(torch.load(os.path.join(actual_path, ACTOR_MODEL_WEIGHTS)))
    agent.critic_model.load_state_dict(torch.load(os.path.join(actual_path, CRITIC_MODEL_WEIGHTS)))
    agent.state_model.load_state_dict(torch.load(os.path.join(actual_path, STATE_MODEL_WEIGHTS)))
    
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
        
        print("actions: ", actions)
        print("states: ", states)
        
        while not env.done:
            if RENDER:
                env.render()
            
            #increase counters
            total_number_of_steps += 1
            episode_steps += 1
            
            observation, reward, done, info = env.step(actions.get()) # Reward not used
            
            state_predicted = states.get()
            if torch.nn.MSELoss(observation, state_predicted) < STATE_DIVERGENCE_TRESHOLD or states.qsize() <= 5:
                logging.info(f"State divergence: {torch.nn.MSELoss(observation, state_predicted)}")
                logging.info(f"States queue size: {states.qsize()}")
                
                logging.info(f"Recalculating future")
                
                actions, states = generate_future(agent,observation,STEPS_IN_FUTURE)
                
                if check_prediction(states):
                    print("Predictions safe")
                else:
                    print("Predictions unsafe")                                
            
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
            
            
            episode_distance_traveled += np.linalg.norm(observation-previous_observation)
            
            
            
            
            writer.add_scalar("Distances/Distance: distance_to_goal", distance_to_goal, total_number_of_steps)
            writer.add_scalar("Distances/Distance: distance_to_goal_predicted", agent.critic_model(observation), total_number_of_steps)
            #Reset the environment and save data
            if distance_to_goal >= 99:
                done = True
            if done or (t == STEPS_PER_EPOCH-1):
                
                writer.add_scalar("Performances/Episode steps", episode_steps, total_number_of_episodes)
                writer.add_scalar("Performances/Episode distance", episode_distance_traveled, total_number_of_episodes)
                total_number_of_episodes +=1
                observation, episode_distance_traveled, episode_steps = env.reset(), 0, 0
                
                logging.info(f"Starting episode {total_number_of_episodes}")
                
            
            if ON_POLICY:
                agent.train_on_policy()
            if OFF_POLICY:
                if total_number_of_steps % UPDATE_FREQUENCY == 0:
                    loss_actor, loss_critic, loss_state = agent.train_off_policy(writer=writer, total_number_of_steps=total_number_of_steps)
                    writer.add_scalar("Loss/Loss: actor_off_policy", loss_actor, total_number_of_steps)
                    writer.add_scalar("Loss/Loss: critic_off_policy", loss_critic, total_number_of_steps)
                    writer.add_scalar("Loss/Loss: state_off_policy", loss_state, total_number_of_steps)
                
                    # if(total_number_of_episodes%1==0):
                    for name,weight in agent.actor_model.named_parameters():
                        writer.add_histogram(name,weight,total_number_of_episodes)
                        writer.add_histogram("actor/"+name+"/weight",weight,total_number_of_episodes)
                        writer.add_histogram("actor/"+name+"/grad",weight.grad,total_number_of_episodes)
                        
                    for name,weight in agent.critic_model.named_parameters():
                        writer.add_histogram(name,weight,total_number_of_episodes)
                        writer.add_histogram("critic/"+name+"/weight",weight,total_number_of_episodes)
                        writer.add_histogram("critic/"+name+"/grad",weight.grad,total_number_of_episodes)
                    
                    for name,weight in agent.state_model.named_parameters():
                        writer.add_histogram(name,weight,total_number_of_episodes)
                        writer.add_histogram("state/"+name+"/weight",weight,total_number_of_episodes)
                        writer.add_histogram("critic/"+name+"/grad",weight.grad,total_number_of_episodes)
                    
                    # agent.exploration_epsilon = max(MIN_EPSILON, agent.exploration_epsilon/total_number_of_episodes)
            writer.close()
        
if __name__ == '__main__':
    main()

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

from acs_pytorch import OFF_POLICY, ON_POLICY, ACSAgent, Prediction

from torch.utils.tensorboard import SummaryWriter

actual_path = os.path.dirname(os.path.realpath(__file__))

config_file =  open(os.path.join(actual_path,'../config/car_goal_hazard.yaml'))
config = yaml.load(config_file, Loader=yaml.FullLoader)

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

INITIAL_VARIANCE = config.get('initial_variance')
FINAL_VARIANCE = config.get('final_variance')

DISCOUNT_FACTOR = config.get('discount_factor')
BATCH_SIZE = config.get('batch_size')

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

    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, action_size=action_size, batch_size=BATCH_SIZE, initial_variance=INITIAL_VARIANCE, final_variance=FINAL_VARIANCE, discount_factor = DISCOUNT_FACTOR)
    logging.info(f"Agent created")

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir=os.path.join(actual_path,"../runs", current_time + '_' + socket.gethostname()))

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
            
            predictions = []
            #predict future:
            future_observation = observation
            for i in range(STEPS_IN_FUTURE):
                future_action = agent.select_action(env, future_observation, exploration_on=True)
                future_state_predicted = agent.state_model(state_input=future_observation, action_input=future_action)
                future_distance_predicted = agent.critic_model(future_state_predicted)
                #check if a path of states to the goal has been found
                for s in future_state_predicted[:16]:
                    if s >= 0.9:
                        print("\n\n\n!!!Found path to goal!!!!\n\n\n")
                        break
                predictions.append(Prediction(action=future_action, 
                                              state=future_state_predicted, 
                                              distance=future_distance_predicted))
                
                future_observation = future_state_predicted
                
            # action = agent.select_action(env, observation, exploration_on=True) #equivalent to line below
            action = predictions[0].action
            previous_observation = observation
            
            observation, reward, done, info = env.step(action) # Reward not used
            
            # print(f"Observation: {observation}")
            goal_vector = observation[:16]
            hazards_vector = observation[16:]
            
            #calculate the distance
            if max(goal_vector) >= 0.9:
                distance_to_goal = 0
                optimal_distance = 0
            else:
                distance_to_goal = 1/max(max(goal_vector),0.001) #(1-max(observation[(-3*16):(-2*16)]))*
                optimal_distance = 1/max(max(goal_vector),0.001)
                if max(hazards_vector) >= 0.5: #observation hazard
                    distance_to_goal += 1/(1-max(hazards_vector))
            
            # distance_to_goal = 1-(1/(1+distance_to_goal))
            logging.debug(f"{episode_steps}. The distance to goal is: {distance_to_goal}")
            
            
            episode_distance_traveled += np.linalg.norm(observation-previous_observation)
            
            # Memory buffer components
            #'state', 
            #'previous_state', 
            #'action', 
            #'distance_critic', 
            #'distance_with_state_prediction',  
            #'optimal_distance', 
            #'predictions')
            agent.memory_buffer.push(observation, #state
                                     previous_observation, #previous_state 
                                     action, #action
                                     0, #distance_critic
                                     0, #distance_with_state_prediction
                                     distance_to_goal, #optimal_distance is the distance to goal when you are enough far away from the obstacles
                                     predictions)
            
            
            
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
        if epoch % 10 == 0 and epoch != 0:
            now = datetime.datetime.now()
            date = now.strftime("%Y%m%d_%H")
            torch.save(agent.actor_model.state_dict(), './models/'+date+'_actor_model_' + str(epoch) + '.pth')
            torch.save(agent.critic_model.state_dict(), './models/'+date+'_critic_model_' + str(epoch) + '.pth')
            torch.save(agent.state_model.state_dict(), './models/'+date+'_state_model_' + str(epoch) + '.pth')
    writer.close()
        
if __name__ == '__main__':
    main()

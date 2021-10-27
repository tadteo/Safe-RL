#!/usr/bin/env python3

import gym
import numpy as np
from collections import deque
import random

import torch
from torch import nn


RENDER = True
EPOCHS = 15 #30
STEPS_PER_EPOCH = 4000
STEPS_IN_FUTURE = 10 #Number of steps in the future to predict 
UPDATE_FREQUENCY = 10 #execute a step of training after this number of environment steps
STEPS_OF_TRAINING = 3 #execute this number of gradient descent steps at each training step
GOAL_STATE = [0,0,0,0] #Goal state for cartpole

class ActorModel(nn.Module):
    def __init__(self, input_size, output_size, sizes=[32,64,32], activation=nn.ReLU()):
        super(ActorModel, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1],sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(sizes[-1],output_size))
        
        self.net = nn.Sequential(*self.layers)

        print("Actor Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        X = torch.Tensor(input)
        for l in self.layers:
            X = l(X)
        return X

class CriticModel(nn.Module):
    def __init__(self, input_size, output_size=1, sizes=[32,64,32], activation=nn.ReLU()):
        super(CriticModel, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1],sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(sizes[-1],output_size))
        
        self.net = nn.Sequential(*self.layers)

        print("Critic Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        X = torch.Tensor(input)
        for l in self.layers:
            X = l(X)
        return X

class StateModel(nn.Module):
    '''
    It is used to predict the next state given in input the state and the action taken 
    The input is state and the action, the output is the next state
    '''
    def __init__(self, state_size, action_size, action_layers_sizes=[8,16,32], state_layers_sizes=[32,128,64,32], layers_sizes=[64,64,32], activation=nn.ReLU()):
        super(StateModel, self).__init__()
        
        #State Head
        self.state_layers = []
        self.state_layers.append(nn.Linear(state_size,state_layers_sizes[0]))
        self.state_layers.append(activation)
        for i in range(1,len(state_layers_sizes)):
            self.state_layers.append(nn.Linear(state_layers_sizes[i-1],state_layers_sizes[i]))
            self.state_layers.append(activation)
        self.state_layers.append(nn.Linear(state_layers_sizes[-1],1))
        
        self.state_features = nn.Sequential(*self.state_layers)

        #Action Head
        self.action_layers = []
        self.action_layers.append(nn.Linear(action_size,action_layers_sizes[0]))
        self.action_layers.append(activation)
        for i in range(1,len(action_layers_sizes)):
            self.action_layers.append(nn.Linear(action_layers_sizes[i-1],action_layers_sizes[i]))
            self.action_layers.append(activation)
        self.action_layers.append(nn.Linear(action_layers_sizes[-1],1))
        
        self.action_features = nn.Sequential(*self.action_layers)

        #Final Piecewise Action
        self.layers = []
        self.layers.append(nn.Linear(state_layers_sizes[-1]+action_layers_sizes[-1], layers_sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers_sizes[-1],state_size))

        self.features_layers = nn.Sequential(*self.layers)

        print("State Model structure: ", self.features_layers, "\n\n")
        # for name, param in self.features_layers.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def forward(self, state_input, action_input):
        X_state = torch.Tensor(state_input)
        for l in self.state_layers:
            X_state = l(X_state)
        
        X_action = torch.Tensor(action_input)
        for l in self.action_layers:
            X_action = l(X_action)
        
        X = torch.cat((X_state,X_action))
        for l in self.layers:
            X = l(X)
        
        return X


class ACSAgent:
    def __init__(self,state_size, action_size, id_nr=0):
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        #Hyperparameters:
        self.batch_size = 32
        self.memory_size = 10000
        self.learning_rate = 0.001
        
        #Replay buffer
        self.memory_buffer = deque(maxlen=self.memory_size)

        #Create networks
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))
        
        self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size).to(device)
        self.critic_model = CriticModel(input_size=self.state_size,output_size=(1)).to(device)
        self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size).to(device)
    
    def train_models(self):
        mini_batch = random.sample(self.memory_buffer)


def main():
    import argparse
    import logging, sys
    
    ### Parsing of the arguments
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0') #Safexp-PointGoal1-v0

    args = parser.parse_args()
    env = gym.make(args.env)

    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, action_size=action_size)
    logging.info(f"Agent created")

    total_num_episodes = 0
    for e in range(EPOCHS):    
        number_of_episodes = 0      
        cumulative_epoch_distance = 0
        sum_episode_length = 0
        #reset environement at the start of each epoch #TODO: Maybe better to put outside the EPOCHS cycle (we can start a new epoch in the middle of an episode)
        observation, episode_return, episode_distance, episode_length = env.reset(), 0, 0, 0

        for t in range(STEPS_PER_EPOCH):
            if RENDER:
                env.render()
            
            observation_old = observation
            logging.debug(f"The Observation type is: {type(observation)}") 
            action = agent.actor_model(observation)  #TODO: Error when running in gpu: "RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat2 in method wrapper_mm)"

            logging.debug(f"The action taken is: {action}")
            
            distance_predicted = agent.critic_model(observation)
            logging.debug(f"The distance predicted is: {distance_predicted}")

            predicted_new_state = agent.state_model(state_input=observation,action_input=action)
            logging.debug(f"The predicted new state is: {predicted_new_state}")

            predictions = []
            future_state = predicted_new_state
            for i in range(STEPS_IN_FUTURE):
                future_action = agent.actor_model(future_state)
                logging.debug(f"The action {i} steps in the future taken is: {future_action}")
                
                future_distance_predicted = agent.critic_model(future_state)
                logging.debug(f"The distance {i} steps in the future predicted is: {future_distance_predicted}")

                future_predicted_new_state = agent.state_model(state_input=future_state,action_input=action)
                logging.debug(f"The new state predicted {i} steps in the future is: {future_predicted_new_state}")
                predictions.append({"action":future_action, "distance":future_distance_predicted, "state":future_predicted_new_state})
                future_state=future_predicted_new_state

                      
            #take a step in the environment
            observation, reward, done, info = env.step(action) #the reward information is not used, just the states
            
            #calculate the distance
            distance = np.linalg.norm(GOAL_STATE-observation) #to train Actor Model
            
            distance_from_previous_state = np.linalg.norm(observation-observation_old) #to train Actor Model
            distance_travelled_to_goal = distance - np.linalg.norm(GOAL_STATE-observation_old) #to plot and evaluate performances while training of actor model
            
            distance_to_goal = np.linalg.norm(predictions[0]["state"]-observation) #to train Critic Model
            for i in range(1,len(predictions)):
                distance_to_goal += np.linalg.norm(predictions[i]["state"] - predictions[i-1]["state"])
            distance_to_goal += agent.critic_model(predictions[-1]["state"]) #adding the predicted distance from last state predicted
            
            delta_distances = distance_to_goal-distance_predicted #to plot and evaluate performances while training of critic model
            
            episode_length += 1
            episode_distance += distance_from_previous_state

            agent.memory_buffer.append({"state":observation, "action": action, "distance_critic":distance_predicted, "distance_with_state_prediction":distance_to_goal, "state":predicted_new_state, "predictions":predictions})    
  
            #Reset the environment and save data
            if done or (t == STEPS_PER_EPOCH-1):
                cumulative_epoch_distance += episode_distance
                sum_episode_length += episode_length
            
            if ((e*STEPS_PER_EPOCH)+t)%UPDATE_FREQUENCY == 0:
                print("Training networks")
                for j in range(STEPS_OF_TRAINING):
                    agent.train_models()

if __name__ == '__main__':
    main()

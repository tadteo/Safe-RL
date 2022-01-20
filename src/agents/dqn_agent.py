#!/usr/bin/env python3

import torch
import numpy as np
from torch import nn, optim
from torch.distributions import Categorical

from agents.agent import Agent

from utils import *

from models.dqn_model import DQNModel

class DQNAgent(Agent):
    def __init__(self,state_size, 
                 action_size,
                 state_memory_size=1,
                 batch_size=32,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 has_continuous_action_space=False,
                 reward_type="DISTANCE",
                 path_for_trained_models=None,
                 dqn_model_weights=None,
                 goal_start=0,
                 goal_end=0,
                 id_nr=0):
        
        super(self.__class__, self).__init__(state_size=state_size,
                action_size=action_size,
                state_memory_size=state_memory_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                has_continuous_action_space=has_continuous_action_space,
                path_for_trained_models=path_for_trained_models,
                )
        
        self.dqn_model_weights = dqn_model_weights

        
        self.reward_type = reward_type
        
        #Hyperparameters
        self.gamma = gamma
        
        self.goal_start = goal_start
        self.goal_end = goal_end
                
        self.model_optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate)

    def create_networks_from_scratch(self):
        self.dqn_model = DQNModel(input_size=self.state_size*self.state_memory_size,output_size=self.action_size)
        self.dqn_model = self.dqn_model.to(self.device)

        self.target_model = DQNModel(input_size=self.state_size*self.state_memory_size,output_size=self.action_size)
        self.target_model.load_state_dict(self.dqn_model.state_dict())
        self.target_model.eval()
        
    def create_networks_from_weights(self, path_for_trained_models, data):
        import os

        #Create networks
        self.dqn_model = DQNModel(input_size=self.state_size*self.state_memory_size,output_size=self.action_size)
        
        #Loading the weights of the models
        if(self.dqn_model_weights != None):
            self.dqn_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, self.dqn_model_weights)))
            self.dqn_model.eval()
            
        self.dqn_model = self.dqn_model.to(self.device)
        
        self.target_model = DQNModel(input_size=self.state_size*self.state_memory_size,output_size=self.action_size)
        self.target_model.load_state_dict(self.dqn_model.state_dict())
        self.target_model.eval()
        
        
    def save_models(self, experiment_name, epoch, path_for_trained_models=None):
        import os
        import json
        
        if path_for_trained_models == None:
            path = f"../models/{experiment_name}/"
        else:
            path = os.path.join(path_for_trained_models, experiment_name)
            
        if not os.path.isdir(path):
            os.makedirs(path)
        #Save the dimensions of the models
        size_actor = []
        for l in self.dqn_model.layers:
            if l.__class__.__name__ == "Linear":
                    size_actor.append(int(l.out_features))
        size_critic = []
        
        data = {"dqn_model": size_actor}
        with open(os.path.join(path,f'model_sizes.json'),'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        #Save parameter of the models
        torch.save(self.dqn_model.state_dict(), os.path.join(path,f'dqn_model_{epoch}.pth'))
        
    def select_action(self, exploration_on = True) -> torch.tensor:
        """Select an action with exploration given the current policy

        Args:
            env ([type]): The environment in which the agent is playing
            obs ([type]): The observation obtained by the environment
        """
        obs = list(self.state_memory)
        obs = torch.stack(obs).to(self.device)
        obs = obs[None,:]
        with torch.no_grad():
            if exploration_on and random.random() < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                Qs = self.dqn_model(obs)
                action = int(torch.argmax(Qs).item())
            return action        

    def train_step(self):
        
        if len(self.replay_memory_buffer) < self.batch_size:
            return 0 ,0 ,0
        
        #Sample batch from memory
        transitions = self.replay_memory_buffer.sample(self.batch_size)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). 
        # This converts batch-array of Transitions to Transition of batch-arrays.
        mini_batch = Transition(*zip(*transitions))
        
        state_batch = list(mini_batch.state)
        for i in range(len(mini_batch.state)):
            state_batch[i]=list(mini_batch.state[i])
        for i in range(len(state_batch)):
            state_batch[i] = torch.stack(state_batch[i]).to(self.device)
        state_batch = torch.stack(state_batch).to(self.device)
        
        previous_state_batch = list(mini_batch.previous_state)
        for i in range(len(mini_batch.previous_state)):
            previous_state_batch[i]=list(mini_batch.previous_state[i])
        for i in range(len(previous_state_batch)):
            previous_state_batch[i] = torch.stack(previous_state_batch[i]).to(self.device)
        previous_state_batch = torch.stack(previous_state_batch).to(self.device)
        
        tmp = np.array(mini_batch.action)        
        action_batch = torch.tensor(tmp).float().to(self.device).type(torch.int64) 
        reward_batch = torch.tensor(mini_batch.reward).float().to(self.device)
        distance_batch = torch.tensor(mini_batch.distance).float().to(self.device)
        distance_critic_batch = torch.tensor(mini_batch.distance_critic).float().to(self.device)
        distance_with_state_prediction_batch = torch.tensor(mini_batch.distance_with_state_prediction).float().to(self.device)
        done_batch = torch.tensor(mini_batch.done).float().to(self.device)
        
        if self.reward_type == "DISTANCE":
            # Training Step with distance batch
            target = self.dqn_model(previous_state_batch)
            Q_future = torch.min(self.dqn_model(state_batch),-1)[0]
            for i in range(len(state_batch)):
                action_index=int(action_batch[i].item())
                target[i][action_index] = distance_batch[i] + Q_future[i]-distance_batch[i]
            
            criterion_Q = torch.nn.MSELoss()
            # Loss_Q = criterion_Q(target, self.dqn_model(previous_state_batch))
            loss_Q = criterion_Q(target, self.dqn_model(state_batch))
            
            self.model_optimizer.zero_grad()
            self.dqn_model.zero_grad()
            loss_Q.backward()
            self.model_optimizer.step()
        elif self.reward_type == "REWARD":
            # Training Step with reward batch DQN
            target = self.dqn_model(previous_state_batch)
           
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s[-1] is not None,
                                                state_batch)), device=self.device, dtype=torch.bool)
            
            # print(f"non_final_mask: {non_final_mask}")
            # print(f"State batch: {state_batch.shape}")
            non_final_next_states = torch.stack([s for s in state_batch
                                                        if s is not None])
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # print(f"action_batch shape: {action_batch.shape}")
            # print(f"self.dqn_model(state_batch).shape: {self.dqn_model(state_batch).shape}")
            state_action_values = self.dqn_model(state_batch).gather(1, action_batch.view(-1, 1))
            
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            # print(f"Non final next states = {non_final_next_states.shape}")
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss_Q = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Optimize the model
            self.model_optimizer.zero_grad()
            loss_Q.backward()
            # for param in self.dqn_model.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.model_optimizer.step()
            
        self.update_epsilon()
                
        return 0, loss_Q.item(), 0, self.epsilon


#!/usr/bin/env python3

import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

from agents.agent import Agent

from utils import *

from models.actor_model import ActorModel
from models.critic_model import CriticModel
from models.state_model import StateModel

import torch.nn.functional as F

class SACAgent(Agent):
    def __init__(self,state_size, 
                 action_size,
                 state_memory_size=1,
                 batch_size=32,
                 alpha=0.2, # Entropy regularization coefficient.
                 gamma=0.99, # Discount factor. (Always between 0 and 1.)
                 polyak = 0.9,
                 learning_rate=0.001, # Learning rate (used for both policy and value learning).
                 initial_variance=None, final_variance=None, 
                 discount_factor=None,
                 has_continuous_action_space=True,
                 path_for_trained_models=None,
                 actor_model_weights=None,
                 critic_model_weights=None,
                 state_model_weights=None,
                 goal_start=0,
                 goal_end=0,
                 id_nr=0):
        
        super(self.__class__, self).__init__(state_size=state_size,
                action_size=action_size,
                state_memory_size=state_memory_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                has_continuous_action_space=has_continuous_action_space,
                path_for_trained_models=path_for_trained_models,
                )
        
        self.actor_model_weights = actor_model_weights
        self.critic_model_weights = critic_model_weights
        self.state_model_weights = state_model_weights
        
        #Hyperparameters:
        self.alpha = alpha #used in train critic
        self.gamma = gamma #used in train critic
        self.polyak = polyak
        
        self.initial_variance = initial_variance
        self.final_variance = final_variance
        self.exploration_factor = 1.0
        
        self.discount_factor = discount_factor
        
        self.goal_start = goal_start
        self.goal_end = goal_end
                
       
    def create_networks_from_scratch(self):
        import copy
        import itertools
        
        self.actor_model = ActorModel(input_size=self.state_size*self.state_memory_size,output_size=self.action_size, has_continuous_action_space=self.has_continuous_action_space)
        self.critic_model = CriticModel(input_size=self.state_size*self.state_memory_size,output_size=(1))
        self.critic_model_2 = CriticModel(input_size=self.state_size*self.state_memory_size,output_size=(1))
        self.state_model = StateModel(state_size=self.state_size, state_memory_size=self.state_memory_size,action_size=self.action_size, has_continuous_action_space=self.has_continuous_action_space)
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model = self.critic_model.to(self.device)
        self.critic_model_2 = self.critic_model_2.to(self.device)
        self.state_model = self.state_model.to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_params =  itertools.chain(self.critic_model.parameters(), self.critic_model_2.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=self.learning_rate)

        self.target_critic_1 = CriticModel(input_size=self.state_size*self.state_memory_size,output_size=(1))
        self.target_critic_1.load_state_dict(copy.deepcopy(self.critic_model.state_dict()))
        self.target_critic_2 = CriticModel(input_size=self.state_size*self.state_memory_size,output_size=(1))
        self.target_critic_2.load_state_dict(copy.deepcopy(self.critic_model_2.state_dict()))

        
    def create_networks_from_weights(self, path_for_trained_models, data):
        import os

        #Create networks
        self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size[0], layers_sizes=data['actor_model'][:-1], has_continuous_action_space=self.has_continuous_action_space)
        self.critic_model = CriticModel(input_size=self.state_size,output_size=(1), layers_sizes=data['critic_model'][:-1])
        self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size[0], layers_sizes=data['state_model'][:-1], has_continuous_action_space=self.has_continuous_action_space)
        
        #print(os.path.join(path_for_trained_models, actor_model_weights))
        #Loading the weights of the models
        if(self.actor_model_weights != None):
            self.actor_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, self.actor_model_weights)))
            self.actor_model.eval()
        if(self.critic_model_weights != None):
            self.critic_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, self.critic_model_weights)))
            self.critic_model.eval()
        if(self.state_model_weights != None):
            self.state_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, self.state_model_weights)))        
            self.state_model.eval()
            
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model = self.critic_model.to(self.device)
        self.state_model = self.state_model.to(self.device)
        
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
        for l in self.actor_model.layers:
            if l.__class__.__name__ == "Linear":
                    size_actor.append(int(l.out_features))
        size_critic = []
        for l in self.critic_model.layers:
            if l.__class__.__name__ == "Linear":
                    size_critic.append(int(l.out_features))
        size_state = []
        for l in self.state_model.layers:
            if l.__class__.__name__ == "Linear":
                    size_state.append(int(l.out_features))
        
        data = {"actor_model": size_actor, "critic_model": size_critic, "state_model": size_state}
        with open(os.path.join(path,f'model_sizes.json'),'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        #Save parameter of the models
        torch.save(self.actor_model.state_dict(), os.path.join(path,f'actor_model_{epoch}.pth'))
        torch.save(self.critic_model.state_dict(), os.path.join(path,f'critic_model_{epoch}.pth'))
        torch.save(self.state_model.state_dict(), os.path.join(path,f'state_model_{epoch}.pth'))
        
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
            action_dist = self.actor_model(obs)
            if self.has_continuous_action_space:
                if exploration_on and random.random() <= self.exploration_factor:
                    action = action_dist.rsample()
                else:
                    action = action_dist.mean
                action = torch.tanh(action)
                # action = action * self.action_range[0] #TODO add maximum range for actions
                return action.detach().numpy()
            else:
                if exploration_on and random.random() <= self.exploration_factor:
                    action = Categorical(logits=action_dist).sample()
                else:
                    action = torch.argmax(action_dist)
                return int(action.detach().numpy())
    
    def train_critic(self, previous_state_batch, state_batch, predictions_batch, distance_batch):
        # Compute targets for the Q functions
        
        q1 = self.critic_model(state_batch)
        q2 = self.critic_model_2(state_batch)
        
        with torch.no_grad():
            
            if self.has_continuous_action_space:
                action_dist = self.actor_model(state_batch)
                
                action=action_dist.rsample()
                log_prob = action_dist.log_prob(action).sum(axis=-1)
                log_prob = log_prob - (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
            else:
                action_prob = torch.max(self.actor_model(state_batch),-1)[0]
                # print(f"action_prob: {action_prob}, type: {type(action_prob)}")
                log_prob = torch.log(action_prob)
                # print(f"log_prob: {log_prob}, type: {type(log_prob)}")
                
            q1_pi_target = self.target_critic_1(state_batch)
            q2_pi_target = self.target_critic_2(state_batch)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            y = distance_batch + self.gamma*(q_pi_target-self.alpha*log_prob)

        loss_critic_1 = ((q1 - y)**2).mean() 
        loss_critic_2 = ((q2 - y)**2).mean()
        
        loss_critic = loss_critic_1 + loss_critic_2
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        return loss_critic 
    
    def train_actor(self, obs):
        
        if self.has_continuous_action_space:
            action_dist = self.actor_model(obs)
            
            action=action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        else:
            action_prob = torch.max(self.actor_model(obs),-1)[0]
            # print(f"action_prob: {action_prob}, type: {type(action_prob)}")
            log_prob = torch.log(action_prob)        

        loss_actor = (torch.minimum(self.critic_model(obs),self.critic_model_2(obs))-self.alpha*log_prob).mean()
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        return loss_actor
    
    def train_state(self, previous_state_batch, state_batch, action_batch):
        #Compute states and predicted states
        state_predictions = self.state_model(previous_state_batch,action_batch)
        criterion_state = nn.MSELoss()
        loss_state = criterion_state(state_predictions,state_batch[:,-1].clone().detach())
        self.state_optimizer.zero_grad()
        self.state_model.zero_grad()
        loss_state.backward()
        self.state_optimizer.step()

        return loss_state

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
        action_batch = torch.tensor(tmp).float().to(self.device)
        distance_critic_batch = torch.tensor(mini_batch.distance_critic).float().to(self.device)
        distance_with_state_prediction_batch = torch.tensor(mini_batch.distance_with_state_prediction).float().to(self.device)
        # distance_with_state_prediction_batch = torch.stack(mini_batch.distance_with_state_prediction).to(self.device).float()

        loss_critic = self.train_critic(previous_state_batch, state_batch, mini_batch.predictions, distance_critic_batch)
        loss_state = self.train_state(previous_state_batch, state_batch, action_batch)
        loss_actor = self.train_actor(state_batch)
        
        self.exploration_factor -= self.exploration_factor*0.05

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1.0 - self.polyak) * param.data)
        
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_model_2.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1.0 - self.polyak) * param.data)
        
                
        return loss_actor.item(), loss_critic.item(), loss_state.item()


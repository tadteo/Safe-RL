#!/usr/bin/env python3

import argparse
import logging, sys
from turtle import distance
import gym
import safety_gym
import numpy as np
from collections import namedtuple, deque
import random
from datetime import datetime

import torch
from torch import nn

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from utils import *
from models.actor_model import ActorModel
from models.critic_model import CriticModel
from models.state_model import StateModel

import copy

class old_SACAgent:
    def __init__(self,state_size, 
                 action_size, 
                 batch_size, 
                 discount_factor, 
                 id_nr=0):
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        #Hyperparameters:
        self.batch_size = batch_size
        self.memory_size = 100
        self.learning_rate = 0.001
        
        self.discount_factor = discount_factor
        self.exploration_epsilon = INITIAL_EPSILON

        #Replay buffer
        self.memory_buffer = ReplayMemory(capacity=self.memory_size)

        #Device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        if self.has_continuous_action_space:
            self.action_size = action_size
            print("Self.action_size: ", self.action_size)

        #Create networks
        self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size[0])
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model = CriticModel(input_size=self.state_size,output_size=(1))
        self.critic_mode_1 = self.critic_model.to(self.device)
        self.critic_model_2 = CriticModel(input_size=self.state_size,output_size=(1))
        self.critic_model_2 = self.critic_model_2.to(self.device)
        self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size[0])
        self.state_model = self.state_model.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_model_2.parameters(), lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=self.learning_rate)

        self.target_critic_1 = CriticModel(input_size=self.state_size,output_size=(1))
        self.target_critic_1.load_state_dict(copy.deepcopy(self.critic_model.state_dict()))
        self.target_critic_2 = CriticModel(input_size=self.state_size,output_size=(1))
        self.target_critic_2.load_state_dict(copy.deepcopy(self.critic_model_2.state_dict()))
    
    def save_models(self, experiment_name, epoch):
        import os
        import json
        
        path = f"../models/{experiment_name}/"
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
    
    def select_action(self, env, obs, exploration_on = False) -> torch.tensor:
        """Select an action with exploration given the current policy

        Args:
            env ([type]): The environment in which the agent is playing
            obs ([type]): The observation obtained by the environment
        """
        if (type(obs) ==np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if exploration_on:
            if has_continuous_action_space:
                action_dist = self.actor_model(obs)
                action = action_dist.rsample()
                # action = action_mean
            else:
                # logging.debug(f"The Observation type is: {observation}")
                action_probs = self.actor_model(obs)
                # logging.debug(f"The action probs are: {action_probs}")
                action_raw= action_probs
                dist = Categorical(action_probs)
                action = dist.sample()           
        else:
            if has_continuous_action_space:
                action_dist = self.actor_model(obs)
                action = action_dist.mean
            else:
                # logging.debug(f"The Observation type is: {observation}")
                action_probs = self.actor_model(obs)
                # logging.debug(f"The action probs are: {action_probs}")
                action_raw= action_probs
                dist = Categorical(action_probs)
                action = dist.sample()
            
        return action.cpu().detach().numpy()
    
    def train_critic(self, distance_batch, state_batch):
        # Compute targets for the Q functions
        
        alpha = 0.2
        gamma = 0.99
        action_dist = self.actor_model(state_batch)
        log_prob = action_dist.log_prob(action_dist.mean)
        # print(log_prob)
        
        # print(self.critic_model(state_batch))
        # print(self.critic_model_2(state_batch))
        y = distance_batch + gamma*(torch.minimum(self.target_critic_1(state_batch),self.target_critic_2(state_batch))-alpha*log_prob)
        
        criterion_critic = nn.MSELoss()
        # loss_critic = criterion_critic(self.critic_model(previous_state_batch),target_Q) #TODO: Extract also the distances in the predictions t have  a mix of distances with and without predictions
        loss_critic = criterion_critic(self.critic_model(state_batch),y) #correct
        self.critic_optimizer_1.zero_grad()
        loss_critic.backward()
        self.critic_optimizer_1.step()
        
        
        alpha = 0.2
        gamma = 0.99
        action_dist = self.actor_model(state_batch)
        log_prob = action_dist.log_prob(action_dist.mean)
        # print(log_prob)
        
        # print(self.critic_model(state_batch))
        # print(self.critic_model_2(state_batch))
        y = distance_batch + gamma*(torch.minimum(self.target_critic_1(state_batch),self.target_critic_2(state_batch))-alpha*log_prob)
        
        criterion_critic_2 = nn.MSELoss()
        loss_critic_2 = criterion_critic_2(self.critic_model_2(state_batch),y) #correct
        self.critic_optimizer_2.zero_grad()
        loss_critic_2.backward()
        self.critic_optimizer_2.step()
        
        return loss_critic
    
    def train_actor(self, obs, action):
        dist = self.actor_model(obs)

        action_tilde = torch.tanh(dist.mean + dist.stddev*torch.randn(dist.mean.shape))
        log_prob_tilde = dist.log_prob(action_tilde).sum(-1,keepdim=True)
        
        alpha = 0.02
        
        loss_actor = (torch.minimum(self.critic_model(obs),self.critic_model_2(obs))-alpha*log_prob_tilde).sum()/self.batch_size
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        return loss_actor
    
    def train_state(self, previous_state_batch, state_batch, action_batch):
        #Compute states and predicted states
        state_predictions = self.state_model(previous_state_batch,action_batch)
        criterion_state = nn.MSELoss()
        loss_state = criterion_state(state_predictions,state_batch)
        self.state_optimizer.zero_grad()
        loss_state.backward()
        self.state_optimizer.step()

        return loss_state

    def train(self):
        """Train the agent"""
        
        if len(self.memory_buffer) < self.batch_size:
            return 0 ,0 ,0
        
        #Sample batch from memory
        transitions = self.memory_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.

        mini_batch = Transition(*zip(*transitions))
        
        state_batch = np.array(mini_batch.state)
        state_batch = torch.tensor(state_batch).float().to(self.device)
        
        previous_state_batch = np.array(mini_batch.previous_state)
        previous_state_batch = torch.tensor(previous_state_batch).float().to(self.device)
        
        action_batch = np.array(mini_batch.action)        
        action_batch = torch.tensor(action_batch).float().to(self.device)

        distance_batch = np.array(mini_batch.distance)
        distance_batch = torch.tensor(distance_batch).float().to(self.device)

        distance_critic_batch = torch.tensor(mini_batch.distance_critic).float().to(self.device)

        distance_with_state_prediction_batch = torch.tensor(mini_batch.distance_with_state_prediction).float().to(self.device)


        loss_critic = self.train_critic(distance_batch, state_batch)
        
        loss_state = self.train_state(previous_state_batch, state_batch, action_batch)
        
        loss_actor = self.train_actor(state_batch, action_batch)
        
        logging.info(f"Training step completed, returning")
        
        polyak = 0.9
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_model.parameters()):
            # print(f"target_param, param: {target_param.data}, {param.data}")
            target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * param.data)
        
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_model_2.parameters()):
            target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * param.data)
        
        return loss_actor.item(), loss_critic.item(), loss_state.item()

#!/usr/bin/env python3

import logging

import numpy as np

import torch
from torch import nn

from torch.distributions import Categorical

from src.agents.agent import Agent

from utils import *

from models.actor_model import ActorModel
from models.critic_model import CriticModel
from models.state_model import StateModel

class ACSAgent(Agent):
    def __init__(self,state_size, 
                 action_size,
                 batch_size=32, 
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
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        torch.autograd.set_detect_anomaly(True)

        
        #Hyperparameters:
        self.batch_size = batch_size
        self.memory_size = 100
        self.learning_rate = 0.001
        
        self.initial_variance = initial_variance
        self.final_variance = final_variance
        self.exploration_factor = 1.0
        
        self.discount_factor = discount_factor

        
        self.goal_start = goal_start
        self.goal_end = goal_end
        
        #Replay buffer
        self.memory_buffer = ReplayMemory(capacity=self.memory_size)

        #Device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        self.has_continuous_action_space = has_continuous_action_space
        self.action_size = action_size
        print("Self.action_size: ", self.action_size)
        
        if path_for_trained_models == None:
            #Create networks
            self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size, has_continuous_action_space=self.has_continuous_action_space)
            self.critic_model = CriticModel(input_size=self.state_size,output_size=(1))
            self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size, has_continuous_action_space=self.has_continuous_action_space)
        else:
            import os
            import json
            with open(path_for_trained_models+"/model_sizes.json", 'r') as f:
                data = json.load(f)
            #Create networks
            self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size[0], layers_sizes=data['actor_model'][:-1], has_continuous_action_space=self.has_continuous_action_space)
            self.critic_model = CriticModel(input_size=self.state_size,output_size=(1), layers_sizes=data['critic_model'][:-1])
            self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size[0], layers_sizes=data['state_model'][:-1], has_continuous_action_space=self.has_continuous_action_space)
            
            print(os.path.join(path_for_trained_models, actor_model_weights))
            #Loading the weights of the models
            if(actor_model_weights != None):
                self.actor_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, actor_model_weights)))
                self.actor_model.eval()
            if(critic_model_weights != None):
                self.critic_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, critic_model_weights)))
                self.critic_model.eval()
            if(state_model_weights != None):
                self.state_model.load_state_dict(torch.load(os.path.join(path_for_trained_models, state_model_weights)))        
                self.state_model.eval()
                
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model = self.critic_model.to(self.device)
        self.state_model = self.state_model.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=self.learning_rate)

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
        
    def select_action(self, obs, exploration_on = True) -> torch.tensor:
        """Select an action with exploration given the current policy

        Args:
            env ([type]): The environment in which the agent is playing
            obs ([type]): The observation obtained by the environment
        """
        # logging.info("Selecting action")
        
        if (type(obs) ==np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        with torch.no_grad():
            action_dist = self.actor_model(obs)
            if self.has_continuous_action_space:
                # logging.debug(f"Action mean: {action_mean}")
                # action_raw = action_mean
                if exploration_on and random.random() <= self.exploration_factor:
                    action = action_dist.rsample()
                else:
                    action = action_dist.mean
            else:
                if exploration_on and random.random() <= self.exploration_factor:
                    action = Categorical(logits=action_dist).sample()
                else:
                    action = torch.argmax(action_dist)

        return action.detach().numpy()
    
    def train_critic(self, total_number_of_steps, previous_state_batch, state_batch, predictions_batch, distance_batch):
        # #Compute Q values
        # alpha = 0.2
        # gamma = 0.99
        # print(f"State batch: {state_batch.shape}")
        # target_Q = distance_batch
        # # print(f"Q: {Q.size}, {Q}")
        # for i in range(self.batch_size):

        #     for p in predictions_batch[i]:                
        #         future_distance_predicted = self.critic_model(p.state)
        #         target_Q[i] += (future_distance_predicted.item() - target_Q[i])
        
        # action_mean = self.actor_model(state_batch)
        # next_action_batch = dist.rsample()
        # log_prob = dist.log_prob(next_action_batch).sum(-1,keepdim=True)
        # print("Log Prob: ", log_prob)
        # target_V = self.critic_model(state_batch) - alpha * log_prob
        # target_Q = distance_batch + self.discount_factor * target_V
        # target_Q = target_Q.detach()
        # target_Q = target_V.detach()
        
        # with torch.no_grad():
            # writer.add_scalar('Q/Predicted Q', self.critic_model(previous_state_batch).mean().item(), total_number_of_steps)
            # writer.add_scalar('Q/Target Q', target_Q.mean().item(), total_number_of_steps)
        criterion_critic = nn.MSELoss()
        # loss_critic = criterion_critic(self.critic_model(previous_state_batch),target_Q) #TODO: Extract also the distances in the predictions t have  a mix of distances with and without predictions
        critic=self.critic_model(state_batch)
        # print(f"Critic: {critic.size()}")
        loss_critic = criterion_critic(critic,distance_batch) #correct
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        return loss_critic
    
    def train_actor(self, previous_obs, obs,  action, Q):
        
        action_dist = self.actor_model(obs)
        # previous_action_dist = self.actor_model(previous_obs)

        #calculate the log_prob of the batch
        # action_tilde = torch.tanh(action_dist.mean + action_dist.stddev*torch.randn(action_dist.mean.shape,device=self.device))
        #  = log_prob.sum(-1,keepdim=True)
        
        if self.has_continuous_action_space:
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = torch.max(action_dist, dim=1)[0]
            print("Action",action.shape)
            log_prob = torch.log(action)

        # print(f"Log prob: {log_prob}")
                
        distance_list = []
        #calculate the distance
        for i in range(self.batch_size):
            distance_list.append(calculate_distance(obs[i],self.goal_start,self.goal_end))
        
        distance_list = torch.tensor(distance_list).to(self.device)
        
        print("Distance list: ", distance_list.shape)
        print("Log prob: ", log_prob.shape)
        loss_actor = ((0.01*log_prob) + distance_list ).mean() # + ((0.01*log_prob) - Q).sum()) / 2

        print(f"Loss actor: {loss_actor}")
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        return loss_actor
    
    def train_state(self, previous_state_batch, state_batch, action_batch):
        #Compute states and predicted states
        state_predictions = self.state_model(previous_state_batch,action_batch)
        criterion_state = nn.MSELoss()
        loss_state = criterion_state(state_predictions,state_batch.clone().detach())
        self.state_optimizer.zero_grad()
        self.state_model.zero_grad()
        loss_state.backward()
        self.state_optimizer.step()

        return loss_state

    def train_off_policy(self, writer , total_number_of_steps):
        
        if len(self.memory_buffer) < self.batch_size:
            return 0 ,0 ,0
        
        #Sample batch from memory
        transitions = self.memory_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # print("Transitions: ", transitions)
        mini_batch = Transition(*zip(*transitions))
        state_batch = np.array(mini_batch.state)
        state_batch = torch.tensor(state_batch).float().to(self.device)
        # state_batch = torch.stack(mini_batch_state).to(self.device).float()
        
        previous_state_batch = np.array(mini_batch.previous_state)
        # previous_state_batch = torch.tensor(mini_batch.previous_state).float().to(self.device)
        # previous_state_batch = torch.stack(mini_batch_previous_state).to(self.device).float()
        
        tmp = np.array(mini_batch.action)        
        action_batch = torch.tensor(tmp).float().to(self.device)
        distance_critic_batch = torch.tensor(mini_batch.distance_critic).float().to(self.device)
        distance_with_state_prediction_batch = torch.tensor(mini_batch.distance_with_state_prediction).float().to(self.device)
        # distance_with_state_prediction_batch = torch.stack(mini_batch.distance_with_state_prediction).to(self.device).float()
        # print(f"Optimal distance batch: {distance_batch.shape}")

        loss_critic = self.train_critic(total_number_of_steps, previous_state_batch, state_batch, mini_batch.predictions, distance_critic_batch)
        loss_state = self.train_state(previous_state_batch, state_batch, action_batch)
        loss_actor = self.train_actor(previous_state_batch, state_batch, action_batch, distance_critic_batch)
        
        self.exploration_factor -= self.exploration_factor*0.05

        writer.add_scalar("Exploration_factor", self.exploration_factor, total_number_of_steps)
        
        logging.info(f"Training step completed, returning")
        return loss_actor.item(), loss_critic.item(), loss_state.item()

def flatten_obs(obs):
    print(f"Observation: {obs}")
    obs_flat_size= sum([np.prod(i.shape) for i in obs.values()])
    flat_obs = np.zeros(obs_flat_size)
    offset = 0
    for k in sorted(obs.keys()):
        k_size = np.prod(obs[k].shape)
        flat_obs[offset:offset + k_size] = obs[k].flat
        offset += k_size
    obs = flat_obs
    print(f"Flat Observation: {obs}")
    return obs

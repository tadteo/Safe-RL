#!/usr/bin/env python3

import argparse
import logging, sys
import gym
import safety_gym
import numpy as np
from collections import namedtuple, deque
import random
from datetime import datetime

import torch
from torch import nn, optim
from torch._C import device, dtype
from torch.nn.modules import distance
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import MSELoss

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

ENV = 'Safexp-CarGoal1-v0'
RENDER = True
EPOCHS = 500
STEPS_PER_EPOCH = 10000
STEPS_IN_FUTURE = 25 #Number of steps in the future to predict 
UPDATE_FREQUENCY = 100 #execute a step of training after this number of environment steps
STEPS_OF_TRAINING = 3 #execute this number of gradient descent steps at each training step

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
# EPSILON = 0.1

###FOR OFF POLICY (SAC)
OFF_POLICY = True

has_continuous_action_space = True


Prediction = namedtuple('Prediction', ("action", 
                                       "distance", 
                                       "state"))

Transition = namedtuple('Transition', ('state', 
                                       'previous_state', 
                                       'action', 
                                       'distance_critic', 
                                       'distance_with_state_prediction',  
                                       'optimal_distance', 
                                       'predictions'))

## adding a comment

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))  #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class ActorModel(nn.Module):

    def __init__(self, input_size, output_size, sizes=[32,128,512,128,32], activation=nn.ReLU()):
        super(ActorModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        self.layers.append(nn.Linear(input_size, sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1],sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(sizes[-1],2*output_size))
        
        if has_continuous_action_space :
            pass
        else:
            self.layers.append(nn.Softmax(dim=-1))

        # self.layers.append(nn.Tanh())
        self.net = nn.Sequential(*self.layers)
        
        print("Actor Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.clone().detach().float()   
        
        # X = (X - X.min())/(X.max()-X.min())
        for l in self.layers:
            X = l(X)
            # print(X)
        
        mu, log_std = X.chunk(2, dim=-1)
        
        log_std = torch.tanh(log_std)
        # log_std_min, log_std_max = self.
        
        std = log_std.exp()
        
        dist = MultivariateNormal(mu, std)
        return dist

class CriticModel(nn.Module):
    '''
    It is used to predict the distance given in input the state of the environment 
    The input is state, the output is the distance to the goal state
    '''
    def __init__(self, input_size, output_size=1, sizes=[32,128,256,128,64,32], activation=nn.ReLU()):
        super(CriticModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        self.layers.append(nn.Linear(input_size, sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1],sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(sizes[-1],output_size))
        # self.layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.layers)

        print("Critic Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.clone().detach().float()    
        #X = torch.tensor(input, dtype=torch.float, device=self.device)
        X = X.clone().detach()
        for l in self.layers:
            X = l(X)
        return X

class StateModel(nn.Module):
    '''
    It is used to predict the next state given in input the state and the action taken 
    The input is state and the action, the output is the next state
    '''
    def __init__(self, state_size, action_size, action_layers_sizes=[8,16,32], state_layers_sizes=[32,128,64,32], layers_sizes=[32,64,128, 512,128, 64,32], activation=nn.Tanh()):
        super(StateModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        
        # #State Head
        # self.state_layers = []
        # self.state_layers.append(nn.Linear(state_size,state_layers_sizes[0]))
        # self.state_layers.append(activation)
        # for i in range(1,len(state_layers_sizes)):
        #     self.state_layers.append(nn.Linear(state_layers_sizes[i-1],state_layers_sizes[i]))
        #     self.state_layers.append(activation)
        # # self.state_layers.append(nn.Linear(state_layers_sizes[-1],1))
        
        # self.state_features = nn.Sequential(*self.state_layers)

        # #Action Head
        # self.action_layers = []
        # self.action_layers.append(nn.Linear(action_size,action_layers_sizes[0]))
        # self.action_layers.append(activation)
        # for i in range(1,len(action_layers_sizes)):
        #     self.action_layers.append(nn.Linear(action_layers_sizes[i-1],action_layers_sizes[i]))
        #     self.action_layers.append(activation)
        # # self.action_layers.append(nn.Linear(action_layers_sizes[-1],1))
        
        # self.action_features = nn.Sequential(*self.action_layers)

        #Final Piecewise Action
        self.layers = []
        self.layers.append(nn.Linear(action_size+state_size, layers_sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers_sizes[-1],state_size))
        

        self.features_layers = nn.Sequential(*self.layers)

        print(f"State Model structure:\n After concat layers: {self.features_layers}\n\n")
        # State input: {self.state_features}\nAction input: {self.action_features},
        # for name, param in self.features_layers.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def forward(self, state_input, action_input):
        if(type(state_input) == np.ndarray):
            state_input = torch.from_numpy(state_input).float().to(self.device)
        if(type(action_input) == np.ndarray):
            action_input = torch.from_numpy(action_input).float().to(self.device)
        
        X_state = state_input.clone().detach().float()
        # # print("State Input: ", X_state)
        # for l in self.state_layers:
        #     X_state = l(X_state)
        
        X_action = action_input.clone().detach().float()
        # # print("Action Input: ", X_action)
        # for l in self.action_layers:
        #     X_action = l(X_action)
        X = torch.cat((X_state,X_action), dim=-1)
        # print(f"X.shape {X.shape}")
        # print("Concatenated Input: ", X)
        for l in self.layers:
            X = l(X)
        
        return X

class SACAgent:
    def __init__(self,state_size, 
                 action_size, 
                 batch_size, 
                 initial_variance, final_variance, 
                 discount_factor, 
                 id_nr=0):
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        #Hyperparameters:
        self.batch_size = batch_size
        self.memory_size = 100
        self.learning_rate = 0.001
        
        self.initial_variance = initial_variance
        self.final_variance = final_variance
        self.actual_variance = initial_variance
        
        self.discount_factor = discount_factor
        self.exploration_epsilon = INITIAL_EPSILON

        #Replay buffer
        self.memory_buffer = ReplayMemory(capacity=self.memory_size)

        #Device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        if has_continuous_action_space:
            self.action_size = action_size
            print("Self.action_size: ", self.action_size)
            self.action_covariance_matrix = torch.diag(torch.full((self.action_size), self.actual_variance, dtype=torch.float, device=self.device))
            # print("Action Covariance Matrix: ", self.action_covariance_matrix)
        #Create networks
        self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size[0])
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model_1 = CriticModel(input_size=self.state_size,output_size=(1))
        self.critic_mode_1 = self.critic_model_1.to(self.device)
        self.critic_model_2 = CriticModel(input_size=self.state_size,output_size=(1))
        self.critic_model_2 = self.critic_model_2.to(self.device)
        self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size[0])
        self.state_model = self.state_model.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_model_1.parameters(), lr=self.learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_model_2.parameters(), lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=self.learning_rate)

        self.target_critic_params_1 = self.critic_model_1
        self.target_critic_params_2 = self.critic_model_2
        
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
                action_mean = self.actor_model(obs)
                # logging.debug(f"Action mean: {action_mean}")
                action_raw = action_mean
                dist = MultivariateNormal(action_mean, self.action_covariance_matrix)
                action = dist.rsample()
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
                action_mean = self.actor_model(obs)
                action = action_mean
            else:
                # logging.debug(f"The Observation type is: {observation}")
                action_probs = self.actor_model(obs)
                # logging.debug(f"The action probs are: {action_probs}")
                action_raw= action_probs
                dist = Categorical(action_probs)
                action = dist.sample()
        
        action_logprob = dist.log_prob(action)
            
        return action.cpu().detach().numpy()
    
    def train_critic(self, writer, total_number_of_steps, previous_state_batch, state_batch, predictions_batch, optimal_distance_batch):
        # #Compute Q values
        alpha = 0.2
        # # print(f"State batch: {state_batch.shape}")
        # target_Q = optimal_distance_batch
        # # print(f"Q: {Q.size}, {Q}")
        # for i in range(self.batch_size):

        #     for p in predictions_batch[i]:                
        #         future_distance_predicted = self.critic_model(p.state)
        #         target_Q[i] += (future_distance_predicted.item() - target_Q[i])
        
        action_mean = self.actor_model(state_batch)
        dist = MultivariateNormal(action_mean, self.action_covariance_matrix)
        next_action_batch = dist.rsample()
        log_prob = dist.log_prob(next_action_batch).sum(-1,keepdim=True)
        # print("Log Prob: ", log_prob)
        # target_V = self.critic_model(state_batch) - alpha * log_prob
        # target_Q = optimal_distance_batch + self.discount_factor * target_V
        # target_Q = target_Q.detach()
        # target_Q = target_V.detach()
        
        # with torch.no_grad():
            # writer.add_scalar('Q/Predicted Q', self.critic_model(previous_state_batch).mean().item(), total_number_of_steps)
            # writer.add_scalar('Q/Target Q', target_Q.mean().item(), total_number_of_steps)
        
        #Target Q functions:
        y = predictions_batch + self.discount_factor*(min(self.critic_model_1(state_batch),self.critic_model_2(state_batch))-self.alpha*log_prob)
        
        criterion_critic = nn.MSELoss()
        # loss_critic = criterion_critic(self.critic_model(previous_state_batch),target_Q) #TODO: Extract also the distances in the predictions t have  a mix of distances with and without predictions
        loss_critic = criterion_critic(self.critic_model_1(state_batch),y) #correct
        self.critic_optimizer_1.zero_grad()
        loss_critic.backward()
        self.critic_optimizer_1.step()
        
        criterion_critic_2 = nn.MSELoss()
        loss_critic_2 = criterion_critic_2(self.critic_model_2(state_batch),y) #correct
        self.critic_optimizer_2.zero_grad()
        loss_critic_2.backward()
        self.critic_optimizer_2.step()
        
        
        
        return loss_critic
    
    def train_actor(self, obs, action, reward, next_obs, done):
        pass
    
    def train_state(self, previous_state_batch, state_batch, action_batch):
        #Compute states and predicted states
        state_predictions = self.state_model(previous_state_batch,action_batch)
        criterion_state = nn.MSELoss()
        loss_state = criterion_state(state_predictions,state_batch)
        self.state_optimizer.zero_grad()
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
        # print("Mini batch: ", mini_batch.state)
        state_batch = np.array(mini_batch.state)
        state_batch = torch.tensor(state_batch).float().to(self.device)
        # print("Mini batch state: ", mini_batch_state)
        # state_batch = torch.stack(mini_batch_state).to(self.device).float()
        
        previous_state_batch = np.array(mini_batch.previous_state)
        # previous_state_batch = torch.tensor(mini_batch.previous_state).float().to(self.device)
        # previous_state_batch = torch.stack(mini_batch_previous_state).to(self.device).float()
        
        action_batch = np.array(mini_batch.action)        
        action_batch = torch.tensor(mini_batch.action).float().to(self.device)
        # action_batch = torch.stack(mini_batch.action).to(self.device).float()
        # print("Action Batch: ", action_batch.shape)
        distance_critic_batch = torch.tensor(mini_batch.distance_critic).float().to(self.device)
        # distance_critic_batch = torch.stack(mini_batch.distance_critic).to(self.device).detach().clone().float()
        optimal_distance_batch = torch.tensor(mini_batch.optimal_distance).float().to(self.device)
        # optimal_distance_batch = torch.stack(mini_batch.optimal_distance).to(self.device).float()
        distance_with_state_prediction_batch = torch.tensor(mini_batch.distance_with_state_prediction).float().to(self.device)
        # distance_with_state_prediction_batch = torch.stack(mini_batch.distance_with_state_prediction).to(self.device).float()
        # print(f"Optimal distance batch: {optimal_distance_batch.shape}")
        
        # for p in mini_batch.predictions:
            # print(f"Prediction: {type(p)} {p}")
        # predictions_batch = torch.cat(Prediction(*zip(*mini_batch.predictions))).to(self.device).float()

        loss_critic = self.train_critic(writer, total_number_of_steps, previous_state_batch, state_batch, mini_batch.predictions, optimal_distance_batch)
        
        loss_state = self.train_state(previous_state_batch, state_batch, action_batch)
        
        #Calculate the distribution of the action based on the state and caluculate the minimum value from the n samples of the distribution
        #Use the minimum and the original action value to calculate the loss
        
        dist = self.actor_model(state_batch)
        log_prob = dist.log_prob().sum(-1,keepdim=True)

        alpha = 0.01
        
        loss_actor = (min(self.critic_model_1(state_batch),self.critic_model_2(state_batch))-alpha*log_prob).sum()
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        logging.info(f"Training step completed, returning")
        
        self.target_critic_params_1 = 0.9*self.target_critic_params_1 + 0.1*self.critic_model_1.parameters()
        self.target_critic_params_2 = 0.9*self.target_critic_params_2 + 0.1*self.critic_model_2.parameters()
        
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

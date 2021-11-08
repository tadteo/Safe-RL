#!/usr/bin/env python3

import gym
import safety_gym
import numpy as np
from collections import namedtuple, deque
import random

import torch
from torch import nn
from torch.nn.modules.loss import MSELoss

from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter


RENDER = True
EPOCHS = 15 #30
STEPS_PER_EPOCH = 4000
STEPS_IN_FUTURE = 10 #Number of steps in the future to predict 
UPDATE_FREQUENCY = 100 #execute a step of training after this number of environment steps
STEPS_OF_TRAINING = 3 #execute this number of gradient descent steps at each training step
GOAL_STATE = [0,0,0,0] #Goal state for cartpole
GOAL_STATE = [0,0,0,0] #Goal state for car 

###FOR ON POLICY (PPO)
ON_POLICY = False

action_std = 0.6
eps_clip = 0.2          # clip parameter for PPO

###FOR OFF POLICY (SAC)
OFF_POLICY = True

has_continuous_action_space = True

def actor_loss(output, target):
    return torch.sum((output - target)**2)

def critic_loss(output, target):
    return torch.sum((output - target)**2)

def state_loss(output, target):
    return torch.sum((output - target)**2)

class ActorModel(nn.Module):

    def __init__(self, input_size, output_size, sizes=[32,64,32], activation=nn.Tanh()):
        super(ActorModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        self.layers.append(nn.Linear(input_size, sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1],sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(sizes[-1],output_size))
        
        if has_continuous_action_space :
            self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.Softmax(dim=-1))

        # self.layers.append(nn.Tanh())
        self.net = nn.Sequential(*self.layers)

        print("Actor Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


    def forward(self, input):
        X = torch.tensor(input, device=self.device).float()
        X = (X - X.min())/(X.max()-X.min())
        for l in self.layers:
            X = l(X)
            # print(X)
        return X

class CriticModel(nn.Module):
    '''
    It is used to predict the distance given in input the state of the environment 
    The input is state, the output is the distance to the goal state
    '''
    def __init__(self, input_size, output_size=1, sizes=[32,64,32], activation=nn.Tanh()):
        super(CriticModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        X = torch.tensor(input, device=self.device).float()
        for l in self.layers:
            X = l(X)
        return X

class StateModel(nn.Module):
    '''
    It is used to predict the next state given in input the state and the action taken 
    The input is state and the action, the output is the next state
    '''
    def __init__(self, state_size, action_size, action_layers_sizes=[8,16,32], state_layers_sizes=[32,128,64,32], layers_sizes=[32,64,128,64,32], activation=nn.ReLU()):
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
        X_state = torch.tensor(state_input, device=self.device).float()
        # # print("State Input: ", X_state)
        # for l in self.state_layers:
        #     X_state = l(X_state)
        
        X_action = torch.tensor(action_input, device=self.device).float()
        # # print("Action Input: ", X_action)
        # for l in self.action_layers:
        #     X_action = l(X_action)
        
        X = torch.cat((X_state,X_action), dim=-1)
        # print(f"X.shape {X.shape}")
        # print("Concatenated Input: ", X)
        for l in self.layers:
            X = l(X)
        
        return X

Prediction = namedtuple('Prediction', ("action", "distance", "state"))
Transition = namedtuple('Transition', ('state', 'previous_state', 'action', 'distance_critic', 'distance_with_state_prediction',  'optimal_distance', 'predictions'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))  #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class ACSAgent:
    def __init__(self,state_size, action_size, id_nr=0):
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr

        #Hyperparameters:
        self.batch_size = 32
        self.memory_size = 10000
        self.learning_rate = 0.001
        
        #ON Policy (PPO) hyperparameters:
        self.eps_clip = eps_clip

        #Replay buffer
        self.memory_buffer = ReplayMemory(capacity=self.memory_size)

        #Device info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        if has_continuous_action_space:
            self.action_size = action_size
            self.action_var = torch.full((action_size,), action_std * action_std).to(self.device)

        #Create networks
        self.actor_model = ActorModel(input_size=self.state_size,output_size=self.action_size)
        self.actor_model = self.actor_model.to(self.device)
        self.critic_model = CriticModel(input_size=self.state_size,output_size=(1))
        self.critic_model = self.critic_model.to(self.device)
        self.state_model = StateModel(state_size=self.state_size,action_size=self.action_size)
        self.state_model = self.state_model.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)
        self.state_optimizer = torch.optim.Adam(self.state_model.parameters(), lr=self.learning_rate)


    def train_off_policy(self):
        
        if len(self.memory_buffer) < self.batch_size:
            return 0 ,0 ,0
        
        #Sample batch from memory
        transitions = self.memory_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        mini_batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(mini_batch.state).to(self.device).float()
        previous_state_batch = torch.stack(mini_batch.previous_state).to(self.device).float()
        # print(f"Previous State batch: {previous_state_batch.shape}")
        action_batch = torch.stack(mini_batch.action).to(self.device).float()
        # print("Action Batch: ", action_batch.shape)
        distance_critic_batch = torch.cat(mini_batch.distance_critic).to(self.device).float()
        optimal_distance_batch = torch.stack(mini_batch.optimal_distance).to(self.device).float()
        print(f"Optimal distance batch: {optimal_distance_batch.shape}")
        # print(f"Mini batch predictions, {type(mini_batch.predictions)}, {mini_batch.predictions}\n\n\n\n")
        # for p in mini_batch.predictions:
            # print(f"Prediction: {type(p)} {p}")
        # print(f"Mini batch predictions, {type(Prediction(*zip(*mini_batch.predictions)))}, {Prediction(*zip(*mini_batch.predictions))}")
        
        # predictions_batch = torch.cat(Prediction(*zip(*mini_batch.predictions))).to(self.device).float()

        #Compute Q values
        alpha = 0.2
        gamma = 0.99
        # print(f"State batch: {state_batch.shape}")
        target_Q = optimal_distance_batch
        # print(f"Q: {Q.size}, {Q}")
        for i in range(self.batch_size):
            # print(f"Q[i] before predictions: {Q[i].size}, {Q[i]}")

            for p in mini_batch.predictions[i]:                
                future_distance_predicted = self.critic_model(p.state)
                target_Q[i] += (future_distance_predicted.item() - target_Q[i])
            # print(f"Q[i] after predictions: {Q[i].size}, {Q[i]}")
            # Q = distance_critic_batch - gamma(self.critic_model(state_batch)-alpha*torch.log(self.actor_model(state_batch))) ### TO be modified for the distance setting
        
        
        criterion_critic = nn.MSELoss()
        loss_critic = criterion_critic(self.critic_model(state_batch),optimal_distance_batch) #TODO: Extract also the distances in the predictions t have  a mix of distances with and without predictions
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #Compute states and predicted states
        state_predictions = self.state_model(previous_state_batch,action_batch)
        criterion_state = nn.MSELoss()
        loss_state = criterion_state(state_predictions,state_batch)
        self.state_optimizer.zero_grad()
        loss_state.backward()
        self.state_optimizer.step()
        
        #compute actions(?)
        actions = self.actor_model(state_batch)
        state_predictions = self.state_model(state_batch,actions)
        
        
        criterion_actor = nn.MSELoss()
        # print(f"Q LOSS actor = {torch.mean(Q)}")
        loss_actor = torch.mean(self.critic_model(state_predictions)) #torch.min(self.critic_model(state_batch))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        return loss_actor.item(), loss_critic.item(), loss_state.item()

def main():
    import argparse
    import logging, sys
    
    ### Parsing of the arguments
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-CarGoal1-v0') #CartPole-v0

    args = parser.parse_args()
    env = gym.make(args.env)

    state_size = env.observation_space.shape[0]
    logging.debug(f'State size = {state_size}')
    
    # action space dimension
    if has_continuous_action_space:
        action_size = env.action_space.shape[0]
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')

    logging.debug(f'Creating agent')
    agent = ACSAgent(state_size=state_size, action_size=action_size)
    logging.info(f"Agent created")

    writer = SummaryWriter()

    logging.debug(f'Starting training')
    steps = 0
    total_num_episodes = 0
    number_of_episodes = 0  
    for e in range(EPOCHS):
        cumulative_epoch_distance = 0
        sum_episode_length = 0
        #reset environement at the start of each epoch #TODO: Maybe better to put outside the EPOCHS cycle (we can start a new epoch in the middle of an episode)
        observation, episode_return, episode_distance, episode_length = env.reset(), 0, 0, 0
        logging.info(f"Starting epoch {e}\n\n")
        

        for t in range(STEPS_PER_EPOCH):
            if RENDER:
                env.render()
            
            # with torch.no_grad():
                # observation = torch.tensor(observation).to(agent.device)
            observation_old = observation

            # logging.debug(f"The Observation type is: {type(observation)}") 
            # with torch.no_grad():
            # actions_values = agent.actor_model(observation)
            # print(f"The actions values are: {actions_values}")
            # actions_values = actions_values.max(1)[1]
            # print(f"The actions values are: {actions_values}")
            # actions_values = actions_values.view(1,1)
            # print(f"The actions values are: {actions_values}")
            if has_continuous_action_space:
                action_mean = agent.actor_model(observation)
                action_raw = action_mean
                cov_mat = torch.diag(agent.action_var).unsqueeze(dim=0)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                # logging.debug(f"The Observation type is: {observation}")
                action_probs = agent.actor_model(observation)
                # logging.debug(f"The action probs are: {action_probs}")
                action_raw= action_probs
                dist = Categorical(action_probs)
            print(f"The action raw are: {action_raw}")
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
            # with torch.no_grad():
            distance_predicted = agent.critic_model(observation)
            # logging.debug(f"The distance predicted is: {distance_predicted}")

            # with torch.no_grad():
            predicted_new_state = agent.state_model(state_input=observation,action_input=action_raw)
            # logging.debug(f"The predicted new state is: {predicted_new_state}")

            predictions = []
            future_state = predicted_new_state
            future_action = action
            for i in range(STEPS_IN_FUTURE):
                with torch.no_grad():
                    future_action = agent.actor_model(future_state)
                    # logging.debug(f"The action {i} steps in the future taken is: {future_action}")
                    
                    future_distance_predicted = agent.critic_model(future_state)
                    # logging.debug(f"The distance {i} steps in the future predicted is: {future_distance_predicted}")
                    future_predicted_new_state = agent.state_model(state_input=future_state,action_input=future_action)
                    # logging.debug(f"The new state predicted {i} steps in the future is: {future_predicted_new_state}")
                    predictions.append(Prediction(future_action,future_distance_predicted,future_predicted_new_state))
                    future_state=future_predicted_new_state
                
            #take a step in the environment
            print("Action:",action)
            observation, _, done, info = env.step(action) #the reward information = "_" is not used, just the states
            
            observation = torch.tensor(observation, device=agent.device)
            # goal_state = torch.tensor(GOAL_STATE, device=agent.device)
            goal_state = torch.zeros(observation.shape, device=agent.device)
            
            # logging.debug(f"The goal state is: {goal_state}, and observation is: {observation}")
            #calculate the distance
            distance_to_goal = torch.linalg.norm(goal_state-observation) #to train Actor Model
            
            
            distance_from_previous_state = torch.linalg.norm(observation-observation_old) #to train Actor Model
            distance_travelled_to_goal = distance_to_goal - torch.linalg.norm(goal_state-observation_old) #to plot and evaluate performances while training of actor model
            
            distance_to_goal_with_predictions = torch.linalg.norm(predictions[0].state-observation) #to train Critic Model
            for i in range(1,len(predictions)):
                distance_to_goal_with_predictions += torch.linalg.norm(predictions[i].state - predictions[i-1].state)
            # logging.debug(f'Distance from last state predicted to goal: {agent.critic_model(predictions[-1]["state"])}')
            distance_to_goal_with_predictions += agent.critic_model(predictions[-1].state).item() #adding the predicted distance from last state predicted
            
            delta_distances = distance_to_goal_with_predictions-distance_predicted #to plot and evaluate performances while training of critic model
            # logging.debug(f"Delta distances: {delta_distances}: Distance_from_state-Distance_with_predictions: {distance_predicted} - {distance_to_goal}")
            writer.add_scalar("Training/Delta: distance with prediction of states - distance predicted in one step ", delta_distances, steps)
            
            episode_length += 1
            episode_distance += distance_from_previous_state
            
            agent.memory_buffer.push(observation, torch.tensor(observation_old, device = agent.device ), action_raw, distance_predicted, distance_to_goal_with_predictions, distance_to_goal, predictions)    
            
            #Reset the environment and save data
            if done or (t == STEPS_PER_EPOCH-1):
                cumulative_epoch_distance += episode_distance
                sum_episode_length += episode_length
                
                writer.add_scalar("Performances/Episode length", episode_length, number_of_episodes)
                writer.add_scalar("Performances/Episode distance", episode_distance, number_of_episodes)
                number_of_episodes +=1
                observation, episode_return, episode_distance, episode_length = env.reset(), 0, 0, 0
            
            if ON_POLICY: #Based on PPO
                # if ((e*STEPS_PER_EPOCH)+t)%UPDATE_FREQUENCY == 0:
                    # agent.train_on_policy() <-- TODO
                if(t>1):
                    if has_continuous_action_space:
                        # action_mean = agent.actor_model(observation)
                        
                        
                        action_var = agent.action_var.expand_as(action_raw)
                        cov_mat = torch.diag_embed(action_var).to(agent.device)
                        dist = MultivariateNormal(action, cov_mat)
                        
                        # For Single Action Environments.
                        if agent.action_size == 1:
                            action = action.reshape(-1, agent.action_size)
                            
                    else:
                        # action_probs = agent.actor_model(observation)
                        # dist = Categorical(action_probs)
                        dist = Categorical(action_raw)
                    
                    # action_logprob = dist.log_prob(action)
                    dist_entropy = dist.entropy()
                    # state_values = agent.critic(observation)
                    state_values = distance_to_goal_with_predictions

                    # match state_values tensor dimensions with rewards tensor
                    state_values = torch.squeeze(state_values)
                    
                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(action_logprob - old_action_logprob.detach())

                    # Finding Surrogate Loss
                    advantages = distance_to_goal - state_values.detach()   
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-agent.eps_clip, 1+agent.eps_clip) * advantages

                    # final loss of clipped objective PPO
                    loss_actor = -torch.min(surr1, surr2) - 0.01*dist_entropy
                    agent.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    agent.actor_optimizer.step()
                    
                    criterion = nn.MSELoss()
                    
                    loss_critic= -criterion(distance_predicted.float(), distance_to_goal.float())
                    loss_critic= loss_critic
                    agent.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    agent.critic_optimizer.step()
                    
                    loss_state = nn.MSELoss()(predicted_new_state.float(),torch.tensor(observation).float())
                    agent.state_optimizer.zero_grad()
                    loss_state.backward()
                    agent.state_optimizer.step()

                    writer.add_scalar("Loss/Loss: actor_on_policy", loss_actor.item(), steps)
                    writer.add_scalar("Loss/Loss: critic_on_policy", loss_critic.item(), steps)
                    writer.add_scalar("Loss/Loss: state_on_policy", loss_state.item(), steps)
                old_action_logprob = action_logprob
                    
            if OFF_POLICY: #Based on SAC
                if ((e*STEPS_PER_EPOCH)+t)%UPDATE_FREQUENCY == 0:
                    print("Updating networks")
                    # for j in range(STEPS_OF_TRAINING):
                    loss_actor, loss_critic, loss_state = agent.train_off_policy()
                    writer.add_scalar("Loss/Loss: actor_off_policy", loss_actor, steps)
                    writer.add_scalar("Loss/Loss: critic_off_policy", loss_critic, steps)
                    writer.add_scalar("Loss/Loss: state_off_policy", loss_state, steps)

            steps += 1

if __name__ == '__main__':
    main()

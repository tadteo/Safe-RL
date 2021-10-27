#!/usr/bin/env python3

from tkinter import SE
import gym
from collections import deque
from tensorflow import keras
RENDER = True

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
        self.memory = deque(maxlen=self.memory_size)

        #Create networks
        self.actor_model = self.build_actor_model()
        # self.critic_model = self.build_critic_model()
        # self.state_model = self.build_state_model()

    
    def build_actor_model(self, sizes=[32,64,32], activation="relu"):
    
        inputs = keras.Input(shape=(self.state_size,), batch_size=self.batch_size)
        x=inputs
        for layer_size in sizes:
            x = keras.layers.Dense(layer_size, activation=activation)(x)
        
        outputs = keras.layers.Dense(self.action_size, activation="tanh")(x)
        

        model = keras.Model(input=inputs,
                            output=outputs, 
                            name = "Actor model")
        
        return model
    
    def build_critic_model(self, sizes=[32,64,32], activation="relu"):

        inputs = keras.Input(shape=(self.state_size,), batch_size=self.batch_size)
        x=inputs
        for layer_size in sizes:
            x = keras.layers.Dense(layer_size, activation=activation)(x)
        
        outputs = keras.layers.Dense([1], activation=None)(x)
        

        model = keras.Model(input=inputs,
                            output=outputs, 
                            name = "Critic model")
        
        return model

    def build_state_model(self, action_sizes=[8,16,32], state_sizes=[32,128,64,32], sizes=[64,32], activation="relu"):
    
        state_inputs = keras.Input(shape=(self.state_size,), batch_size=self.batch_size)
        action_inputs = keras.Input(shape=(self.action_size,), batch_size=self.batch_size)
        
        #Layers for state head, state input
        state_x=state_inputs
        for layer_size in state_sizes:
            state_x = keras.layers.Dense(layer_size, activation=activation)(state_x)
        # state_x = keras.Model(input=state_inputs, outputs=state_x)

        #Layers for state head, state input
        action_x=action_inputs
        for layer_size in action_sizes:
            action_x = keras.layers.Dense(layer_size, activation=activation)(action_x)
        # action_x = keras.Model(input=action_inputs, outputs=action_x)
        
        #combining the two model for the separate input
        x = keras.layers.Concatenate()([state_x,action_x])
        for layer_size in sizes:
            x = keras.layers.Dense(layer_size, activation=activation)(x)
        
        outputs = keras.layers.Dense(self.action_size, activation="tanh")(x)
        

        model = keras.Model(input=[state_inputs,action_inputs],
                            output=outputs, 
                            name = "State model")




def main():
    import argparse
    import logging, sys
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

    


if __name__ == '__main__':
    main()

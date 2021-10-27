import sys
import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf
import pdb
import plot
import math
EPISODES = 1000 #Maximum number of episodes
goal_state =[0,0,0,0]
#DQN Agent for the Cartpole
#Q function approximation with NN, experience replay, and target network
class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size, id_nr = 0):
        self.check_solve = False    #If True, stop if you satisfy solution confition
        self.render = True        #If you want to see Cartpole learning, then change to True

        #Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.id_nr = id_nr
       # Modify here

        #Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.epsilon = 0.02 #Fixed
        self.batch_size = 32 #Fixed
        self.memory_size = 1000
        self.train_start = 64 #Fixed
        self.target_update_frequency = 1

        #Number of test states for Q value plots
        self.test_state_no = 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Initialize target network
        self.update_target_model()

    #Approximate Q function using Neural Network
    #State is the input and the Q Values are the output.
###############################################################################
###############################################################################
        #Edit the Neural Network model here
        #Tip: Consult https://keras.io/getting-started/sequential-model-guide/
    def build_model(self):
        print("The input model need to be of the shape: ", type(self.state_size))

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
###############################################################################
###############################################################################

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #Get action from model using epsilon-greedy policy
    def get_action(self, state):
###############################################################################
###############################################################################
        #Insert your e-greedy policy code here
        #Tip 1: Use the random package to generate a random action.
        #Tip 2: Use keras.model.predict() to compute Q-values from the state.
        rand_nr = random.uniform(0, 1)
        if rand_nr < self.epsilon: #select random action
            action = random.randrange(self.action_size)
        else:
            Qs = self.model.predict(state)
            action = np.argmin(Qs) #We want to find the minimum distance
        return action
###############################################################################
###############################################################################
    #Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, distance, next_state, done):
        self.memory.append((state, action, distance, next_state, done)) #Add sample to the end of the list

    #Sample <s,a,d,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start: #Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size) #Uniformly sample the memory buffer
        #Preallocate network and target network input matrices.
        # actual_state = np.zeros((batch_size, self.state_size)) #batch_size by state_size two-dimensional array (not matrix!)
        # new_state = np.zeros((batch_size, self.state_size)) #Same as above, but used for the target network
        # action, distance, done = [], [], [] #Empty arrays that will grow dynamically
        states = []
        actions = []
        distances = []
        new_states = []
        dones = []

        targets = []

        y_update = np.zeros(batch_size)
        for sample in mini_batch:
            state, action, distance, new_state, done = sample
            states.append(state)
            actions.append(action)
            distances.append(distance)
            new_states.append(new_state)
            dones.append(done)

            target = self.model.predict(state)

            if done:
                target[0][action] = distance
            else:
                Q_future = min(self.target_model.predict(new_state)[0])
                target[0][action] = distance + (Q_future-distance) ######## WHY DOES THIS WORK?
            targets.append(target)
            # self.model.fit(state, target, epochs = 1, verbose = 0)

        np_targets = np.array(targets).reshape((batch_size, self.action_size))
        np_states = np.array(states).reshape((batch_size, self.state_size))
        self.model.fit(np_states, np_targets, epochs = 1, verbose = 0)
#         for i in range(self.batch_size):
#             actual_state[i] = mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
#             action.append(mini_batch[i][1]) #Store a(i)
#             distance.append(mini_batch[i][2]) #Store d(i)
#             new_state[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
#             done.append(mini_batch[i][4])  #Store done(i)

#         q_value = self.model.predict(actual_state) #Generate target values for training the inner loop network using the network model
#         print("Q_val: ",type(q_value), q_value.shape, q_value)

#         next_q_val = self.target_model.predict(new_state) #Generate the target values for training the outer loop target network

#         #Q Learning: get minimum Q value at s' from target network
# ###############################################################################
# ###############################################################################
#         #Insert your Q-learning code here
#         #Tip 1: Observe that the Q-values are stored in the variable target
#         #Tip 2: What is the Q-value of the action taken at the last state of the episode?
#         for i in range(self.batch_size): #For every batch element
#             #target[i][action[i]] = random.randint(0,1) #WHY OVERWRITE?
#             # if done[i] == True:
#             q_value[i][action[i]] = distance[i]
#             # else:
#                 # q_value[i][action[i]] = distance[i] + self.discount_factor * np.amin(target_val[i])
#         # Train the inner loop network
#         self.model.fit(x=q_value, y=np.zeros(q_value.shape), batch_size=self.batch_size,
#                         epochs = 1, verbose = 0)
        return


###############################################################################
###############################################################################

def main():
    #For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0') #Generate Cartpole-v0 environment object from the gym library
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    #Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_size, action_size)

    #Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, state_size))
    min_q = np.zeros((EPISODES, agent.test_state_no))
    min_q_mean = np.zeros((EPISODES,1))

    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, _, done, info = env.step(action)
            distance = np.linalg.norm(goal_state[:2]-next_state[:2])
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state
            # print("distance: ",distance)

    scores, episodes = [], [] #Create dynamically growing score and episode counters
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset() #Initialize/reset the environment
        state = np.reshape(state, [1, state_size]) #Reshape state so that to a 1 by state_size two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]
        #Compute Q values for plotting
        tmp = agent.model.predict(test_states)
        min_q[e][:] = np.min(tmp, axis=1)
        min_q_mean[e] = np.min(min_q[e][:])

        while not done:
            if agent.render:
                env.render() #Show cartpole animation

            #Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, _, done, info = env.step(action)
            distance = np.linalg.norm(goal_state[:2]-next_state[:2])
            next_state = np.reshape(next_state, [1, state_size]) #Reshape next_state similarly to state

            #Save sample <s, a, d, s'> to the replay memory
            agent.append_sample(state, action, distance, next_state, done)
            #Training step
            print("Training step")
            agent.train_model()
            score += _ #Store episodic reward
            state = next_state #Propagate state
            # print("distance: ",distance)
            
            if done:
                #At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    print("Updating target")
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score," q_value:", min_q_mean[e],"  memory length:", len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e-100, "episodes")
                        agent.plot_data(episodes,scores,min_q_mean[:e+1])
                        sys.exit()
    plot.plot_data(episodes,scores,min_q_mean,"safe")


if __name__ == "__main__":
    main()
    
    

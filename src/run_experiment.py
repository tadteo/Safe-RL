#!/usr/bin/env python3

from cmath import log
import os
import logging, sys
import socket

from safety_gym.envs.engine import Engine 
import gym

from agents.acs_agent import ACSAgent
from agents.sac_agent import SACAgent
from agents.dqn_agent import DQNAgent

from utils import *
from torch.utils.tensorboard import SummaryWriter

def run_experiment(config):
    
    #SETTING UP PARAMETERS
    ENV = config.get('environment')

    RENDER = config.get('render')
    HAS_CONTINUOUS_ACTION_SPACE = config.get('has_continuous_action_space')
    EPOCHS = config.get('epochs')
    STEPS_PER_EPOCH = config.get('steps_per_epoch')
    UPDATE_FREQUENCY = config.get('update_frequency')

    AGENT_KIND = config.get('agent_kind') #'SAC' or 'ACS' or 'DQN'
    REWARD_TYPE = config.get('reward_kind') #'reward' or 'distance'
    STEPS_IN_FUTURE = config.get('steps_in_future')

    #Agent hyperparameters
    LEARNING_RATE = config.get('learning_rate') 
    EPSILON_START = config.get('epsilon_start')
    EPSILON_END = config.get('epsilon_end')
    EPSILON_DECAY = config.get('epsilon_decay')
    
    ALPHA = config.get('alpha')
    GAMMA = config.get('gamma')
    POLYAK = config.get('polyak')
    
    UPDATE_TARGET_FREQUENCY = config.get('update_target_frequency')
    
    ON_POLICY = config.get('on_policy')
    OFF_POLICY = config.get('off_policy')

    DISCOUNT_FACTOR = config.get('discount_factor')
    BATCH_SIZE = config.get('batch_size')
    EXPERIMENT_NAME = config.get('experiment_name')
    STATE_IN_MEMORY = config.get('state_in_memory')

    goal_start = config.get('goal_start')
    goal_end = config.get('goal_end')

    ACTOR_MODEL_WEIGHTS = config.get('actor_model_weights')
    CRITIC_MODEL_WEIGHTS = config.get('critic_model_weights')
    STATE_MODEL_WEIGHTS = config.get('state_model_weights')
    DQN_MODEL_WEIGHTS = config.get('dqn_model_weights')
    
    TRAINING = config.get('training')
    
    LOGGING_LVL = config.get('logging_level')
    
###############################################################################
    
    if(LOGGING_LVL == 'DEBUG'):
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    elif(LOGGING_LVL == 'INFO'):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    elif(LOGGING_LVL == 'NO_LOG'):
        logging.basicConfig(filename='/tmp/experiment_log.txt', level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    actual_path = os.path.dirname(os.path.realpath(__file__))

####Start Experiment###########################################################

    
    experiment_name = generate_exp_name(EXPERIMENT_NAME)
    logging.info(f"\n\nExperiment: {experiment_name}\n\n")
    
    #Create the environment
    logging.info(f"Creating the environment")
    logging.info(f"Environment: {ENV}, type: {type(ENV)}")
    if type(ENV) == str:
        env = gym.make(ENV)
    elif type(ENV) == dict:
        env = Engine(config=ENV)
    else:
        raise Exception("Environment type not supported")
    
    state_size = int(env.observation_space.shape[0])
    logging.debug(f'State size = {state_size}')
   
    if HAS_CONTINUOUS_ACTION_SPACE:
        action_size = int(env.action_space.shape[0])
    else:
        action_size = env.action_space.n
    logging.debug(f'Action size = {action_size}')
    state_memory_size = STATE_IN_MEMORY
        
    logging.debug(f'Creating agent')
    if AGENT_KIND == 'ACS':
        if REWARD_TYPE == 'REWARD':
            raise Exception("Reward kind not supported for ACS")
            sys.exit(1)
        agent = ACSAgent(state_size=state_size, 
                        action_size=action_size, 
                        state_memory_size=state_memory_size,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        discount_factor = DISCOUNT_FACTOR,
                        epsilon_start=EPSILON_START,
                        epsilon_end=EPSILON_END,
                        epsilon_decay=EPSILON_DECAY,
                        goal_start=goal_start, 
                        goal_end=goal_end, 
                        has_continuous_action_space=HAS_CONTINUOUS_ACTION_SPACE,
                        actor_model_weights=ACTOR_MODEL_WEIGHTS,
                        critic_model_weights=CRITIC_MODEL_WEIGHTS,
                        state_model_weights=STATE_MODEL_WEIGHTS
                        )
    elif AGENT_KIND == 'SAC':
        agent = SACAgent(state_size=state_size,
                        action_size=action_size, 
                        state_memory_size=state_memory_size,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        epsilon_start=EPSILON_START,
                        epsilon_end=EPSILON_END,
                        epsilon_decay=EPSILON_DECAY,
                        alpha=ALPHA,
                        gamma=GAMMA,
                        polyak=POLYAK,
                        discount_factor = DISCOUNT_FACTOR, 
                        goal_start=goal_start, 
                        goal_end=goal_end, 
                        has_continuous_action_space=HAS_CONTINUOUS_ACTION_SPACE,
                        reward_type=REWARD_TYPE,
                        actor_model_weights=ACTOR_MODEL_WEIGHTS,
                        critic_model_weights=CRITIC_MODEL_WEIGHTS,
                        state_model_weights=STATE_MODEL_WEIGHTS)
    elif AGENT_KIND == 'DQN':
        if HAS_CONTINUOUS_ACTION_SPACE:
            raise Exception("DQN does not support continuous action spaces")
        else:
            agent = DQNAgent(state_size=state_size,
                             action_size=action_size,
                             state_memory_size=state_memory_size,
                             batch_size=BATCH_SIZE,
                             learning_rate=LEARNING_RATE,
                             epsilon_start=EPSILON_START,
                             epsilon_end=EPSILON_END,
                             epsilon_decay=EPSILON_DECAY,
                             gamma=GAMMA,
                             update_target_frequency=UPDATE_TARGET_FREQUENCY,
                             has_continuous_action_space = HAS_CONTINUOUS_ACTION_SPACE,
                             reward_type=REWARD_TYPE,
                             dqn_model_weights=DQN_MODEL_WEIGHTS,
                             goal_start=goal_start,
                             goal_end=goal_end)
    else:
        raise Exception("Agent type not supported")
    
    logging.info(f"Agent created")

    #save models
    logging.info(f"Saving models")
    agent.save_models(experiment_name,0)
    
    writer = SummaryWriter(log_dir=os.path.join(actual_path,"../runs", experiment_name + '_' + socket.gethostname()))
    
    logging.info(f'Starting training')
    total_steps = 0
    episode = 0
    
    ### for debug
    random_action_counter = 1
    deterministic_action_counter = 1
    ###
    
    for epoch in range(EPOCHS):
        logging.info(f"Starting epoch {epoch}\n\n")
        
        state, episode_steps, episode_cumulative_distance, episode_cumulative_reward = env.reset(), 0, 0, 0
        
        agent.reset()
        agent.observe(state)

        distance = calculate_distance(state,goal_start,goal_end)
        
        for t in range(STEPS_PER_EPOCH):
            if RENDER:
                env.render()
            
            #increase counters
            total_steps += 1
            episode_steps += 1
            
            # #predict future:
            predictions = []
            
            action, action_info = agent.select_action(exploration_on=True)
            if(action_info=="random"):
                random_action_counter += 1
            elif(action_info=="deterministic"):
                deterministic_action_counter += 1
            
            # logging.debug(f"Action selected: {action}, type: {type(action)}")
            state, reward, done, info = env.step(action) # Reward not used (used just for logging)
            agent.observe(state)

            previous_distance = distance
            distance = calculate_distance(state, goal_start, goal_end)

            if episode_steps%100 == 0:
                logging.debug(f"\r{episode_steps}. The distance to goal is: {distance}")
            

            episode_cumulative_distance += distance
            episode_cumulative_reward += reward
            
            agent.update_replay_memory(action, reward, distance, done)
                    
            #Reset the environment and save data
            if distance >= 99:
                done = True
            if done or (t == STEPS_PER_EPOCH-1):    
                writer.add_scalar("Performances/Episode steps", episode_steps, total_steps)
                writer.add_scalar("Performances/Episode final distance to goal", previous_distance, total_steps) # Printing previous distance not to have random location if goal is reached
                writer.add_scalar("Performances/Episode cumulative distance", episode_cumulative_distance, total_steps)
                writer.add_scalar("Performances/Episode cumulative reward", episode_cumulative_reward, total_steps)
                
                ### for debug
                writer.add_scalar("Debug/ Random/Deterministic action counter", random_action_counter/deterministic_action_counter, total_steps)
                # writer.add_scalar("Debug/Deterministic action counter", deterministic_action_counter, total_steps)
                ###
                
                episode +=1
                observation, episode_cumulative_distance, episode_steps, episode_cumulative_reward = env.reset(), 0, 0, 0
                
                print(f"Starting episode {episode}", end="\r")
                
            if TRAINING:
                if ON_POLICY:
                    agent.train_on_policy()
                if OFF_POLICY:
                    if total_steps % UPDATE_FREQUENCY == 0:
                        loss_actor, loss_critic, loss_state, epsilon= agent.train_step()
                        if AGENT_KIND == 'SAC' or AGENT_KIND == 'ACS':
                            write_losses_log(loss_actor, loss_critic, loss_state, total_steps, writer, agent)
                        if AGENT_KIND == 'DQN':
                            write_losses_DQN(loss_critic, total_steps, writer, agent)
                        
                        write_epsilon(epsilon, total_steps, writer)
                            
        if epoch % 5 == 0 and epoch != 0:
            agent.save_models(experiment_name,epoch)
    
    logging.info(f"Training finished\n\n--------------------------------------------------------------------\n\n")
    
    writer.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model given the configuration file.\nConfiguration Files are located in config/\nExample: python run_experiment.py car_goal_acs.yaml')
    
    parser.add_argument('config', type=str, help='Path to the configuration file')
    # parser.add_argument('--render', type=bool, default=False, help='Render the environment')
    args = parser.parse_args()
    
    config = read_config(args.config)

    run_experiment(config)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import unittest
from utils import *

from run_experiment import run_experiment


class TestACS(unittest.TestCase):
    
    #test continuous action space envs
    def test_acs_car_goal_training(self):
        config = read_config('tests/test_car_goal_acs_distance_training.yaml')
        print("Start testing ACS continuous env training\n")
        run_experiment(config)
    
    def test_acs_car_goal_inference(self):
        config = read_config('tests/test_car_goal_acs_distance_inference.yaml')
        print("Start testing ACS continuous env inference\n")
        run_experiment(config)    

    #test discrete action space envs
    def test_cartpole_training(self):
        config = read_config('tests/test_cartpole_acs_distance_training.yaml')
        print("Start testing ACS discrete env training\n")
        run_experiment(config)    
    
    def test_cartpole_inference(self):
        config = read_config('tests/test_cartpole_acs_distance_inference.yaml')
        print("Start testing ACS discrete env inference\n")
        run_experiment(config)

class TestSAC(unittest.TestCase):
    def test_sac_car_goal_distance_training(self):
        config = read_config('tests/test_car_goal_sac_distance_training.yaml')
        print("Start testing SAC distance continuous env training\n")
        run_experiment(config)
            
    def test_sac_car_goal_distance_inference(self):
        config = read_config('tests/test_car_goal_sac_distance_inference.yaml')
        print("Start testing SAC distance continuous env inference\n")
        run_experiment(config)
        
    def test_sac_car_goal_reward_training(self):
        config = read_config('tests/test_car_goal_sac_reward_training.yaml')
        print("Start testing SAC reward continuous env training\n")
        run_experiment(config)
            
    def test_sac_car_goal_reward_inference(self):
        config = read_config('tests/test_car_goal_sac_reward_inference.yaml')
        print("Start testing SAC reward continuous env inference\n")
        run_experiment(config)
    
    #test discrete action space envs
    def test_sac_cartpole_distance_training(self):
        config = read_config('tests/test_cartpole_sac_distance_inference.yaml')
        print("Start testing SAC distance discrete env training\n")
        run_experiment(config)    
    
    def test_sac_cartpole_distance_inference(self):
        config = read_config('tests/test_cartpole_sac_distance_inference.yaml')
        print("Start testing SAC distance discrete env inference\n")
        run_experiment(config)
    
    def test_sac_cartpole_reward_training(self):
        config = read_config('tests/test_cartpole_sac_reward_inference.yaml')
        print("Start testing SAC reward discrete env training\n")
        run_experiment(config)
    
    def test_sac_cartpole_reward_inference(self):
        config = read_config('tests/test_cartpole_sac_reward_inference.yaml')
        print("Start testing SAC reward discrete env inference\n")
        run_experiment(config)

class TestDQN(unittest.TestCase):
    def test_dqn_cartpole_distance_training(self):
        config = read_config('tests/test_cartpole_dqn_distance_training.yaml')
        print("Start testing DQN distance discrete env training\n")
        run_experiment(config)
    
    def test_dqn_cartpole_distance_inference(self):
        config = read_config('tests/test_cartpole_dqn_distance_inference.yaml')
        print("Start testing DQN distance discrete env inference\n")
        run_experiment(config)
    
    def test_dqn_cartpole_reward_training(self):
        config = read_config('tests/test_cartpole_dqn_reward_training.yaml')
        print("Start testing DQN reward discrete env training\n")
        run_experiment(config)

    def test_dqn_cartpole_reward_inference(self):
        config = read_config('tests/test_cartpole_dqn_reward_inference.yaml')
        print("Start testing DQN reward discrete env inference\n")
        run_experiment(config)
        
if __name__ == '__main__':
    unittest.main()

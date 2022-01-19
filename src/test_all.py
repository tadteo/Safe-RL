#!/usr/bin/env python3

import unittest
from utils import *

from run_experiment import run_experiment


class TestACS(unittest.TestCase):
    
    #test continuous action space envs
    def test_acs_car_goal_training(self):
        config = read_config('tests/test_car_goal_acs_training.yaml')
        print("Start testing ACS continuous env training\n")
        run_experiment(config)
    
    def test_acs_car_goal_inference(self):
        config = read_config('tests/test_car_goal_acs_inference.yaml')
        print("Start testing ACS continuous env inference\n")
        run_experiment(config)    

    #test discrete action space envs
    def test_cartpole_training(self):
        config = read_config('tests/test_cartpole_acs_training.yaml')
        print("Start testing ACS discrete env training\n")
        run_experiment(config)    
    
    def test_cartpole_inference(self):
        config = read_config('tests/test_cartpole_acs_inference.yaml')
        print("Start testing ACS discrete env inference\n")
        run_experiment(config)

class TestSAC(unittest.TestCase):
    def test_sac_car_goal_training(self):
        config = read_config('tests/test_car_goal_sac_inference.yaml')
        print("Start testing SAC continuous env training\n")
        run_experiment(config)
            
    def test_sac_car_goal_inference(self):
        config = read_config('tests/test_car_goal_sac_inference.yaml')
        print("Start testing SAC continuous env inference\n")
        run_experiment(config)    
    
    #test discrete action space envs
    def test_sac_cartpole_training(self):
        config = read_config('tests/test_cartpole_sac_inference.yaml')
        print("Start testing SAC discrete env training\n")
        run_experiment(config)    
    
    def test_sac_cartpole_inference(self):
        config = read_config('tests/test_cartpole_sac_inference.yaml')
        print("Start testing SAC discrete env inference\n")
        run_experiment(config)

if __name__ == '__main__':
    unittest.main()

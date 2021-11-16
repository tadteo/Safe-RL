from safety_gym.envs.engine.Engine import Engine 

config = {
    'robot_base': 'xmls/car.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    # 'observe_box_lidar': True,
    'observe_hazards': True,
    # 'observe_vases': True,
    'constrain_hazards': True,
    # 'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    # 'vases_num': 4
}

env = Engine(config=config)


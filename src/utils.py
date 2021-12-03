#!/usr/bin/env python3


def generate_exp_name(exp_name):
    """
    Generate a unique experiment name.
    """
    import datetime
    
    # Get the current time
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Generate a unique experiment name
    exp_name = exp_name + '_' + now
    
    return exp_name

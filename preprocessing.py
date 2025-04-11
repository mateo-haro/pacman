import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation

def create_env(render_mode, stack_size=4):
    """Create and wrap the MsPacman environment with necessary preprocessing."""
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    
    # Apply preprocessing wrappers
    grayscale_env = GrayscaleObservation(env)
    env = FrameStackObservation(grayscale_env, stack_size=stack_size)  # Stack frames
    
    return env

def get_state(env):
    """Get the current state from the environment."""
    state = np.array(env.get_observation())
    return state 
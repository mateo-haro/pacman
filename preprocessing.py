import numpy as np
import cv2
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

def preprocess_frame(frame):
    """Convert frame to grayscale and normalize."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame.astype(np.float32) / 255.0
    return frame

def create_env():
    """Create and wrap the MsPacman environment with necessary preprocessing."""
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    
    # Apply preprocessing wrappers
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)  # Resize to 84x84
    env = FrameStack(env, num_stack=4)  # Stack 4 frames
    
    return env

def get_state(env):
    """Get the current state from the environment."""
    state = np.array(env.get_observation())
    return state 
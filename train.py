import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing import create_env, get_state
from dqn_agent import DQNAgent

def train(episodes=1000, render=False):
    # Create environment
    env = create_env()
    
    # Get state shape and number of actions
    state_shape = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    n_actions = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_shape, n_actions)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    scores = []
    for episode in tqdm(range(episodes)):
        state = get_state(env)
        done = False
        score = 0
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update state and score
            state = next_state
            score += reward
            
            if render:
                env.render()
        
        scores.append(score)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            # Save model
            agent.save(f'models/dqn_agent_{episode + 1}.pth')
            
            # Plot scores
            plt.figure(figsize=(10, 5))
            plt.plot(scores)
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig('training_progress.png')
            plt.close()
    
    env.close()
    return scores

if __name__ == "__main__":
    train() 
import gymnasium as gym
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from preprocessing import create_env, get_state
from agent import DQNAgent


def load_hyperparameters(config_path='hyperparameters.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train(episodes, agent=None, env=None):
    scores = []
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()
        total_reward = 0
        episode_over = False
        while not episode_over:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)
            agent.replay_training()
            obs = next_obs

            total_reward += reward
            episode_over = terminated or truncated

    scores.append(total_reward)

    # Print progress
    if (episode + 1) % 10 == 0:
        avg_score = np.mean(scores[-10:])
        print(f"Episode {episode + 1}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save intermediate model
        agent.save(f'models/dqn_agent_{episode + 1}.pth')
        
        # Plot scores
        plt.figure(figsize=(10, 5))
        plt.plot(scores)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig('training_progress.png')
        plt.close()

    # Save final model
    agent.save('models/final_model.pth')

    env.close()

def test_policy(model_path, agent=None, env=None):
    """Load a trained model and run it in the environment with human rendering."""
    agent.load(model_path)
    
    # Run episodes with loaded model
    obs, info = env.reset()
    total_reward = 0
    episode_over = False
    
    while not episode_over:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
        
    print(f"Episode finished with reward: {total_reward}")


def main():
    # Load hyperparameters
    config = load_hyperparameters()
    
    gym.register_envs(ale_py)

    # Create environment with config
    env = create_env(
        render_mode=config['environment']['render_mode'],
        stack_size=config['environment']['stack_size']
    )
    
    # Create agent with config
    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        memory_size=config['agent']['memory_size'],
        batch_size=config['agent']['batch_size'],
        gamma=config['agent']['gamma'],
        epsilon=config['agent']['epsilon'],
        epsilon_min=config['agent']['epsilon_min'],
        epsilon_decay=config['agent']['epsilon_decay'],
        learning_rate=config['agent']['learning_rate']
    )

    if config['training']['train_flag']:
        train(episodes=config['training']['episodes'], agent=agent, env=env)
    else:
        test_policy(model_path='models/final_model.pth', agent=agent, env=env)

if __name__ == "__main__":
    main()
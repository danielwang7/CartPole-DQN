import gymnasium as gym
import time
import torch
import random

from model import DQN
from agent import Agent
from plot import display_plot


TARGET_UPDATE = 10  # Update target network every X episodes

env = gym.make("CartPole-v1")

def test_env():
    # Initialize environment
    env = gym.make("CartPole-v1", render_mode="human")  # Use "human" mode for visualization
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load trained policy network
    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load("model/model.pth"))
    policy_net.eval()  # Set to evaluation mode (no training)

    # Run a new game using the trained model
    num_episodes = 5
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            # Select the best action (no randomness, pure exploitation)
            with torch.no_grad():
                action = torch.argmax(policy_net(state)).item()

            # Take action
            next_state, reward, done, _, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close() 


def train_with_agent():

    num_episodes = 500

    plot_scores = []
    plot_mean_scores = []

    total_score = 0
    record = 0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim)

    for episode in range(num_episodes + 1):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        done = False
        total_reward = 0

        while not done:
            
            # GET NEXT ACTION
            if random.random() < agent.epsilon: # EXPLORE
                action = env.action_space.sample()
            else: # EXPLOIT
                agent.get_action(state)

            # GET NEXT STATE
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            agent.store(state, action, reward, next_state, done)

            # PROGRESS THE STATE AND STORE REWARD
            state = next_state
            total_reward += reward

            # TRAIN NET WORK 
            agent.train()
        
        # UPDATE TARGET NETWORK EVERY FEW EPISODES
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())


        # UPDATE AGENT EPSILON
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode}, Total Reward: {total_reward}")


        # STORE INFO FOR GRAPH
        plot_scores.append(total_reward)
        total_score += total_reward
        mean_score = total_score / episode if episode != 0 else 0
        plot_mean_scores.append(mean_score)
        display_plot(plot_scores, plot_mean_scores, "training_progress.png" if episode == num_episodes else "")

    env.close()
    agent.policy_net.save()


def main():
    # train_with_agent()
    test_env()

if __name__ == "__main__":
    main()
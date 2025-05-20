import gymnasium as gym
import torch
import torch.nn.functional as F
import random
import numpy as np

from agent import Agent
from plot import training_process_plot, generate_plots
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, TimeLimit

TARGET_UPDATE = 5  # Update target network every X episodes
training_period = 250

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="recordings", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)
# env = TimeLimit(env, max_episode_steps=1000)  # default is 500


def evaluate_agent(agent, env, num_test_episodes=5):
    """
    Runs the agent in test mode (with no exploration) over a few episodes and
    returns the average test reward.
    """

    test_rewards = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_test_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        done = False
        total_reward = 0

        while not done:
            # NO RANDOM EXPLORATION
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            state = torch.tensor(next_state, dtype=torch.float32)

        test_rewards.append(total_reward)


    agent.epsilon = original_epsilon
    mean_reward = np.mean(test_rewards)

    return mean_reward

def train_with_agent():

    # ========= IMPORTANT TRAINING PARAMETERS =========
    num_episodes = 650
    test_interval = 15 # Evaluate the agent every 15 episodes

    # ==================== PLOT DATA =====================
    # Lists for training data
    train_scores = []
    train_mean_scores = []
    total_train_score = 0

    # Lists for testing data
    test_scores = []
    test_mean_scores = []
    total_test_score = 0

    # List for epsilon values
    epsilon_values = []


    # ====================================================

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
                action = agent.get_action(state)

            # GET NEXT STATE
            next_state, reward, done, _, _ = env.step(action)
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

        # UPDATE AGENT LEARNING RATE
        # agent.trainer.lr_scheduler.step(episode)

        # ===== STORE INFO FOR GRAPH =====
        train_scores.append(total_reward)
        total_train_score += total_reward
        mean_score = total_train_score / episode if episode != 0 else 0
        train_mean_scores.append(mean_score)
        training_process_plot(train_scores, train_mean_scores, "training_progress.png" if episode == num_episodes else "")

        epsilon_values.append(agent.epsilon)


        # ===== PERIODICALLY EVALUATE THE AGENT FOR TEST SCORES =====
        if episode % test_interval == 0:
            test_reward = evaluate_agent(agent, env)
            test_scores.append(test_reward)
            total_test_score += test_reward
            mean_test_score = total_test_score / len(test_scores)
            test_mean_scores.append(mean_test_score)
            print(f"Test evaluation at episode {episode}: Avg Reward = {test_reward}")

    generate_plots(train_scores, train_mean_scores, test_scores, test_mean_scores, epsilon_values)

    env.close()
    agent.policy_net.save()

def main():
    train_with_agent()

if __name__ == "__main__":
    main()
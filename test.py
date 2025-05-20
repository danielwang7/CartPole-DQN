import gymnasium as gym
import torch
import torch.nn.functional as F

from model import DQN
from gymnasium.wrappers import RecordVideo, TimeLimit

def test_env():
    env = gym.make("CartPole-v1", render_mode="human")  # Use "rbg_array" mode for recording
    # env = RecordVideo(env, video_folder="recordings", name_prefix="testing", episode_trigger=lambda x: True, video_length=2000)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load("model/model.pth"))
    policy_net.eval()  # Set to evaluation mode (no training)

    num_episodes = 1
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                q_values = policy_net(state)
                action = torch.argmax(q_values).item()

            print(f"Q-values: {q_values.numpy()}, Selected action: {action}")

            # Take action
            next_state, reward, done, _, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close() 


def main():
    test_env()

if __name__ == "__main__":
    main()
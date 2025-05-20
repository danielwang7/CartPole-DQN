import os
import torch
import gymnasium as gym

from model import DQN

# Ensure the output directory exists
output_dir = "model/"
os.makedirs(output_dir, exist_ok=True)

def main():

    # Initialize environment
    env = gym.make("CartPole-v1")  # Use "human" mode for visualization
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load("model/model.pth"))

    # Dummy state for export
    dummy_state = torch.randn(1, 4)  # Adjust the shape according to your model's input

    # Export to ONNX; this writes public/models/cartpole/model.onnx
    torch.onnx.export(
        policy_net,
        dummy_state,
        os.path.join(output_dir, "model.onnx"),
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch_size"}}
    )

    print("Model exported to ONNX format successfully.")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import os

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x
    
    def save(self, file_name = "model.pth"):
        model_folder_path = "./model"

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, policy_net, target_net, lr, gamma):
        self.gamma = gamma 
        self.lr = lr

        # NOTE: PASSING IS BY REFERENCE
        self.policy_net = policy_net
        self.target_net = target_net

        self.criterion = nn.MSELoss()

        # NOTE: SHOULD BE FOR POLICY NET, NOT TARGET
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        
        
    def train_step(self, states, actions, rewards, next_states, dones):

        # COMPUTE TARGET Q VALUES
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # COMPUTE PREDICTED Q VALUES FOR TAKEN ACTIONS
        all_q_values = self.policy_net(states)
        q_values_for_actions = all_q_values.gather(1, actions.unsqueeze(1)) # Select Q values for chosen actions
        q_values = q_values_for_actions.squeeze()

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






        






        

        
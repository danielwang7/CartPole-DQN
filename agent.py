import torch
import random
from collections import deque
from model import DQN, QTrainer

LEARNING_RATE = 0.001
MEMORY_SIZE = 10000  # Experience replay memory size
BATCH_SIZE = 64  # Mini-batch size for training

class Agent:

    def __init__(self, state_dim, action_dim):
        # self.n_games = 0 
        self.epsilon = 1
        self.min_epsilon = 0.001
        self.epsilon_decay = 0.995

        self.gamma = 0.9
        self.memory = deque(maxlen=MEMORY_SIZE)

        # INITIALIZE THE ***MODELS***
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        self.trainer = QTrainer(self.policy_net, self.target_net, LEARNING_RATE, self.gamma)

    def get_action(self, state):
        """
        Gets the predicted next best action (exploitation)
        """
        
        with torch.no_grad():
            action = torch.argmax(self.policy_net(state)).item()

        return action
    
    def store(self, state, action, reward, next_state, game_over):
        """
        Stores a SINGULAR action in memory
        """

        self.memory.append((state, action, reward, next_state, game_over))


    def train(self):
        """
        Train the policy model from the built up memory; update the target if required
        """

        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        # UNPACK THE BATCH
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        self.trainer.train_step(states, actions, rewards, next_states, dones)




        


    





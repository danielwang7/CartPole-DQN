import torch
import random
from collections import deque
from model import DQN, QTrainer
import torch.nn.functional as F

LEARNING_RATE = 0.001
MEMORY_SIZE = 50000 
BATCH_SIZE = 64
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.01
GAMMA = 0.99

class Agent:

    def __init__(self, state_dim, action_dim):
        # self.n_games = 0 
        # TODO: DETERMINE IF THE EPSILON EVEN HAS TO BE IN THE AGENT?
        self.epsilon = 1
        self.min_epsilon = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY

        self.gamma = GAMMA
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
            # action = torch.argmax(self.policy_net(state)).item()

            # TODO: UNDERSTAND THIS AND SEE IF ITS CORRECT
            q_values = self.policy_net(state)
            action_probs = F.softmax(q_values, dim=-1)  # Convert Q-values to probabilities
            action = torch.multinomial(action_probs, 1).item()  # Sample based on probability

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




        


    





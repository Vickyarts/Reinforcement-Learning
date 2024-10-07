import numpy as np 
import random
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from collections import deque 




class DNetwork(nn.Module): 
    def __init__(self, observation_space: int, action_space: int): 
        super().__init__()
        self.fc1 = nn.Linear(observation_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_space)
        
    def forward(self, x):
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.fc4(y)
        return y



class DQAgent(): 
    def __init__(self, observation_space: int, action_space: int): 
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device('cuda:0')
        self.local_net = DNetwork(observation_space, action_space).to(self.device)
        self.target_net = DNetwork(observation_space, action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=100000)
        self.t_step = 0
        self.gamma = 0.99
        
    
    def action(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_net.eval()
        with torch.no_grad():
            Qvals = self.local_net(state)
        self.local_net.train()
        if random.random() > epsilon:
            return np.argmax(Qvals.cpu().numpy())
        else: 
            act = random.choice(np.arange(self.action_space))
            return act
        
        
    def step(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        self.t_step = (self.t_step+1) % 4
        if self.t_step == 0: 
            self.learn()
            
            
    def learn(self): 
        if len(self.memory) >= 100:
            experiences = random.sample(self.memory, k=100)
            state = torch.from_numpy(np.vstack([s[0] for s in experiences])).float().to(self.device)
            action = torch.from_numpy(np.vstack([s[1] for s in experiences])).long().to(self.device)
            reward = torch.from_numpy(np.vstack([s[2] for s in experiences])).float().to(self.device)
            next_state = torch.from_numpy(np.vstack([s[3] for s in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([s[4] for s in experiences]).astype(np.uint8)).float().to(self.device)
            
            q_next = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
            q_targets = reward + self.gamma *(q_next * (1 - dones))
            q_values = self.local_net(state).gather(1, action)
            loss = F.mse_loss(q_values, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.merge_networks()
        
        
    def merge_networks(self): 
        for local, target in zip(self.local_net.parameters(), self.target_net.parameters()): 
            target.data.copy_((local.data * 0.999)+ (target.data * 0.001))

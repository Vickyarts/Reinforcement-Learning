import numpy as np
import sys
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import deque 




class ActorNetwork(nn.Module): 
    def __init__(self, observation_space: int, action_space: int): 
        super().__init__() 
        
        self.fc1 = nn.Linear(observation_space, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, action_space)
        
    
    def forward(self, x): 
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.fc4(y)
        y = torch.tanh(y) 
        return y 

    
class CriticNetwork(nn.Module): 
    def __init__(self, observation_space: int, action_space: int): 
        super().__init__() 
        
        self.fc1 = nn.Linear(observation_space + action_space, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        
    
    def forward(self, x): 
        y = self.fc1(x)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        y = self.fc4(y)
        return y 
    

def soft_merge(local, target, tau): 
    for local_param, target_param in zip(local.parameters(), target.parameters()): 
        target_param.data.copy_((local_param.data * tau) + (target_param.data * (1 - tau)))


def normalize(n_states, mean, std): 
    return (n_states - mean) / (std + 1e-8)


class ReplayBuffer(): 
    def __init__(self): 
        self.memory = deque(maxlen=100000)
        self.device = torch.device('cuda:0')
        
    def append(self, state, action, reward, next_state, done): 
        self.memory.append([state, action, reward, next_state, done])
        
    def sample(self): 
        experiences = random.sample(self.memory, k=128)
            
        states = np.vstack([exp[0] for exp in experiences])
        states = normalize(states, np.mean(states, axis=0), np.std(states, axis=0))
        states = torch.tensor(states).float().to(self.device)
        
        actions = torch.tensor(np.vstack([exp[1] for exp in experiences])).float().to(self.device)
        rewards = torch.tensor(np.vstack([exp[2] for exp in experiences])).float().to(self.device)
        
        next_states = np.vstack([exp[3] for exp in experiences])
        next_states = normalize(next_states, np.mean(next_states, axis=0), np.std(next_states, axis=0))
        next_states = torch.tensor(next_states).float().to(self.device)
        
        dones = torch.tensor(np.vstack([exp[4] for exp in experiences]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones 
        


class DDPGAgent(): 
    def __init__(self, observation_space: int, action_space: int):
        self.observation_shape = observation_space
        self.action_shape = action_space
        
        self.device = torch.device('cuda:0')
        self.Actor = ActorNetwork(observation_space, action_space).to(self.device)
        self.Critic = CriticNetwork(observation_space, action_space).to(self.device)
        self.TargetActor = ActorNetwork(observation_space, action_space).to(self.device)
        self.TargetCritic_1 = CriticNetwork(observation_space, action_space).to(self.device)       # Using Two
        self.TargetCritic_2 = CriticNetwork(observation_space, action_space).to(self.device)       # Critic Network
        self.learning_rate = 0.001
        
        self.Actor_Optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.learning_rate)
        self.Critic_Optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer()
        
        self.t_step = 0
        self.learn_action_step = 0
        self.learn_step = 0 
        self.gamma = 0.99
        self.tau = 0.001
        
        self.target_action_noise = 0.2
        
    def act(self, state, noise=0.):
        state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor(state).cpu().numpy()
        self.Actor.train()
        
        action += noise * np.random.randn(*action.shape)
        return np.clip(action,-1,1)
    
    def learn(self): 
        if len(self.buffer.memory) >= 128:
            self.Actor.train()
            self.Critic.train()
            states, actions, rewards, next_states, dones = self.buffer.sample()
            new_states = torch.cat((states, actions), 1)
            q_values = self.Critic(new_states)
            next_actions = self.TargetActor(states)
            noise = torch.clamp(torch.randn_like(next_actions)*self.target_action_noise, -0.5, 0.5).to(self.device)    # Target Policy Smoothening (Added noise to TargetActor's actions)
            next_actions = torch.clamp(next_actions+noise, min=-1, max=1)
            critic_new_states = torch.cat((states, next_actions), 1)
            q_next_values_1 = self.TargetCritic_1(critic_new_states).detach()
            q_next_values_2 = self.TargetCritic_2(critic_new_states).detach()
            
            q_next_values = torch.min(q_next_values_1, q_next_values_2)
            q_targets = rewards + (self.gamma * q_next_values * (1 - dones))
            
            loss = F.mse_loss(q_values, q_targets)
            self.Critic_Optimizer.zero_grad()
            loss.backward()
            self.Critic_Optimizer.step()
            
            if self.learn_action_step % 2 == 0:       # Actor is updated less frequent then Critic Network.
                new_actor_actions = self.Actor(states)
                new_actor_states = torch.cat((states, new_actor_actions), 1)
                actor_loss = -self.Critic(new_actor_states).mean()
                
                self.Actor_Optimizer.zero_grad()
                actor_loss.backward()
                self.Actor_Optimizer.step()
                
                soft_merge(self.Actor, self.TargetActor, self.tau)
                
            
            soft_merge(self.Critic, self.TargetCritic_1, self.tau)
            soft_merge(self.Critic, self.TargetCritic_2, self.tau)
            self.learn_action_step += 1
            
    
    def step(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)
        self.t_step += 1
        self.learn_step = (self.learn_step+1) % 4
        if self.learn_step == 0: 
            self.learn()
            
    def load_actor(self): 
        weights = torch.load('weights.pth')
        self.Actor.load_state_dict(weights)
    
    def load_checkpoint(self): 
        checkpoint = torch.load('checkpoint.pth') 
        self.Actor.load_state_dict(checkpoint['Actor'])
        self.Critic.load_state_dict(checkpoint['Critic'])
        self.TargetActor.load_state_dict(checkpoint['TargetActor'])
        self.TargetCritic_1.load_state_dict(checkpoint['TargetCritic_1'])
        self.TargetCritic_2.load_state_dict(checkpoint['TargetCritic_2'])
        self.Actor_Optimizer.load_state_dict(checkpoint['Optim_Actor'])
        self.Critic_Optimizer.load_state_dict(checkpoint['Optim_Critic'])
        
        return checkpoint['Episode'], checkpoint['Epsilon'] 
    
    def save_actor_weights(self):
        torch.save(self.Actor.state_dict(), 'weights.pth')
    
    def save_checkpoint(self, episode=1, epsilon=0.): 
        torch.save({
            'Actor': self.Actor.state_dict(), 
            'Critic': self.Critic.state_dict(), 
            'TargetActor': self.TargetActor.state_dict(), 
            'TargetCritic_1': self.TargetCritic_1.state_dict(),
            'TargetCritic_2': self.TargetCritic_2.state_dict(),
            'Optim_Actor': self.Actor_Optimizer.state_dict(), 
            'Optim_Critic': self.Critic_Optimizer.state_dict(), 
            'Epsilon': epsilon, 
            'Episode': episode
            },'checkpoint.pth')
        
        

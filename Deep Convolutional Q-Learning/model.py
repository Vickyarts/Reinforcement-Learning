import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler 
from PIL import Image 
from torchvision import transforms  
from collections import deque




class DCNetwork(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        
        self.cl1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.cl2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.cl3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.cl4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_space) 
        
    def forward(self, x):
        y = self.cl1(x)
        y = self.bn1(y)
        y = F.relu(y)
        
        y = self.cl2(y)
        y = self.bn2(y)
        y = F.relu(y)
        
        y = self.cl3(y)
        y = self.bn3(y)
        y = F.relu(y)
        
        y = self.cl4(y)
        y = self.bn4(y)
        y = F.relu(y)
        
        y = self.flatten(y) 
        
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return y
    

def preprocess_frame(frame): 
    if type(frame) == tuple: 
        frame = frame[0]
    frame = Image.fromarray(frame)
    transformer = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    return transformer(frame).unsqueeze(0)


class ReplayBuffer(): 
    def __init__(self, maxlen: int): 
        self.device = torch.device('cuda:0')
        self.memory = deque(maxlen=maxlen) 
        
    def append(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        
        
    def sample(self, k=64): 
        experiences = random.sample(self.memory, k=64)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).half().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).half().to(self.device)
        return states, actions, rewards, next_states, dones
        
    def __len__(self): 
        return len(self.memory)
        

    
    

class DCQAgent():
    def __init__(self, action_space: int): 
        super().__init__()
        self.action_space = action_space
        self.device = torch.device('cuda:0')
        self.local_net = DCNetwork(action_space).to(self.device)
        self.target_net = DCNetwork(action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=0.001)
        self.buffer = ReplayBuffer(maxlen=10000)
        self.gamma = 0.99
        self.scaler = GradScaler()    # Object used in Mixed Precision
        
    
    def action(self, state, epsilon=0.):
        state = preprocess_frame(state).float().to(self.device)
        self.local_net.eval()
        with torch.no_grad():
            Qvals = self.local_net(state)
        self.local_net.train()
        if random.random() > epsilon:
            return np.argmax(Qvals.cpu().numpy())
        else: 
            return random.choice(np.arange(self.action_space))
        
        
    def step(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)
        self.learn()
            
            
    def learn(self): 
        if len(self.buffer) >= 64:
            self.optimizer.zero_grad()
            state, action, reward, next_state, dones = self.buffer.sample(k=64)
            with autocast():     # Using Mixed Precision Training to save Memory.
                q_next = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
                q_targets = reward + self.gamma *(q_next * (1 - dones))
                q_values = self.local_net(state).gather(1, action)
                loss = F.mse_loss(q_values, q_targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
    def can_load_checkpoint(self):
        try:
            checkpoint = torch.load('checkpoint.pth')
            return True 
        except Exception:
            return False

    def load_main_net(self): 
        weights = torch.load('weights.pth')
        self.local_net.load_state_dict(weights)
    
    def load_checkpoint(self): 
        checkpoint = torch.load('checkpoint.pth') 
        self.local_net.load_state_dict(checkpoint['LocalNet'])
        self.target_net.load_state_dict(checkpoint['TargetNet'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        return checkpoint['Episode'], checkpoint['Epsilon'] 
    
    def save_main_net_weights(self):
        torch.save(self.local_net.state_dict(), 'weights.pth')
    
    def save_checkpoint(self, episode=1, epsilon=0.): 
        torch.save({
            'LocalNet': self.local_net.state_dict(), 
            'TargetNet': self.target_net.state_dict(), 
            'Optimizer': self.optimizer.state_dict(),  
            'Epsilon': epsilon, 
            'Episode': episode
            },'checkpoint.pth')
        

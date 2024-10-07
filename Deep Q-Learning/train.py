import numpy as np 
import gym 
import torch 
import torch.nn as nn
from model import DQAgent
from collections import deque


env = gym.make("LunarLander-v2")


agent = DQAgent(8, 4)

epsilon = 1.0 
epsilon_decay = 0.999
episodes = 2000 
last_scores = deque(maxlen=100)
max_steps_per_episode = 2000


for episode in range(1, episodes+1): 
    done = False
    score = 0
    state, _ = env.reset()
    for t in range(max_steps_per_episode):
        action = agent.action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action) 
        agent.step(state, action, reward, next_state, done) 
        score += reward  
        state = next_state
        if done: 
            break
        print(f'\rEpisode:{episode}/{episodes} Score:{np.mean(last_scores)}', end='')
    epsilon = max(0.1, (epsilon*epsilon_decay))
    last_scores.append(score) 
    
    if np.mean(last_scores) > 200: 
        torch.save(agent.local_net.state_dict(), 'weights.pth')
        break


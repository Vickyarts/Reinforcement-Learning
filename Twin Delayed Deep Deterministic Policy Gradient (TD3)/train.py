import numpy as np
import gym 
import random
import torch 
from collections import deque
from model import DDPGAgent





env = gym.make("BipedalWalker-v3")
observations = env.observation_space.shape[0]
actions = env.action_space.shape[0]


agent = DDPGAgent(observations, actions)


episodes = 2000
last_scores = deque(maxlen=100)
epsilon = 1.0
epsilon_decay = 0.995
minimum_epsilon = 0.1
max_reward = 0
min_reward = 1

for episode in range(1, episodes+1): 
    score = 0
    done = False 
    t = 0
    state, _ = env.reset()
    while not done: 
        action = agent.act(state, epsilon)[0]
        next_state, reward, done, *_ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        
        state = next_state 
        score += reward
        t += 1
        if reward > max_reward: 
            max_reward = reward 
        if reward < min_reward:
            min_reward = reward
        
    
    print(f'\rEpisode:{episode} T:{t} Score:{score/t} Max:{max_reward:.2f} Min:{min_reward:.2f}', end='')    
    epsilon = max(minimum_epsilon, epsilon*epsilon_decay)
    last_scores.append(score)
    if episode % 50 == 0: 
        agent.save_actor_weights()
        agent.save_checkpoint(episode, epsilon)
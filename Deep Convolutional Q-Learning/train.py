import numpy as np
import gym 
import torch 
import torch.nn as nn
from model import DCQAgent, preprocess_frame 
from collections import deque 




env = gym.make('MsPacmanDeterministic-v4')
action_space = env.action_space.n

agent = DCQAgent(action_space=action_space)



epsilon = 1.0 
epsilon_decay = 0.999
episodes = 2000 
last_scores = deque(maxlen=10)
max_steps_per_episode = 2000
starting_episode = 1

print('Training:\n')

if agent.can_load_checkpoint(): 
    starting_episode, epsilon = agent.load_checkpoint()

for episode in range(starting_episode+1, episodes+1):
    score = 0
    episode_scores = deque(maxlen=3000)
    done = False
    t = 0
    state = env.reset()
    while not done:
        action = agent.action(state, epsilon)
        next_state, reward, done, *_ = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward
        t += 1
        episode_scores.append(reward)

    last_scores.append(score)
    print(f'\rEpisode:{episode} T:{t} Avg Score:{np.mean(last_scores):.4f} Episode Reward:{np.mean(episode_scores):.4f}', end='')
    epsilon = max(0.1, epsilon*epsilon_decay)
    if np.mean(last_scores) >= 500:
        agent.save_main_net_weights()
        agent.save_checkpoint(episode, epsilon)
        print('Training Completed!')
        break

    if episode % 5 == 0:
        agent.save_main_net_weights()
        agent.save_checkpoint(episode, epsilon)
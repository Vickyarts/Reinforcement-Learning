import torch
import gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from model import DDPGAgent 



agent = DDPGAgent(24, 4)
agent.load_actor()


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)[0]
        state, reward, done, _, _ = env.step(action)

    env.close()
    imageio.mimsave('video.mp4', frames, fps=30, format='FFMPEG')

show_video_of_model(agent, 'BipedalWalker-v3')


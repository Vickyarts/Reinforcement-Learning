import torch
import gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from model import DCQAgent 



env = gym.make('MsPacmanDeterministic-v4', render_mode='rgb_array')
action_space = env.action_space.n

agent = DCQAgent(action_space)
agent.load_main_net()


def show_video_of_model():
    state, _ = env.reset()
    done = False
    frames = []
    t = 0
    while not done:
        frame = env.render()
        frames.append(frame)
        print(f'Frame: {t}')
        action = agent.action(state)
        state, reward, done, _, _ = env.step(action)
        if t > 3000:
            break
        t+=1
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30, format='FFMPEG')

show_video_of_model()

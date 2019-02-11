import copy
import glob
import os
import time
import types
from collections import deque
import csv 
import shutil

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import test_mp_envs
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot, get_reward_log

import sys 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from tensorboardX import SummaryWriter
from tqdm import tqdm 

from scipy.ndimage.filters import gaussian_filter1d as gf1d


args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes


def plot_rewards(rewards, args): 

    plt.cla()
    p = plt.plot(rewards, alpha = 0.3)
    plt.plot(gf1d(rewards, sigma = 15), c = p[0].get_color())

    plt.title('BipedalWalker-v2: Test MP')
    plt.savefig('./{}_rewards.png'.format(args.env_name))

def main():

    print('Preparing parameters')

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print('Creating envs: {}'.format(args.env_name))



    envs = test_mp_envs(args.env_name, args.num_processes)
  
    print('Creating network')
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)


    print('Initializing PPO')
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                        args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                        max_grad_norm=args.max_grad_norm)

    print('Memory')
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = []


    num_episodes = [0 for _ in range(args.num_processes)]


    last_index = 0
    
    print('Starting ! ')

    start = time.time()
    for j in tqdm(range(num_updates)):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step], rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info_num, info in enumerate(infos):
                if info_num == 0: 
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        # end_episode_to_viz(writer, info, info_num, num_episodes[info_num])
                        num_episodes[info_num] += 1
                        plot_rewards(episode_rewards, args)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        losses = agent.update(rollouts)
        rollouts.after_update()


if __name__ == "__main__":
    main()

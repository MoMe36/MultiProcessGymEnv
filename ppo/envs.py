import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box


import csv 
import sys 

sys.path.append('./../')

from vec_env import VEnv 


class TensorEnv(VEnv): 

    def __init__(self, nb_ps, env_name): 

        super().__init__(nb_ps, env_name)

    def process(self, x): 
        return torch.tensor(x).float()
    def step(self, ac): 

        ac = ac.numpy()

        ns ,r ,done, infos = super().step(ac)
        return self.process(ns), self.process(r), done, infos
    def reset(self): 
        s = super().reset()
        return self.process(s)

def test_mp_envs(env_name, nb_processes): 

    envs = TensorEnv(nb_processes, env_name)
    return envs



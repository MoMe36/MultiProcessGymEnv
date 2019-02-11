import os
import types

import numpy as np
import torch

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.running_mean_std import RunningMeanStd 
from envs import VecPyTorch, make_vec_envs

import sys 
sys.path.append('/home/mehdi/PythonBuilds/reacher/reacher_4/')
sys.path.append('/home/mehdi/Codes/Env/catcher/catcher/')
import reacher_4
import catcher 

import numpy as np 
import csv 
from arguments import get_args
from model import Policy 

def rms_from_csv(path): 

    with open(path, 'r') as file:
        reader = csv.reader(file)
        values = []

        for r in reader: 
            values.append([float(i) for i in r])

    mean = np.array(values[0])
    var = np.array(values[1])

    rms = RunningMeanStd(shape = mean.shape)
    rms.mean = mean 
    rms.var = var

    return rms


args = get_args()

env = make_vec_envs(args.env_name, args.seed, 1, None, None, args.add_timestep, device='cpu', allow_early_resets = False)


# Get a render function
render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

# We need to use the same statistics for normalization as used in training
# actor_critic = torch.load('./trained_models/{}/model'.format(args.run_id))
actor_critic = Policy(env.observation_space.shape, env.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
actor_critic.load_state_dict(torch.load('./trained_models/{}/model_state_dict'.format(args.run_id)))
ob_rms = rms_from_csv('./trained_models/{}/env.csv'.format(args.run_id))

if isinstance(env.venv, VecNormalize):
    env.venv.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    env.venv._obfilt = types.MethodType(_obfilt, env.venv)

# recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
# masks = torch.zeros(1, 1)

# if render_func is not None:
#     render_func('human')

obs = env.reset()


while True:
    with torch.no_grad():
        value, action, _ = actor_critic.act(obs, None, deterministic=True)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    # masks.fill_(0.0 if done else 1.0)

    if render_func is not None:
        render_func()

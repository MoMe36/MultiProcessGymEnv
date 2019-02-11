import numpy as np 
import multiprocessing as mp 
from multiprocessing import connection 
import gym 
from gym.core import Wrapper 

def worker_process(remote, env, seed): 


    while True: 

        cmd, data = remote.recv()
        if cmd == 'step': 
            obs, r, done, infos = env.step(data)
            if done: 
                obs = env.reset()
            remote.send((obs, r, done, infos))
            # remote.send(seed)

        elif cmd == 'reset': 
            s = env.reset()
            remote.send(s)

        elif cmd == 'close': 
            env.close()
            remote.close()
            break

        else: 
            raise NotImplementedError


class Worker: 

    def __init__(self, s, env):      

        self.child, parent = mp.Pipe()
        self.ps = mp.Process(target = worker_process, args = (parent, env, s))
        self.ps.start()


class Monitor(Wrapper): 

    def __init__(self, env): 

        super().__init__(env = env)

        self.episode_reward = 0
        self.episode_length = 0

    def step(self, ac): 

        ns, r, done, infos = super().step(ac)
        self.episode_reward += r
        self.episode_length += 1
        
        if done: 
            infos['episode'] = {'r':self.episode_reward, 'l':self.episode_length}
        else: 
            infos = {}

        return ns, r, done, infos

    def reset(self): 

        self.episode_reward = 0. 
        self.episode_length = 0.

        return super().reset()

class VEnv: 

    def __init__(self, nb_ps, env_name): 

        self.nb_ps = nb_ps
        self.workers = [Worker(i, Monitor(gym.make(env_name))) for i in range(self.nb_ps)]

        sample_env = gym.make(env_name)
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space 

    def reset(self): 

        for w in self.workers: 
            w.child.send(('reset', None))

        return np.vstack([w.child.recv() for w in self.workers])

    def close(self): 

        for w in self.workers: 
            w.child.send(('close', None))

    def step(self, action): 

        for w,a in zip(self.workers, action): 
            w.child.send(('step', a))

        obs, rewards, dones, infos = [],[],[],[]
        for i,w in enumerate(self.workers): 
            data = w.child.recv()
            obs.append(data[0])
            rewards.append(data[1])
            dones.append(data[2])
            infos.append(data[-1])
            
        return np.vstack(obs), np.array(rewards).reshape(-1,1), np.stack(dones), infos

# nb_envs = 10
# envs = VEnv(nb_envs, 'BipedalWalker-v2')

# print(envs)
# print(envs.reset())

# for _ in range(10): 
#     # result = envs.step(np.random.randint(0,2, (nb_envs)))
#     result = envs.step(np.random.uniform(-1.,1., (nb_envs, 4)))
#     print(result[1])

#     print('\n')
# envs.close()
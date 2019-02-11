# MultiProcess Gym environment 

This repo holds a simple implementation for multi-workers gym environments. Only the `vec_env.py` script is needed, but I included a working example with a PPO implementation based on OpenAI baselines.

### Usage

Designed to be as simple to use as possible: 

```python

from vec_env import VEnv

name = 'BipedalWalker-v2'
nb_processes = 10
envs = VEnv(name, nb_processes)

s = envs.reset()

print(s)

envs.close() 

```

However, most of the time, it is useful to wrap the environments into classes that eases the interaction (such as Normalizing spaces, states stacking and so on). To do so, I suggest adding one's custom classes in the script or in an external module and modifying the `__init__` method in the VEnv class in the following manner: 

```python 
# BEFORE WRAPPING 

class VEnv: 

    def __init__(self, nb_ps, env_name): 

        self.nb_ps = nb_ps
        self.workers = [Worker(i, gym.make(env_name)) for i in range(self.nb_ps)]


# WITH WRAPPING

class RandomResetWrapper(gym.Wrapper):

    def __init__(self, env): 
        super().__init__(env = env)
        self.random_counter = random.randint(self.env.max_steps)

    def step(self, ac): 

        ns, r, done, infos = super().step(ac)
        self.random_counter -= 1 
        if self.random_counter == 0: 
            self.reset()

    def reset(self): 

        self.random_counter = random.randint(self.env.max_steps)
        return super().reset()

class VEnv: 

    def __init__(self, nb_ps, env_name): 

        self.nb_ps = nb_ps
        self.workers = [Worker(i, RandomResetWrapper(gym.make(env_name))) for i in range(self.nb_ps)]


```


### Results

![BipedalWalker-perf](./BipedalWalker-v2_rewards.png)
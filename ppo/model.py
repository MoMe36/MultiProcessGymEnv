import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0])
       

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.hidden_size, num_outputs)
        else:
            raise NotImplementedError


    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class MLPBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        # super(nn.Modu, self).__init__(recurrent, num_inputs, hidden_size)

        nn.Module.__init__(self)

        self.hidden_size = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor



class Base(nn.Module): 

    def __init__(self, obs_size, ac_size, inner_size): 

        super().__init__()

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor_base = nn.Sequential(init_(nn.Linear(obs_size, inner_size)), nn.Tanh())
        self.actor_out = DiagGaussian(inner_size, ac_size)

        self.critic = nn.Sequential(init_(nn.Linear(obs_size, inner_size)), 
                                    nn.Tanh(), 
                                    init_(nn.Linear(inner_size, inner_size)), 
                                    nn.Tanh(),
                                    init_(nn.Linear(inner_size, 1))) 

    def f1(self, x): 

        value = self.critic(x)
        actor_features = self.actor_base(x)
        return value, actor_features

    def f2(self, x): 

        return self.actor_out(x)

class MultiPolicy(nn.Module):

    def __init__(self, venv, inner_size = 64): 

        super().__init__()

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        obs, ac = venv.get_spaces() 
        self.bases = nn.ModuleList() 
        for o,a in zip(obs, ac): 
            self.bases.append(Base(o,a, inner_size))

        self.core = nn.Sequential(init_(nn.Linear(inner_size, inner_size)), 
                                  nn.Tanh(), 
                                  init_(nn.Linear(inner_size, inner_size)), 
                                  nn.Tanh(),
                                  init_(nn.Linear(inner_size, inner_size)), 
                                  nn.Tanh(),
                                  init_(nn.Linear(inner_size, inner_size)), 
                                  nn.Tanh())

    def f_(self, x): 

        values = []
        features = []

        for b, x_, in zip(self.bases, x): 
            value, feature = b.f1(x_)
            values.append(value)
            features.append(feature)

        stacked_features = torch.cat(features)

        out_core = self.core(stacked_features)
        dists = [b.f2(oc.reshape(1,-1)) for b, oc in zip(self.bases, out_core)]

        return values, dists

    def f_specific(self, x, iden): 

        value, actor_features = self.bases[iden].f1(x)
        core_out = self.core(actor_features)
        dist = self.bases[iden].f2(core_out)

        return value, dist 

    def enjoy(self, x, iden): 

        _, features = self.bases[iden].f1(x)
        
        out_core = self.core(features)
        dist = self.bases[iden].f2(out_core)

        return dist.mode()

    def act(self, x, deterministic = False): 

        values, dists = self.f_(x)

        if deterministic: 
            actions = [d.mode() for d in dists]
        else: 
            actions = [d.sample() for d in dists]

        actions_log_probs = [d.log_probs(a) for a,d in zip(actions, dists)]
        dists_entropy = [d.entropy().mean() for d in dists]

        return values, actions, actions_log_probs, dists_entropy

    def evaluate_actions(self, x, actions): 

        values, dists = self.f_(x)

        action_log_probs = [d.log_probs(a) for d, a in zip(dists, actions)]
        dist_entropy = [d.entropy().mean() for d in dists]

        return values, action_log_probs, dist_entropy

    def evaluate_actions_specific(self, x, actions, iden): 

        value, dist = self.f_specific(x, iden)
        actions_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, actions_log_probs, dist_entropy


    def get_value(self, x):

        values = []
        features = []

        for b, x_ in zip(self.bases, x): 
            value, feature = b.f1(x_)
            values.append(value)

        return values

    def yield_bases(self):

        for b in self.bases: 
            yield b

    def __len__(self): 
        return len(self.bases)





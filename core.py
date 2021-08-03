import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs, return_mean=False):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, h_ppo=False): #return action from here
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        if h_ppo:
            pi, mean, std, log_std = self._distribution(obs, return_mean=True)
            logp_a = None
            if act is not None:
                logp_a = self._log_prob_from_distribution(pi, act)
            return pi, logp_a, mean, std.expand_as(mean), log_std.expand_as(mean)

        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, log_std_init=-0.5):
        super().__init__()
        print(f"Log std: {log_std_init}, std: {np.exp(log_std_init)}")
#         exit()
        log_std = log_std_init * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)) #global, still trains
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, return_mean=False):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        if return_mean:
            return Normal(mu, std), mu, std, self.log_std
        return Normal(mu, std)

    # def _get_mean_sigma(self, obs):
    #     with torch.no_grad():
    #         mu = self.mu_net(obs)
    #         std = torch.exp(self.log_std)
    #         return mu, std

    def _get_log_std_exp(self):
        return torch.clamp(self.log_std, min = np.log(0.1))


    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


# policy and value function
class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh, env_params=None, log_std_init=-0.5):
        super().__init__()

        if env_params:
            obs_dim = env_params['obs'] + env_params['goal']
        else:
            obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, log_std_init=log_std_init)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

        if env_params: #goal envs
            if isinstance(action_space, Box):
                obs_dim_qf = env_params['obs'] + env_params['goal'] + env_params['action']
            elif isinstance(action_space, Discrete):
                obs_dim_qf = env_params['obs'] + env_params['goal']
        else: # non-goal envs
            if isinstance(action_space, Box):
                obs_dim_qf = observation_space.shape[0] + action_space.shape[0]
            elif isinstance(action_space, Discrete):
                obs_dim_qf = observation_space.shape[0] + 1 #action_space.n

        #build q function
        self.qf = QFunction(obs_dim_qf, hidden_sizes=[100 for i in hidden_sizes], env_params=env_params)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs) #distribution with mu and std
            a = pi.sample() #sample action from this distribution
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            # mean, std = self.pi._get_mean_sigma(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# IPG paper: Q function 2 hidden layer 100-100 with ReLU act
class QFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(100, 100), activation=nn.ReLU, env_params=None):
        # TODO: perform check to create either continous or discrete
        super(QFunction, self).__init__()
        self.is_disc = env_params['env_type'] == 'Discrete'
        if not self.is_disc:
            self.q_func = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        elif self.is_disc:
            self.q_func = mlp([obs_dim] + list(hidden_sizes) + [env_params['action_max']], activation)

    def forward(self, obs, input2):
        #input2 can either be an action Q(s,a) or goal Q(s,g)
        obs_dim = len(obs.size())
        if self.is_disc:
            if obs_dim==1: #calls to get an action during simulations
                input_tensor = torch.cat([obs, input2])
            else: #calls during DQN loss calculation
                input_tensor = torch.cat([obs, input2], dim=1)
        else:
            input_tensor = torch.cat([obs, input2], dim=1)

        q_value = self.q_func(input_tensor)
        return torch.squeeze(q_value, -1) #q_value


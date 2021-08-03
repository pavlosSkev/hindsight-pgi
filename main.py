import math
import random
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import warnings

from gym.spaces import Box, Discrete

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from torch.optim import Adam, RMSprop
import gym
import time
import core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from on_buffer import HPPOBuffer
from off_buffer import DDPGBuffer
from samplers import her_sampler
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv
import os.path
from os import path
from tqdm import tqdm
import torch.nn.functional as F
from mpi_utils.normalizer import normalizer

class PPO():
    def __init__(self, env_fn, env_name=None, actor_critic=core.MLPActorCritic, q_func=core.QFunction, ac_kwargs=dict(), seed=0,
                num_episodes=50, steps_per_epoch=4000, epochs=50, batch_sample_size=128, gamma=0.99, clip_ratio=0.2,
                pi_lr=3e-4, vf_lr=1e-3, qf_lr=1e-3, inter_nu=0.2, train_pi_iters=80, train_v_iters=80, train_qf_iters=80,
                tau=0.001, num_cycles=1, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=10,
                exp_qf_sample_size=5, use_cv=None, cv_type=None, goal_append=False, beta=None, log_std_init=-0.5):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.batch_sample_size = batch_sample_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.qf_lr = qf_lr
        self.inter_nu = inter_nu
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.train_qf_iters = train_qf_iters
        self.tau = tau
        self.num_cycles = num_cycles
        self.lam = lam
        self.target_kl = target_kl
        self.use_cv=use_cv
        self.cv_type = cv_type
        self.goal_append = goal_append
        self.beta=beta
        self.save_freq = save_freq
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()
        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        # TODO: check locals()
#         self.seed = seed
        self.logger.save_config(locals())

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.log_std_init = log_std_init

        # Instantiate environment
        self.env = env_fn()
        self.test_env = env_fn()
        # get robotic environment params
        self.env_params = set()
        self.env_params = self.get_env_params(self.env)

        self.steps_per_epoch = self.env_params['max_timesteps']

        self.max_ep_len = self.env_params['max_timesteps']

            
        self.env_info = self.env.reset()
        obs, desired_goal = self.env_info['observation'], self.env_info['desired_goal']
        obs_dim = obs.shape
        act_dim = self.env.action_space.shape


        self.ac = actor_critic(self.env.observation_space['observation'], self.env.action_space,
                          env_params=self.env_params, log_std_init=self.log_std_init, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Sync params across processes
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # initialize sampler.
        self.her_module = her_sampler(replay_strategy='future', replay_k=8, reward_func=self.env.compute_reward)

        # Set up Parallelization parameters
        self.local_train_qf_iters = train_qf_iters #int(train_qf_iters / num_procs())
        self.local_episodes_per_epoch = int(num_episodes / num_procs())
        self.local_steps_per_epoch = self.steps_per_epoch

        print(f"steps per epoch: {self.local_steps_per_epoch}, episodes: {self.local_episodes_per_epoch}, "
              f"cycles: {self.num_cycles}, total: "
              f"{self.local_steps_per_epoch * self.local_episodes_per_epoch * self.num_cycles}")

        self.buf_on = HPPOBuffer(obs_dim, act_dim, self.env_params['goal'], self.local_steps_per_epoch * self.local_episodes_per_epoch, gamma, lam, reward_func=self.env.compute_reward)

        # initialize off policy sampler
        self.buf_off = DDPGBuffer(self.env_params, sample_func=self.her_module.sample_her_transitions, use_her=True,
                                  cat_goal_state=False)

        # normalizers
        self.o_norm = normalizer(size=self.env_params['obs'])
        self.g_norm = normalizer(size=self.env_params['goal'])

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        self.qf_optimizer = Adam(self.ac.qf.parameters(), lr=self.qf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        self.tens_counter=0 #for plotting in tensorboards (we use multiple updates per epoch, so we need more per epoch)


    def get_env_params(self, env):
        obs = env.reset()

        params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
                  'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
                  'max_timesteps': env._max_episode_steps, 'env_type': 'Box'}

        return params

    def _preproc_input(self, input_convention):
        # convert to tensors
        return torch.as_tensor(input_convention, dtype=torch.float32)

    def _preproc_input_norm(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'act': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs), self._preproc_og(g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, inp, clip_obs=200):
        inp = np.clip(inp, -clip_obs, clip_obs)
        return inp

    def compute_loss_pi(self, data, plot=False):
        # TODO: add value loss in pi loss
        obs, act, adv, logp_old, g = data['obs'], data['act'], data['adv'], data['logp'], data['g']
        obs = self._preproc_og(obs.numpy())
        g = self._preproc_og(g.numpy())
        obs = self._preproc_input(self.o_norm.normalize(obs.copy()))
        g = self._preproc_input(self.g_norm.normalize(g.copy()))

        if plot:
            learning_signals = adv * (1 - self.inter_nu)
        else:
            if self.use_cv:
                print("Using Control Variate...")
                #add if to control which cv to use
                crit_based_adv = self.get_control_variate(data)
                learning_signals = (adv - crit_based_adv) * (1 - self.inter_nu)
            else:
                learning_signals = adv * (1 - self.inter_nu)  # line 10-12 IPG pseudocode

        # Policy loss
        input = torch.cat((obs, g), dim=1)
        pi, logp = self.ac.pi(input, act)  # pi is a distribution
        ratio = torch.exp(logp - logp_old)  # same thing
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * learning_signals  # ratio, min, max
        loss_pi = -(torch.min(ratio * learning_signals, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, g, ret = data['obs'], data['g'], data['ret']
        obs = self._preproc_og(obs.numpy())
        g = self._preproc_og(g.numpy())
        obs = self._preproc_input(self.o_norm.normalize(obs.copy()))
        g = self._preproc_input(self.g_norm.normalize(g.copy()))
        input = torch.cat((obs, g), dim=1)
        return ((self.ac.v(input) - ret) ** 2).mean()

    def compute_loss_qf(self, data):
        obs, act, r, obs_next, g = data['obs'], data['act'], data['ret'], data['obs_next'], data['g']
        obs, obs_next, g = self._preproc_og(obs.numpy()), self._preproc_og(obs_next.numpy()), self._preproc_og(g.numpy())

        obs_norm = self.o_norm.normalize(obs.copy()) #numpy
        g_norm = self.g_norm.normalize(g.copy()) #numpy
        obs_next_norm = self.o_norm.normalize(obs_next.copy())

        #convert back to tensors
        obs, g, obs_next = self._preproc_input(obs_norm), self._preproc_input(g_norm), self._preproc_input(obs_next_norm)
        r = torch.reshape(r, (r.shape[0],))
        
        input = torch.cat((obs, g), dim=1) #dim: the dimension over which the tensors are concatenated
        input_next = torch.cat((obs_next, g), dim=1)
        q_value_real = self.ac.qf(input, act)  # correct
        with torch.no_grad():
            # TODO: policy target
            mean_next = self.ac.pi.mu_net(input_next)
            q_next_value = self.ac_targ.qf(input_next, mean_next)
            q_next_value = q_next_value.detach()  # detach tensor from graph
            # Bellman backup for Q function (td)
            q_value_target = r + self.gamma * q_next_value #expected
        assert q_value_real.shape == q_value_target.shape
        return (q_value_real - q_value_target).pow(2).mean()


    def compute_loss_off_pi(self, data):
        obs, g = data['obs'], data['g']
        obs, g = self._preproc_og(obs.numpy()), self._preproc_og(g.numpy())
                            
        obs_norm = self.o_norm.normalize(obs.copy()) #numpy
        g_norm = self.g_norm.normalize(g.copy()) #numpy
        #convert back to tensors
        obs, g = self._preproc_input(obs_norm), self._preproc_input(g_norm)    
                            
        input = torch.cat((obs, g), dim=1)
        mu = self.ac.pi.mu_net(input)
        off_loss = self.ac.qf(input, mu)
        return -(off_loss).mean()

    def get_expected_q(self, obs):
        #for reparametarized control trick control variate from Appendix 11.1 of IPG
        mu, std = self.ac.pi._get_mean_sigma(obs)
        actions_noise = torch.normal(mean=0, std=1, size=mu.shape) * std + mu
        return self.ac.qf(obs, actions_noise)

    def get_control_variate(self, data): 
        obs, act = data['obs'], data['act']

        if self.cv_type == 'reparam_critic_cv':
            q_value_real = self.ac.qf(obs, act)
            q_value_expected = self.get_expected_q(obs)
            return (q_value_real - q_value_expected)  #from 3.1 of interpolated policy gradients

        elif self.cv_type=='taylor_exp_cv': #not implemented correctly. Do not use, only fix.
            #second part of equation 8 in QPROP paper.
            mu0 = self.ac.pi.mu_net(obs)
            q_value_real = self.ac.qf(obs, mu0)
            q_prime = torch.autograd.grad(q_value_real.mean(), mu0, retain_graph=True)[0]
            deltas = act - mu0
            return q_value_real - (q_prime * deltas).sum()  #shape (4000,6)
        else:
            raise ValueError(f"Wrong value for parameter: cv_type, value: {self.cv_type} does not exist")

    def evaluate(self, n_traj=10):
        #Evalation
        rewards = []
        for j in range(n_traj):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.env_params['max_timesteps'])):
                obs, g = o['observation'], o['desired_goal']
                with torch.no_grad():
                    if isinstance(self.test_env.action_space, Box):
                        # self.test_env.render()
                        input_tensor = self._preproc_input_norm(obs,g)
                        a = self.ac.pi.mu_net(input_tensor)
                        o, r, d, info = self.test_env.step(a.numpy().flatten())
                        obs = o['observation']
                        is_success = info['is_success']
                        ep_ret += float(is_success)
                        if is_success:
                            break
                ep_len += 1
            rewards.append(ep_ret)
        return sum(rewards) / len(rewards)

    def _select_actions(self, pi):
        #exploration from HER paper. Instead of adding noise to the deterministic action, we use the stochastic action.
        action = pi
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions, 0.3 is random_eps
        #np binomial is 0-> 70% and 1-> 30%.
        action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)
        return action

    def update(self, epoch, average_return):
        data_on = self.buf_on.get()
        data_off = self.buf_off.sample(self.batch_sample_size)
        with torch.no_grad():
            pi_l_old, pi_info_old = self.compute_loss_pi(data_on, plot=True)
            pi_l_old = pi_l_old.item()
            v_l_old = self.compute_loss_v(data_on).item()
            qf_l_old = self.compute_loss_qf(data_off).item()
            pi_l_off_old = self.compute_loss_off_pi(data_off).item()
            inter_l_old = pi_l_old + self.inter_nu * pi_l_off_old
            eval_reward = self.evaluate()
        if proc_id()==0:
            # write with tensorbard
            writer.add_scalar("Loss/train (policy)", pi_l_old, self.tens_counter)
            writer.add_scalar("Loss/train (value)", v_l_old, self.tens_counter)
            writer.add_scalar("Loss/train (q function)", qf_l_old, self.tens_counter)
            writer.add_scalar("Loss/train (off policy)", pi_l_off_old, self.tens_counter)
            writer.add_scalar("Loss/train (inter)", inter_l_old, self.tens_counter)
            writer.add_scalar("Return/Epoch", np.array(average_return).mean(), self.tens_counter)
            writer.add_scalar("Return/Epoch-det(mean)", eval_reward, self.tens_counter)
                            
#         row = [eval_reward,self.qf_lr,self.batch_sample_size,np.exp(self.log_std_init),self.seed, proc_id(),self.env_name, self.inter_nu]
        self.tens_counter += 1
        

        # change parameters of QF
        for i in range(self.local_train_qf_iters):
            # sample batch
            data_off = self.buf_off.sample(self.batch_sample_size)
            self.qf_optimizer.zero_grad()
            loss_qf = self.compute_loss_qf(data_off)
            loss_qf.backward()
            mpi_avg_grads(self.ac.qf)  # average grads across MPI processes
            self.qf_optimizer.step()
        self.soft_update(self.ac_targ.qf, self.ac.qf)

        # freeze q network parameters.
        for param in self.ac.qf.parameters():
            param.requires_grad = False

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            data_off = self.buf_off.sample(self.batch_sample_size)  # batch_sample_size
            self.pi_optimizer.zero_grad()
            # update with previous loss only
            loss_pi, pi_info = self.compute_loss_pi(data_on)
            # choose between off policy and on policy sampling.
            assert self.beta!=None
            if self.beta == 'off_policy_sampling':
                loss_off_pi = self.compute_loss_off_pi(data_off)
            elif self.beta == 'on_policy_sampling':
                loss_off_pi = self.compute_loss_off_pi(data_on)
            # b value for multiplication with off policy loss, shown in algorithm 1 of IPG paper.
            if self.use_cv:
                b = 1
            else:
                b = self.inter_nu
            loss_pi_inter = loss_pi + b * loss_off_pi
            kl = mpi_avg(pi_info['kl'])
            # KL here is used as a constraint to avoid large updates. When complete off-policy we don't use it.
            if self.inter_nu<1.0:
                if kl > self.target_kl:
                    self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                    break
            loss_pi_inter.backward()
            mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data_on)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

        # unfreeze q net params for next iteration
        for param in self.ac.qf.parameters():
            param.requires_grad = True

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old, LossQ=qf_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - 0.95) * param.data + 0.95 * target_param.data)


    def learn(self):
        # Prepare for interaction with environment
        start_time = time.time()
        success_rate = 0

        for epoch in range(self.epochs):
            average_return = []
            successes = 0
            for _ in tqdm(range(self.num_cycles)):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for episode in range(self.local_episodes_per_epoch):
                    o, ep_ret, ep_len = self.env.reset(), 0, 0 #test
                    obs = o['observation']
                    ag = o['achieved_goal']
                    g = o['desired_goal']
                    # holders for off policy data
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    counter=0
                    for t in range(self.local_steps_per_epoch):
                        input_tensor = self._preproc_input_norm(obs, g)
                        a, v, logp = self.ac.step(input_tensor)
                        a = a.flatten()
                        #following commented command implements exploration.
                        #a = self._select_actions(a)
                        o2, r, d, info = self.env.step(a)
                        next_obs = o2['observation']
                        ag_new = o2['achieved_goal']
                        # store off-policy data
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())

                        if isinstance(a, int):
                            ep_actions.append([a])
                        else:
                            ep_actions.append(a.copy())
                        ep_ret += r
                        ep_len += 1
                        # save and log
                        self.buf_on.store(obs, a, g, ag, r, v, logp)
                        self.logger.store(VVals=v)
                        # Update obs (critical!)
                        obs = next_obs
                        ag = ag_new
                        success = int(info['is_success'])
                        if info['is_success']:
                            counter += 1
                            if counter <= 1:
                                successes += 1

                        timeout = ep_len == self.max_ep_len
                        terminal = timeout
                        epoch_ended = t == self.local_steps_per_epoch - 1
                        if terminal or epoch_ended:
                            if epoch_ended and not (terminal):
                                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)

                            # if trajectory didn't reach terminal state, bootstrap value target
                            if timeout or epoch_ended:
                                _, v, _ = self.ac.step(input_tensor)
                            else:
                                v = 0
                            self.buf_on.finish_path(v)  # computes GAE after episode is done, we want this after all the gather
                            if terminal:
                                # only save EpRet / EpLen if trajectory finished
                                average_return.append(ep_ret)
                                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                            o, ep_ret, ep_len = self.env.reset(), 0, 0

                    # store episodes
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert to numpy arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                self.buf_off.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                    self.logger.save_state({'env': self.env}, None)

                # Perform PPO update!
                self.update(epoch, average_return)
                # soft_update(qf_target, qf)

            success_rate = successes / (self.num_episodes * self.num_cycles)
            self.logger.store(SuccessRate=success_rate, EpLen=self.max_ep_len)
            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('SuccessRate', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('LossQ', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()
        if proc_id()==0:
            writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchPush-v1')

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=None)  # set 50 for robotic tasks, 4000 for locomotion
    parser.add_argument('--epochs', type=int, default=1000)  # 50 default, 200 works for robotic amnipulation
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--num_cycles', type=int, default=16) #consider 16 cycles
    parser.add_argument('--qf_batch_size', type=int, default=1024)
    parser.add_argument('--log_std', type=float, default=-0.9)
    parser.add_argument('--use_cv', default=False)
    parser.add_argument('--cv_type', default='reparam_critic_cv', help='determines which cv to use. Possible vals: '
                                                                       '"reparam_critic_cv" and "taylor_exp_cv"')
    parser.add_argument('--beta', default='off_policy_sampling', help='determines sampling for off-policy loss. '
                                                                     'Possible values: "off_policy_sampling", "on_policy_sampling"')
    parser.add_argument('--train_qf_iters', type=int, default=40)  # same as batch size of on-policy collection
    parser.add_argument('--train_pi_iters', type=int, default=40)  # default 80 for all of them
    parser.add_argument('--train_v_iters', type=int, default=40)
    #TODO: Consider more train qf iters
    parser.add_argument('--qf_lr', type=float, default=1e-3)  # same as batch size of on-policy collection
    parser.add_argument('--inter_nu', type=float, default=1.0)
    parser.add_argument('--exp_name', type=str, default='ppo3')

    args = parser.parse_args()
    create_env = lambda: gym.make(args.env)
    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if proc_id()==0:
        # ------for tensorboard------
        experiment_folder = str(datetime.now()).replace(':', '-')
        tensorboard_path = f'./tensorboard_results/{experiment_folder}-' \
                         f'{args.env}-nu-{args.inter_nu}-qfBatch-{args.qf_batch_size}-eps-{args.num_episodes}-qf' \
                         f'-{args.train_qf_iters}-cpu-{args.cpu}-qf-lr-{args.qf_lr}-std-{np.exp(args.log_std)}'
        writer = SummaryWriter(log_dir=tensorboard_path)
        # --------------------------

    agent = PPO(create_env, env_name=args.env, actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps,
            train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters, train_qf_iters=args.train_qf_iters,
            epochs=args.epochs, batch_sample_size=args.qf_batch_size, num_cycles=args.num_cycles, 
            num_episodes=args.num_episodes, qf_lr=args.qf_lr, use_cv=args.use_cv, cv_type=args.cv_type, beta=args.beta,
            inter_nu=args.inter_nu, log_std_init=args.log_std, logger_kwargs=logger_kwargs)

    agent.learn()

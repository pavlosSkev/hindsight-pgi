import threading
import numpy as np
import torch

class DDPGBuffer:
    def __init__(self, env_params, buffer_size=1000000, sample_func=None, use_her=False, cat_goal_state = True, sample_her=True):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.use_her = use_her
        self.sample_her=sample_her
        if cat_goal_state:
            self.env_params['obs'] = self.env_params['obs'] + self.env_params['goal']
        # create the buffer to store info
        if self.use_her:
            self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                            'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                            'g': np.empty([self.size, self.T, self.env_params['goal']]),
                            'act': np.empty([self.size, self.T, self.env_params['action']]),
                            }
        else:
            self.buffers = {'obs': np.empty([self.size, self.T, self.env_params['obs']]), #here was T+1
                            'act': np.empty([self.size, self.T, self.env_params['action']]),
                            'ret': np.empty([self.size, self.T, 1]),
                            'obs_next': np.empty([self.size, self.T, self.env_params['obs']]), #here added extra
                            'done': np.empty([self.size, self.T])  # newly added
                            }

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        if self.use_her:
            mb_obs, mb_ag, mb_g, mb_actions = episode_batch
            batch_size = mb_obs.shape[0]
            with self.lock:
                idxs = self._get_storage_idx(inc=batch_size)
                # store the informations
                self.buffers['obs'][idxs] = mb_obs
                self.buffers['ag'][idxs] = mb_ag
                self.buffers['g'][idxs] = mb_g
                self.buffers['act'][idxs] = mb_actions
                self.n_transitions_stored += self.T * batch_size
        else:
            mb_obs, mb_actions, mb_rewards, mb_next_obs, mb_dones = episode_batch
            batch_size = mb_obs.shape[0]
            with self.lock:
                idxs = self._get_storage_idx(inc=batch_size)
                # store the informations
                self.buffers['obs'][idxs] = mb_obs
                self.buffers['act'][idxs] = mb_actions
                self.buffers['ret'][idxs] = mb_rewards
                self.buffers['obs_next'][idxs] = mb_next_obs
                self.buffers['done'][idxs] = mb_dones
                self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        if self.use_her:
            temp_buffers = {}
            with self.lock:
                for key in self.buffers.keys():
                    temp_buffers[key] = self.buffers[key][:self.current_size]
            temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
            temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
            # sample transitions
            transitions = self.sample_func(temp_buffers, batch_size, sample_her=self.sample_her)
        else:
            temp_buffers = {}
            with self.lock:
                for key in self.buffers.keys():
                    temp_buffers[key] = self.buffers[key][:self.current_size]
            # sample transitions
            transitions = self.sample_func(temp_buffers, batch_size)
        # return transitions
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in transitions.items()}

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
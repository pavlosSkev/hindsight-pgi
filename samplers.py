import numpy as np


class her_sampler:
    def __init__(self, replay_strategy='future', replay_k=None, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, sample_her=True):
        assert sample_her!=None
        T = episode_batch['act'].shape[1] #shape of transitions: (n_trans_total_episodes, ep_len, {obs,r,a}_len)
        rollout_batch_size = episode_batch['act'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) #batch size shape
        t_samples = np.random.randint(T, size=batch_size) #max generated number T-1, therefore last obs cant be selected
        if not sample_her:
            episode_batch['ret'] = np.expand_dims(self.reward_func(episode_batch['ag_next'], episode_batch['g'], None),
                                                  axis=-1)
            transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
            return transitions
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        #T-t_samples tells us how many steps each t_sample need to reach terminal. Multiply by random nums in [0,1]
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # chooses random future timesteps. Adds 1 in order to avoid timestep 0, and also adds random future offset
        #so that randomness can be either future point up until T. Only chooses her indexes (which are 0 to )
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goal with achieved goal. Chooses some achiefed goals in future timesteps (around 219) and places
        #them at the same inexes of the original goals. So the result is 219 hindsight goals, and batch_size-219 og.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['ret'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions


    def sample_random_transitions(self, episode_batch, batch_size_in_transitions):
        # exit("inside sampler")
        T = episode_batch['act'].shape[1]
        rollout_batch_size = episode_batch['act'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)  # random indexes
        #implemented for testing goal environments with random sampling. Adds corresponding reward.
        if 'ret' not in episode_batch.keys():
            episode_batch['ret'] = np.expand_dims(self.reward_func(episode_batch['ag_next'], episode_batch['g'], None),
                                                  axis=-1)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        return transitions

    def sample_last_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['act'].shape[1]
        rollout_batch_size = episode_batch['act'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)  # random indexes
        print(episode_idxs)
        print(t_samples)
        exit('samplers line 61')
        #implemented for testing goal environments with random sampling. Adds corresponding reward.
        if 'ret' not in episode_batch.keys():
            episode_batch['ret'] = np.expand_dims(self.reward_func(episode_batch['ag_next'], episode_batch['g'], None),
                                                  axis=-1)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        return transitions

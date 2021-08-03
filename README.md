# Hindsight Policy Gradient Interpolation
This project is my Master thesis for Leiden University. I took into consideration the fact that off-policy RL methods are difficult to tune and tend to be unstable, while on-policy methods are easier to tune and more stable in order to build a hybrid HER method. My code for the hybrid RL algorithm (without hindsight) can be found [here](https://github.com/pavlosSkev/ipg_ppo). For this hindsight project, I use classes and chunks of code from [this](https://github.com/TianhongDai/hindsight-experience-replay) HER project, such as the normalization classes for the neural network input.

### Project description
I propose a hybrid RL algorithm which I call PPGI, that combines on-policy and off-policy updates. This allows us to introduce HER on the off-policy part of our algorithm, achieving hindsight combined with on-policy updates.  PPGI  interpolates  between  the  updates  of  two  state  of  the  art  RL methods,  Proximal  Policy  Optimization  (PPO)  and  Deep  Deterministic  Policy  Gradients (DDPG). My version of DDPG uses a stochastic policy instead of an exploration strategy and a deterministic policy.

### Important Hyperparameters
The hyperparameters can be found in the end of the file [main.py](https://github.com/pavlosSkev/hindsight-pgi/blob/main/main.py). The interpolation parameter is the `inter_nu`, which is used during the update as follows: `loss = inter_nu * ppo_loss + (1-inter_nu) * ddpg_loss`. Another important hyperparameter that can vastly affect the performance is the off-policy batch size named `qf_batch_size`.

### Experiment example
`python main.py --env=FetchPush-v1 --epochs=1000 --inter_nu=0.8 --qf_batch_size=1024`

### Package versions
python = '3.7.4'  
pytorch = '1.7.0+cpu'  
numpy = '1.20.2'  
gym = '0.17.3'  
mujoco = '2.0.2.13'  
tensorboard = '1.15.0'  
mpi4py = '3.0.3'

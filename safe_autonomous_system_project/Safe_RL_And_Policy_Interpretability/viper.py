import warnings

import argparse
import gym
import safety_gymnasium
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from stable_baselines3.common.base_class import BaseAlgorithm
from itertools import combinations, combinations_with_replacement
import glob
import os
import json

from tqdm import tqdm

from gym_env import make_env
#from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from evaluate import evaluate_policy
#from stable_baselines3.common.monitor import Monitor
from monitor import Monitor

from safepo.common.model import ActorVCritic
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safepo.common.wrappers import SafeNormalizeObservation
#from train.oracle import get_model_cls

# taken from config for agents trained on safepo
num_environments = 10
total_timesteps = 10000 #10000000 
n_iter = 100 #500


# setting up max leaves and depths
max_depth = None
max_leaves = None


def train_viper(agent, level, verbose):
    print(f"Training Viper on SafetyCarGoal{level}-v0")

    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(n_iter)):
        beta = 1 if i == 0 else 0
        dataset += sample_trajectory(agent,level, policy, beta)
        clf = DecisionTreeRegressor(ccp_alpha=0.0001, max_depth=max_depth,max_leaf_nodes=max_leaves)
        x = np.array([traj[0][0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2][0] for traj in dataset])

        #print(x.shape)
        #print(y.shape)
        #print(weight.shape)

        clf.fit(x, y, sample_weight=weight)

        policies.append(clf)
        policy = clf

        env =  Monitor(safety_gymnasium.make(f"SafetyCarGoal{level}-v0")) #make_env(level, num_environments)
        mean_reward, std_reward = evaluate_policy(TreeWrapper(policy), env, n_eval_episodes=100)
        if verbose == 2 or verbose == 0:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")
        rewards.append(mean_reward)

    print(f"Viper iteration complete. Dataset size: {len(dataset)}")
    best_policy = policies[np.argmax(rewards)]
    path = f'../viper_agents/{agent}.joblib' #f'viper_agents/{agent}_{n_leaves}_{max_depth}.joblib'#get path for viper #get_viper_path(args)
    print(f"Best policy:\t{np.argmax(rewards)}")
    print(f"Mean reward:\t{np.max(rewards):0.4f}")
    wrapper = TreeWrapper(best_policy)
    wrapper.print_info()
    wrapper.save(path)


def load_oracle_env(model, level): # change this -- load in environment and oracle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env =  Monitor(safety_gymnasium.make(f"SafetyCarGoal{level}-v0")) #make_env(level, num_environments)
        # modify environment for safety gym requirements
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)
        #model_cls, _ = get_model_cls(args)

        #get action space and obs space
        obs_space = env.observation_space
        act_space = env.action_space

        # get oracle
        oracle_path = glob.glob(os.path.join(f'../safepo/runs/{model}_exp/SafetyCarGoal{level}-v0/{model}/*/torch_save/model499.pt'))[0] #model_cls.load(get_oracle_path(args), env=env)
        with open( glob.glob(os.path.join(f'../safepo/runs/{model}_exp/SafetyCarGoal{level}-v0/{model}/*/config.json'))[0]) as f:
            config = json.load(f)
        
        if level == 0:
            obs_space = 40
        else:
            obs_space = obs_space.shape[0]
        oracle = ActorVCritic( # look into how this is working
                obs_dim=obs_space, #-- Change back to obs_space.shape[0] if Level 1 or 2. If level 0, 40
                act_dim=act_space.shape[0],
                hidden_sizes=config['hidden_sizes'],
            )
        
        oracle.actor.load_state_dict(torch.load(oracle_path))


        #oracle.verbose = 
        # SB will add additional wrappers to the env
        #env = oracle.env
        return env, oracle


def sample_trajectory(model,level, policy, beta):
    # We create a new environment for each viper step since
    # vectorized stable baseline environments can only be reset once
    env, oracle = load_oracle_env(model,level)
    policy = policy or oracle

    trajectory = []

    obs, _ = env.reset()
    n_steps = total_timesteps // n_iter
    while len(trajectory) < n_steps:
        active_policy = [policy, oracle][np.random.binomial(1, beta)]
        if isinstance(active_policy, DecisionTreeRegressor):
            action = active_policy.predict(obs)[0]
        else:
            with torch.no_grad():
                obs = torch.as_tensor(obs, dtype=torch.float32)
                action, _, _, _ = policy.step(obs, deterministic=True)
                action = action.detach().squeeze().cpu().numpy()
            #action, _states = active_policy.predict(obs, deterministic=True)

        if not isinstance(active_policy, DecisionTreeRegressor):
            oracle_action = action
        else:
            with torch.no_grad():
                obs = torch.as_tensor(obs, dtype=torch.float32)
                oracle_action, _, _, _ = oracle.step(obs, deterministic=True)
                oracle_action = oracle_action.detach().squeeze().cpu().numpy()
            #oracle_action = oracle.predict(obs, deterministic=True)[0]

        next_obs, reward, cost, terminated, truncated, info = env.step(action)

        state_loss = get_loss(env, oracle, obs)
        #print(obs.numpy()[0])
        #print(oracle_action)
        #trajectory += list(zip(obs.numpy(), oracle_action, state_loss))
        trajectory.append((obs.numpy(), oracle_action, state_loss))

        obs = next_obs

    return trajectory


def get_loss(env, model: BaseAlgorithm, obs):
    """
    This is the ~l loss from the paper that tries to capture
    how "critical" a state is, i.e. how much of a difference
    it makes to choose the best vs the worst action

    Instead of training the decision tree with this loss directly (which is not possible because it is not convex)
    we use it as a weight for the samples in the dataset which in expectation leads to the same result
    """
    #if isinstance(model, DQN):
        # For q-learners it is the difference between the best and worst q value
        #q_values = model.q_net(torch.from_numpy(obs)).detach().numpy()
        # q_values n_env x n_actions
        #return q_values.max(axis=1) - q_values.min(axis=1)
    #if isinstance(model, PPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        #assert isinstance(env.action_space,
                          #gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        
    # lets try discretizing the action space to deal with the continuous action space
    #possible_actions = np.arange(env.action_space.n)
    combo = list(combinations_with_replacement(np.linspace(-1,1,20),2))
    possible_actions =  [np.array(pair) for pair in combo]

    #obs = torch.from_numpy(obs)
    log_probs = []
    for action in possible_actions:
        #action = torch.from_numpy(np.array([action])).repeat(obs.shape[0])
        action = torch.from_numpy(action).repeat(obs.shape[0])
        dist = model.actor(obs)
        log_prob = dist.log_prob(action).sum(axis=-1)
        #_, log_prob, _ = model.policy.evaluate_actions(obs, action) # can get log prob from safe po
        log_probs.append(log_prob.detach().numpy().flatten())

    log_probs = np.array(log_probs).T
    #print((log_probs.max(axis=1) - log_probs.min(axis=1)).item())
    return log_probs.max(axis=1) - log_probs.min(axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent",type=str)
    parser.add_argument("-l", "--level",type=int, default=0,)
    parser.add_argument("-v", "--verbose",type=int, default=0,)

    args = parser.parse_args()
    level = args.level
    agent = args.agent
    verbose = args.verbose

    #print(agent)
    #print(level)

    train_viper(agent,level,verbose)
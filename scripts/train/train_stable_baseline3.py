#!/usr/bin/env python3

# Simple self-play PPO trainer
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import os
import gym

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.ppo import PPO
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from shutil import copyfile  # keep track of generations

from envs.carbon_trainer_env import CarbonTrainerEnv

from config.main_config import config

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
BEST_THRESHOLD = 0.5  # must achieve a mean score above this to replace prev best self

RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.

LOGDIR = "ppo_selfplay/"


def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class CarbonSelfPlayEnv(CarbonTrainerEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self, cfg: dict):
        super(CarbonSelfPlayEnv, self).__init__(cfg)
        self.policy = self
        self.best_model = None
        self.best_model_filename = None

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(100,))
        self.reward_range = (-300, 300)
        self.metadata = None

    def predict(self, obs):  # the policy
        if self.best_model is None:
            return self.action_space.sample()  # return a random action
        else:
            action, _ = self.best_model.predict(obs)
            return action

    def reset(self, players=None):
        # load model if parameters's there
        modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
        modellist.sort()
        if len(modellist) > 0:
            filename = os.path.join(LOGDIR, modellist[-1])  # the latest best model
            if filename != self.best_model_filename:
                print("loading model: ", filename)
                self.best_model_filename = filename
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=None)
        return super(CarbonSelfPlayEnv, self).reset(players)


class SelfPlayCallback(EvalCallback):
    # hacked parameters to only save new version of best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = BEST_THRESHOLD
        self.generation = 0

    def _on_step(self) -> bool:
        result = super(SelfPlayCallback, self)._on_step()
        if result and self.best_mean_reward > BEST_THRESHOLD:
            self.generation += 1
            print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
            print("SELFPLAY: new best model, bumping up generation to", self.generation)
            source_file = os.path.join(LOGDIR, "best_model.zip")
            backup_file = os.path.join(LOGDIR, "history_" + str(self.generation).zfill(8) + ".zip")
            copyfile(source_file, backup_file)
            self.best_mean_reward = BEST_THRESHOLD
        return result


def multiagent_rollout(env, policy_left, policy_right):
    """ play one agent vs the other in modified gym-style loop. """
    obs_right = env.reset()  # TODO
    obs_left = obs_right

    done = False
    total_reward = 0

    while not done:

        action_right = policy_right.predict(obs_right)
        action_left = policy_left.predict(obs_left)

        # uses a 2nd (optional) parameter for step to put in the other action
        # and returns the other observation in the 4th optional "info" param in gym's step()
        obs_right, reward, done, info = env.step(action_right, action_left)

        total_reward += reward

        if RENDER_MODE:
            env.render()

    return total_reward


def train(cfg):
    # train selfplay agent
    logger = configure(LOGDIR, ["stdout", "csv", "tensorboard"])

    carbon_env_config = cfg.carbon_game
    carbon_env_config["randomSeed"] = cfg.envs.seed
    env = CarbonSelfPlayEnv(carbon_env_config)
    # env.seed(SEED)

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)

    model = PPO(CustomActorCriticPolicy,
                env,
                verbose=1,
                tensorboard_log="./carbon_tensorboard/",
                learning_rate=0.001,
                gamma=0.9967,
                gae_lambda=0.95,
                batch_size=512,
                n_steps=300,
                )
    model.set_logger(logger)

    eval_callback = SelfPlayCallback(env,
                                     best_model_save_path=LOGDIR,
                                     log_path=LOGDIR,
                                     eval_freq=EVAL_FREQ,
                                     n_eval_episodes=EVAL_EPISODES,
                                     deterministic=False)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))  # probably never get to this point.

    env.close()


if __name__ == "__main__":
    train(config)

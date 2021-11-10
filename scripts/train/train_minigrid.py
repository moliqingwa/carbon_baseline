import gym
import os
import copy
from pathlib import Path

import numpy as np
from gym_minigrid import *

import torch
import torch.nn as nn
from easydict import EasyDict

from config.main_config import config
from scripts.runner.carbon_game_runner import CarbonGameRunner
from utils.parallel_env import ParallelEnv


class MiniGridEnv:
    def __init__(self, env_key, seed=None):
        self.env = gym.make(env_key)
        self.env.seed(seed)

    def reset(self, *args, **kwargs):
        output = self.env.reset()

        return_value = {}
        return_value['agent_id'] = ['agent-0']
        return_value['obs'] = [np.concatenate([output['image'].reshape(-1), [output['direction']]])]
        return_value['available_actions'] = [np.ones(7)]

        return return_value

    def step(self, action):
        cmd = next(iter(action.values()))
        state, reward, done, info = self.env.step(cmd)

        return_value = {}
        return_value['agent_id'] = ['agent-0']
        return_value['obs'] = [np.concatenate([state['image'].reshape(-1), [state['direction']]])]
        return_value['available_actions'] = [np.ones(7)]
        return_value['reward'] = [reward]
        return_value['env_reward'] = reward
        return_value['done'] = [done]
        return_value['info'] = [info]

        return return_value


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_dim = 1

        self.action_dim = 7

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), (1, 1)),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (3, 3), (1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (2, 2), (2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.action_header = nn.Sequential(
            nn.Linear(32 + self.dense_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size = len(x) if len(x.shape) == 2 else 1
        if len(x.shape) == 1:
            state = x[:-1].reshape(batch_size, 7,7,3).transpose(1, 3)
            dense = x[-1:].reshape(batch_size, 1)
        else:
            state = x[:, :-1].reshape(batch_size, 7,7,3).transpose(1, 3)
            dense = x[:, -1:].reshape(batch_size, 1)
        x = self.backbone(state)
        assert x.shape[0] == dense.shape[0]
        x = torch.hstack([x, dense])

        action_probs = self.action_header(x)
        return action_probs


class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_dim = 1

        self.action_dim = 7

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), (1, 1)),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (3, 3), (1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (2, 2), (2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.critic_header = nn.Sequential(
            nn.Linear(32 + self.dense_dim, 128),
            nn.LeakyReLU(),
            # nn.Linear(128, 1),
        )
        self.value_out = nn.Linear(128, 1)

    def forward(self, x):
        batch_size = len(x) if len(x.shape) == 2 else 1
        if len(x.shape) == 1:
            state = x[:-1].reshape(batch_size, 7,7,3).transpose(1, 3)
            dense = x[-1:].reshape(batch_size, 1)
        else:
            state = x[:, :-1].reshape(batch_size, 7,7,3).transpose(1, 3)
            dense = x[:, -1:].reshape(batch_size, 1)
        x = self.backbone(state)
        assert x.shape[0] == dense.shape[0]
        x = torch.hstack([x, dense])

        x = self.critic_header(x)
        x = self.value_out(x)
        return x


def main(cfg):
    cfg.envs.n_threads = 4
    cfg.runner.policy.actor_model = ActorNet()
    cfg.runner.policy.critic_model = CriticNet()

    # Load environments
    envs = []
    for i in range(cfg.envs.n_threads):
        envs.append(MiniGridEnv(env_key="MiniGrid-DoorKey-5x5-v0", seed=cfg.envs.seed + 1000 * i))
    env = ParallelEnv(envs)

    runner_config = {
        "main_config": cfg,
        "env": env,
        "run_dir": Path("./minigrid"),
    }
    runner_config = EasyDict(runner_config)

    runner = CarbonGameRunner(runner_config)
    runner.selfplay = False
    runner.run()


if __name__ == "__main__":
    main(config)


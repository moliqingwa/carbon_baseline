import copy

import torch
from easydict import EasyDict

from algorithms.model import ActorNet, CriticNet
from config.main_config import config
from envs.carbon_trainer_env import CarbonTrainerEnv
from scripts.runner.carbon_game_runner import CarbonGameRunner
from utils.parallel_env import ParallelEnv


def main(cfg):
    # Load environments
    envs = []
    for i in range(cfg.envs.n_threads):
        carbon_env_config = copy.deepcopy(cfg.carbon_game)
        carbon_env_config["randomSeed"] = cfg.envs.seed + 1000 * i
        envs.append(CarbonTrainerEnv(carbon_env_config))
    env = ParallelEnv(envs)

    runner_config = {
        "main_config": cfg,
        "env": env,
    }
    runner_config = EasyDict(runner_config)

    runner = CarbonGameRunner(runner_config)
    runner.run()

    # post process


if __name__ == "__main__":
    main(config)

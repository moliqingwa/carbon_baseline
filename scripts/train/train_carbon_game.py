import os
import copy
from pathlib import Path

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

    # Running dir
    run_dir = Path(__file__).parent / cfg.envs.experient_name
    if not run_dir.exists():
        current_run = "run1"
        run_dir = run_dir / current_run
        os.makedirs(str(run_dir))
    else:
        exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                            if str(folder.name).startswith("run")]
        if len(exist_run_nums) == 0:
            current_run = "run1"
        else:
            current_run = f"run{(max(exist_run_nums)+1)}"
        run_dir = run_dir / current_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    runner_config = {
        "main_config": cfg,
        "env": env,
        "run_dir": run_dir,
    }
    runner_config = EasyDict(runner_config)

    runner = CarbonGameRunner(runner_config)
    runner.run()

    # post process


if __name__ == "__main__":
    main(config)

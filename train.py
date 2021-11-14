import copy
from pathlib import Path

from easydict import EasyDict
import tensorboardX

from config.main_config import config
from envs.carbon_trainer_env import CarbonTrainerEnv
from carbon_game_runner import CarbonGameRunner
from utils.parallel_env import ParallelEnv
from utils.utils import create_folders_if_necessary


def main(cfg):
    # Load parallel environments
    envs = []
    for i in range(cfg.envs.n_threads):
        carbon_env_config = copy.deepcopy(cfg.carbon_game)
        carbon_env_config["randomSeed"] = cfg.envs.seed + i
        envs.append(CarbonTrainerEnv(carbon_env_config))
    env = ParallelEnv(envs)

    # Running dir
    training_from_scratch = cfg.envs.training_from_scratch
    run_dir = Path(__file__).parent / cfg.envs.experient_name
    model_path = None
    if not run_dir.exists():
        current_run = "run1"
        run_dir = run_dir / current_run
        create_folders_if_necessary(run_dir)
    else:
        exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                          if str(folder.name).startswith("run")]
        if len(exist_run_nums) == 0:
            current_run = "run1"
        elif not training_from_scratch:
            current_run = f"run{(max(exist_run_nums))}"
            model_dir = run_dir / current_run / "models"
            max_suffix = -1
            for path in model_dir.glob("*.pth"):
                suffix = path.stem[path.stem.index('_') + 1:]
                if suffix.isdigit() and max_suffix < int(suffix):
                    model_path = path
                    max_suffix = int(suffix)
            print(f"Use model: {model_path}")
        else:
            current_run = f"run{(max(exist_run_nums)+1)}"
        run_dir = run_dir / current_run
        create_folders_if_necessary(run_dir)
    print(f"Current run_dir: {run_dir}")

    # Tensorboard writer
    tb_folder = run_dir / "tensor_board"
    create_folders_if_necessary(tb_folder)
    tb_writer = tensorboardX.SummaryWriter(str(tb_folder))

    # runner
    runner_config = {
        "main_config": cfg,
        "env": env,
        "run_dir": run_dir,
        "tb_writer": tb_writer,
    }
    runner_config = EasyDict(runner_config)

    runner = CarbonGameRunner(runner_config)
    runner.restore(model_path, strict=True)
    runner.run()

    # post process


if __name__ == "__main__":
    main(config)

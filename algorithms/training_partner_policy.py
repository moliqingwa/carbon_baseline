
import copy
import glob
from pathlib import Path
from easydict import EasyDict

import random
import numpy as np
import torch

from utils.utils import to_tensor, update_linear_schedule


class TrainingPartnerPolicy:
    """
    陪练策略
    """
    def __init__(self, cfg: EasyDict, save_model_dir: Path):
        self.actor_model = copy.deepcopy(cfg.main_config.runner.policy.actor_model)
        self.critic_model = copy.deepcopy(cfg.main_config.runner.policy.critic_model)
        self.tensor_kwargs = dict(dtype=torch.float32, device=cfg.main_config.runner.device)

        self.save_dir = save_model_dir

        self.actor_model_path = None

    def policy_reset(self):
        all_paths = glob.glob(str(self.save_dir / "actor*"))
        best_paths = glob.glob(str(self.save_dir / "actor_best*"))
        paths = best_paths if random.random() < 0.8 else all_paths
        if paths:
            self.actor_model_path = random.choice(paths)

            actor_state_dict = torch.load(self.actor_model_path)
            self.actor_model.load_state_dict(actor_state_dict)

            self.actor_model.eval()
        pass

    def can_sample_trajectory(self):
        return False

    def get_actions_values(self, observation, available_actions=None):
        if self.actor_model_path is None:  # 选取随机动作
            batch_size = len(observation)
            action = np.zeros(batch_size, dtype=np.int32)
            for i, available_action in enumerate(available_actions):
                cmd_value = random.choice(list(map(lambda v: v[0],
                                                   filter(lambda v: v[1] == 1,
                                                          enumerate(available_action)))))
                action[i] = cmd_value
        else:  # 选取模型动作
            obs = to_tensor(observation).to(**self.tensor_kwargs)

            action_logits = self.actor_model(obs)
            if available_actions is not None:
                available_actions_tensor = to_tensor(available_actions).to(**self.tensor_kwargs)
                action_logits[available_actions_tensor == 0] = torch.finfo(torch.float32).min

            action_index = action_logits.sort(dim=1, descending=True)[1].numpy()  # 按照概率值倒排
            action = action_index[:, 0]  # 选取最大概率的可用动作

        return action, np.zeros_like(action), np.zeros_like(action)

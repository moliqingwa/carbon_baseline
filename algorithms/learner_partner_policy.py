from typing import Tuple
import copy
import glob
from pathlib import Path
from easydict import EasyDict

import random
import numpy as np
import torch

from algorithms.learner_policy import LearnerPolicy


class LeanerPartnerPolicy(LearnerPolicy):
    """
    陪练策略,self-play开启时使用
    """
    def __init__(self, cfg: EasyDict, save_model_dir: Path):
        super().__init__(cfg)

        self.actor_model = copy.deepcopy(self.actor_model)

        self.save_dir = save_model_dir

        self.model_path = None

    def policy_reset(self, episode: int, n_episodes: int):
        """
        选择当前时刻的最优策略
        """
        all_paths = glob.glob(str(self.save_dir / "model_*"))
        best_paths = glob.glob(str(self.save_dir / "model_best*"))
        paths = best_paths

        if paths:
            self.model_path = random.choice(paths)

            state_dict = torch.load(self.model_path)
            self.actor_model.load_state_dict(state_dict['actor'])
            self.actor_model.eval()

    def can_sample_trajectory(self):
        return False

    def get_actions(self, observation, available_actions=None):
        if self.model_path is None:  # 未加载模型,选取随机动作
            batch_size = len(observation)
            action = np.zeros(batch_size, dtype=np.int32)
            log_prob = np.zeros_like(action, dtype=np.float32)
            for i, available_action in enumerate(available_actions):
                cmd_value = random.choice(list(map(lambda v: v[0],
                                                   filter(lambda v: v[1] == 1,
                                                          enumerate(available_action)))))
                action[i] = cmd_value
                log_prob[i] = 1
        else:  # 选取模型动作
            action, log_prob = super().get_actions(observation, available_actions)
        return action, log_prob

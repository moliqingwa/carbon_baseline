from typing import Tuple, Dict, OrderedDict
import copy

import torch

from zerosum_env.envs.carbon.helpers import Board, Point

from algorithms.model import Model
from algorithms.base_policy import BasePolicy
from envs.obs_parser import ObservationParser
from utils.utils import to_tensor
from utils.categorical_masked import CategoricalMasked


class EvalPolicy(BasePolicy):
    """
    展示策略训练结果使用
    """
    def __init__(self):
        super().__init__()
        self.obs_parser = ObservationParser()
        self.previous_obs = None

        self.tensor_kwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
        self.actor_model = Model(is_actor=True)

    def restore(self, model_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict=True):
        """
        Restore models and optimizers from model_dict.

        :param model_dict: (dict) State dict of models and optimizers.
        :param strict: (bool, optional) whether to strictly enforce the keys of torch models
        """
        self.actor_model.load_state_dict(model_dict['actor'], strict=strict)
        self.actor_model.eval()

    def get_actions(self, observation, available_actions=None) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions)

        dist = CategoricalMasked(logits=action_logits, mask=available_actions)
        action = dist.argmax()
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def take_action(self, observation, configuration):
        current_obs = Board(observation, configuration)
        previous_obs = self.previous_obs if current_obs.step > 0 else None

        agent_obs_dict, dones, available_actions_dict = self.obs_parser.obs_transform(current_obs, previous_obs)
        self.previous_obs = copy.deepcopy(current_obs)

        agent_ids, agent_obs, avail_actions = zip(*[(agent_id, torch.from_numpy(obs_), available_actions_dict[agent_id])
                                                    for agent_id, obs_ in agent_obs_dict.items()])

        actions, _ = self.get_actions(agent_obs, avail_actions)
        agent_actions = {agent_id: action.item() for agent_id, action in zip(agent_ids, actions)}
        command = self.to_env_commands(agent_actions)

        return command


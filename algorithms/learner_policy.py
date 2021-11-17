from typing import Tuple, Dict, OrderedDict
from easydict import EasyDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import to_tensor, update_linear_schedule, calculate_gard_norm
from utils.replay_buffer import ReplayBuffer
from utils.categorical_masked import CategoricalMasked

from algorithms.base_policy import BasePolicy


class LearnerPolicy(BasePolicy):
    """
    Learner Policy class for training purpose.
    """
    def __init__(self, cfg: EasyDict):
        super().__init__()
        self.actor_model = cfg.main_config.runner.policy.actor_model
        self.critic_model = cfg.main_config.runner.policy.critic_model
        self.learning_rate = cfg.main_config.runner.policy.learning_rate
        self.critic_learning_rate = cfg.main_config.runner.policy.critic_learning_rate

        self.training_times = cfg.main_config.runner.policy.training_times
        self.batch_size = cfg.main_config.runner.policy.batch_size
        self.clip_epsilon = cfg.main_config.runner.policy.clip_epsilon
        self.entropy_coef = cfg.main_config.runner.policy.entropy_coef
        self.value_loss_coef = cfg.main_config.runner.policy.value_loss_coef
        self.target_kl = cfg.main_config.runner.policy.target_kl
        self.actor_max_grad_norm = cfg.main_config.runner.policy.actor_max_grad_norm
        self.critic_max_grad_norm = cfg.main_config.runner.policy.critic_max_grad_norm
        self.device = cfg.main_config.runner.device
        self.tensor_kwargs = dict(dtype=torch.float32, device=cfg.main_config.runner.device)

        self.actor_optimizer = optim.AdamW(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic_model.parameters(), lr=self.learning_rate)

        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

    def policy_reset(self, episode: int, n_episodes: int):
        self.actor_model.eval()
        self.critic_model.eval()
        self._lr_decay(episode, n_episodes)

    def can_sample_trajectory(self):
        return True

    def get_actions(self, observation, available_actions=None):
        """
        Compute actions predictions for the given inputs.
        :param observation:  (np.ndarray)local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        """
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions)

        dist = CategoricalMasked(logits=action_logits, mask=available_actions)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    def state_dict(self) -> Dict[str, OrderedDict[str, torch.Tensor]]:
        """
        Returns a whole state of models and optimizers.
        :return:
            dict:
                a dictionary containing a whole state of the module
        """
        state_dict = {
            "actor": self.actor_model.state_dict(),
            "critic": self.critic_model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        return state_dict

    def restore(self, model_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict=True):
        """
        Restore models and optimizers from model_dict.

        :param model_dict: (dict) State dict of models and optimizers.
        :param strict: (bool, optional) whether to strictly enforce the keys of torch models
        """
        self.actor_model.load_state_dict(model_dict['actor'], strict=strict)
        self.actor_optimizer.load_state_dict(model_dict['actor_optimizer'])

        self.critic_model.load_state_dict(model_dict['critic'], strict=strict)
        self.critic_optimizer.load_state_dict(model_dict['critic_optimizer'])

        self.actor_model.to(self.device)
        self.critic_model.to(self.device)

    def evaluate_actions(self, observation, action, available_actions=None):
        """
        Compute log probability and entropy of given actions.
        :param observation: observations inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to calculate.
        :param available_actions: (torch.Tensor) denotes which actions are available for agent
        :return log_prob: (torch.Tensor) log probabilities of the input actions.
        :return entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions)

        dist = CategoricalMasked(logits=action_logits, mask=available_actions)
        log_prob = dist.log_prob(action)

        return log_prob, dist.entropy().mean()

    def get_values(self, observation, to_numpy=True):
        """
        Get value function predictions.
        :param observation: Agent observations.
        :param to_numpy: predicted value converted to numpy or not (for training or evaluating).

        :return values: (torch.Tensor) value function predictions.
        """
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        value = self.critic_model(obs)
        if to_numpy:
            value = value.detach().cpu().numpy().flatten()
        return value

    def train(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """
        train actor and critic models using PPO Algorithms.

        :param buffer: (ReplayBuffer) the replay buffer which provide data for training.
        :return train_logs: (Dict[str, float]) the statistical log of the training.
        """
        self.actor_model.train()
        self.critic_model.train()

        train_logs = []
        for _ in range(self.training_times):
            batch_size = self.batch_size
            for indices in buffer.get_batches_starting_indexes(batch_size):
                train_log = self._train(buffer.sample_batch_by_indices(indices))
                if not train_log:
                    continue
                train_logs.append(train_log)
        train_logs = {key: np.mean([d[key] for d in train_logs]) for key in train_logs[0]}

        return train_logs

    def _train(self, batch: EasyDict) -> Dict[str, float]:
        """
        train actor and critic models once using batch samples.

        :param batch: (EasyDict) batch data for training
        :return output: (Dict[str, float]) the statistical log of the training.
        """
        log_prob, dist_entropy = self.evaluate_actions(batch.obs, batch.action, batch.available_actions)
        value = self.get_values(batch.obs, to_numpy=False)

        # actor loss
        ratio = torch.exp(log_prob - batch.log_prob)
        advantage = batch.advantage
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        actor_loss = policy_loss - self.entropy_coef * dist_entropy

        # value loss
        experience_value = batch.value
        experience_return = batch.return_
        value = value.reshape_as(batch.value)
        value_clipped = experience_value + torch.clamp(value - experience_value, -self.clip_epsilon, self.clip_epsilon)
        value_loss = F.mse_loss(value, experience_return)
        value_clipped_loss = F.mse_loss(value_clipped, experience_return)
        value_loss = torch.max(value_loss, value_clipped_loss).mean()
        critic_loss = self.value_loss_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = log_prob - batch.log_prob
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                print(f"approx_kl = {approx_kl.item()}, skip this training!")
                return {}

        # train
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = calculate_gard_norm(self.actor_model.parameters())
        if self.actor_max_grad_norm is not None and self.actor_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.actor_max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = calculate_gard_norm(self.critic_model.parameters())
        if self.critic_max_grad_norm is not None and self.critic_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), self.critic_max_grad_norm)
        self.critic_optimizer.step()

        # 返回统计情况
        output = EasyDict({
            "entropy": dist_entropy.mean().item(),
            "policy_loss": policy_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm,
            "advantage": advantage.mean().item(),
            "approx_kl": approx_kl.item(),
            "value": value.mean().item(),
            "critic_loss": critic_loss.item(),
            "critic_grad_norm": critic_grad_norm,
            "ratio": ratio.mean().item(),
        })
        return output

    def _lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.learning_rate)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_learning_rate)

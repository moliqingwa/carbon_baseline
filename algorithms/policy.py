
from easydict import EasyDict

import torch
import torch.optim as optim
from torch.distributions import Categorical


from utils.utils import to_tensor, update_linear_schedule


class Policy:
    def __init__(self, cfg: EasyDict):
        self.actor_model = cfg.main_config.runner.policy.actor_model
        self.critic_model = cfg.main_config.runner.policy.critic_model
        self.learning_rate = cfg.main_config.runner.policy.learning_rate
        self.critic_learning_rate = cfg.main_config.runner.policy.critic_learning_rate
        self.tensor_kwargs = dict(dtype=torch.float32, device=cfg.main_config.runner.device)

        self.actor_optimizer = optim.AdamW(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic_model.parameters(), lr=self.learning_rate)

    def policy_reset(self):
        pass

    def can_sample_trajectory(self):
        return True

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.learning_rate)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_learning_rate)

    def get_actions_values(self, observation, available_actions=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param observation:  (np.ndarray)local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return values: (torch.Tensor) value function predictions.
        """
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions).to(**self.tensor_kwargs)
            action_logits[available_actions == 0] = torch.finfo(torch.float32).min

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic_model(obs)

        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy()
        value = value.detach().cpu().numpy().flatten()
        return action, log_prob, value

    def evaluate_actions(self, observation, action, available_actions=None):
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions).to(**self.tensor_kwargs)
            action_logits[available_actions == 0] = torch.finfo(torch.float32).min

        dist = Categorical(logits=action_logits)
        # action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic_model(obs)

        return log_prob, dist.entropy().mean(), value

    def get_values(self, observation):
        """
        Get value function predictions.
        :param observation: Agent observations.

        :return values: (torch.Tensor) value function predictions.
        """
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        value = self.critic_model(obs)
        value = value.detach().cpu().numpy().flatten()
        return value

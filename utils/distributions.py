# -*- encoding: utf-8 -*-
from abc import ABC, abstractmethod

from typing import List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical
from torch.distributions.utils import logits_to_probs


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        parameters will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in torch.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return torch.stack([dist.sample(sample_shape) for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, parameters is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
            self,
            probs: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            validate_args: Optional[bool] = None,
            masks: Optional[np.ndarray] = None,
    ):
        self.masks: Optional[torch.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, parameters is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = torch.as_tensor(masks, dtype=torch.bool, device=device).reshape(self.logits.shape)
            HUGE_NEG = torch.tensor(-1e8, dtype=self.logits.dtype, device=device)

            logits = torch.where(self.masks, self._original_logits, HUGE_NEG)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # self.probs may already be cached, so we must force an update
        self.probs = logits_to_probs(self.logits)

    def entropy(self) -> torch.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class MaskableDistribution(Distribution, ABC):
    @abstractmethod
    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, parameters is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """


class MaskableCategoricalDistribution(MaskableDistribution):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: Optional[MaskableCategorical] = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        parameters will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: torch.Tensor) -> "MaskableCategoricalDistribution":
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return torch.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        self.distribution.apply_masking(masks)

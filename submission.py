
import base64
import pickle
import torch
import torch.nn as nn


def init_(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.)
    return module


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_action_dim = 3
        self.planter_action_dim = 5
        self.n_planters = 1
        self.collector_action_dim = 5

        self.dense_dim = 8
        self.action_dim = 5

        gain = nn.init.calculate_gain('leaky_relu', 0.01)

        self.backbone = nn.Sequential(
            init_(nn.Conv2d(13, 64, kernel_size=(3, 3), stride=(1, 1)), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            init_(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), groups=64), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            init_(nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2)), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        self.action_header = nn.Sequential(
            nn.LayerNorm(1600 + self.dense_dim),
            init_(nn.Linear(1600 + self.dense_dim, 256), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(256),
            init_(nn.Linear(256, 64), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),
            init_(nn.Linear(64, self.action_dim), gain=gain),
            # nn.Softmax(dim=-1),
        )

    def forward(self, x):
        dense = x[:, :self.dense_dim]  # TODO
        state = x[:, self.dense_dim:].reshape((-1, 13, 15, 15))
        x = self.backbone(state)
        # assert x.shape[0] == dense.shape[0]
        x = torch.hstack([x, dense])

        action_probs = self.action_header(x)
        return action_probs


actor_model =ActorNet()


def agent(obs, configuration):
    obs_tensor = obs
    action_logits = actor_model(obs_tensor)
    action
    return action

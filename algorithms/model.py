# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from utils.utils import init_


class Model(nn.Module):
    def __init__(self, is_actor=True):
        super().__init__()
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

        self.header = nn.Sequential(
            nn.LayerNorm(1600 + self.dense_dim),
            init_(nn.Linear(1600 + self.dense_dim, 256), gain=gain),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            init_(nn.Linear(256, 128), gain=gain),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(128),
        )
        if is_actor:
            self.out = init_(nn.Linear(128, self.action_dim))
        else:
            self.out = init_(nn.Linear(128, 1))

    def forward(self, x):
        dense = x[:, :self.dense_dim]
        state = x[:, self.dense_dim:].reshape((-1, 13, 15, 15))
        x = self.backbone(state)
        x = torch.hstack([x, dense])

        x = self.header(x)
        output = self.out(x)
        return output


from easydict import EasyDict

import torch

from algorithms.model import ActorNet, CriticNet

config = dict(
    carbon_game=dict(  # 游戏环境配置项

    ),
    envs=dict(  # 训练Env
        experient_name='runs',
        n_threads=32,
        seed=1,
    ),
    runner=dict(  # 训练配置项
        episodes=10000,
        episode_length=300,
        training_times=15,
        replay_buffer=dict(size=10000,
                           deepcopy=False,
                           ),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        policy=dict(
            actor_model=ActorNet(),
            critic_model=CriticNet(),  # TODO
            learning_rate=0.001,
            critic_learning_rate=0.001,
        ),
        gamma=0.9967,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        actor_max_grad_norm=0.5,
        critic_max_grad_norm=0.5,
    ),
    eval_envs=dict(  # 评估Env

    ),
)
config = EasyDict(config)

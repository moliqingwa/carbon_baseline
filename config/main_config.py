from easydict import EasyDict

import torch

from algorithms.model import Model

config = dict(
    carbon_game=dict(  # 游戏环境配置项

    ),
    envs=dict(  # 训练Env
        experient_name='runs',
        n_threads=4,  # 并行进程数
        seed=1,
        training_from_scratch=False,
    ),
    runner=dict(  # 训练配置项
        episodes=1000,
        episode_length=300,

        save_interval=10,  # 多少个epoch保存下模型
        buffer_size=20000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        policy=dict(  # 训练策略参数
            actor_model=Model(),
            critic_model=Model(is_actor=False),
            learning_rate=0.001,
            critic_learning_rate=0.001,
            training_times=15,
            batch_size=512,
            clip_epsilon=0.2,
            entropy_coef=0.018,
            value_loss_coef=1.0,
            target_kl=None,
            actor_max_grad_norm=5,
            critic_max_grad_norm=5,
        ),
        gamma=0.995,
        use_gae=False,
        gae_lambda=0.95,

        # self-play parameters
        selfplay=False,
    ),
)
config = EasyDict(config)

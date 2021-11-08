import random

import pytest
import numpy as np
import torch

from envs.carbon_trainer_env import CarbonTrainerEnv
from algorithms.model import ActorNet


class TestCarbonTrainerEnv:
    def setup_class(self):
        env_cfg = {"randomSeed": 123}
        self.env = CarbonTrainerEnv(env_cfg)
        assert self.env.configuration.randomSeed == 123

    def teardown_class(self):
        self.env.close()

    def test_env_random_action(self):
        env_output = self.env.reset([None, "random"])
        for i in range(300):
            actions = {}
            for agent_id, obs, available_actions in zip(env_output.agent_id,
                                                        env_output.obs,
                                                        env_output.available_actions):
                cmd_value = random.choice(list(map(lambda v: v[0],
                                                   filter(lambda v: v[1] == 1,
                                                          enumerate(available_actions)))))
                actions[agent_id] = cmd_value

            env_output = self.env.step(actions)

            print(f"===== STEP {i} =====")
            for agent_id, obs, reward, done, info, available_actions in zip(env_output.agent_id,
                                                                            env_output.obs,
                                                                            env_output.reward,
                                                                            env_output.done,
                                                                            env_output.info,
                                                                            env_output.available_actions):
                print(f"agent id: {agent_id}, "
                      f"observation: {obs.shape}, "
                      f"available_actions: {available_actions}, ",
                      f"reward: {reward:.2f}, "
                      f"done: {done}, "
                      f"info: {info}")

            if all(env_output.done):
                break

    def test_env_actor_action(self):
        actor_net = ActorNet()
        env_output = self.env.reset([None, "random"])
        for i in range(300):
            obs = torch.from_numpy(np.stack(env_output.obs))
            actor_logits = actor_net(obs)
            actions = {}
            for agent_id, actor_logit, available_actions in zip(env_output.agent_id,
                                                                actor_logits,
                                                                env_output.available_actions):
                action_index = actor_logit.sort(descending=True)[1].numpy()  # 按照概率值倒排
                available_action_index = set([i for i, available in enumerate(available_actions)
                                              if available == 1])  # 过滤不可用的动作索引
                cmd_value = [i for i in action_index if i in available_action_index][0]  # 选取最大概率的可用动作
                actions[agent_id] = cmd_value

            env_output = self.env.step(actions)

            print(f"===== STEP {i} =====")
            for agent_id, obs, reward, done, info, available_actions in zip(env_output.agent_id,
                                                                            env_output.obs,
                                                                            env_output.reward,
                                                                            env_output.done,
                                                                            env_output.info,
                                                                            env_output.available_actions):
                print(f"agent id: {agent_id}, "
                      f"observation: {obs.shape}, "
                      f"available_actions: {available_actions}, ",
                      f"reward: {reward:.2f}, "
                      f"done: {done}, "
                      f"info: {info}")

            if all(env_output.done):
                break

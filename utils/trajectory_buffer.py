from typing import Dict, AnyStr, Any, List
from collections import defaultdict
import copy

import numpy as np


class TrajectoryBuffer:
    """
    收集玩家多agent多环境的轨迹数据.
    """
    def __init__(self):
        self.policy_data = defaultdict(dict)  # policy_id -> env_id -> agent_id -> key -> [step1, step2, ...]

    def get_transitions(self, policy_id: int, env_id: int) -> Dict[AnyStr, Dict[AnyStr, List[Any]]]:
        """
        返回玩家的transition数据,并清除

        :param policy_id: 玩家ID
        :param env_id: 游戏环境ID
        :return: 每个agent对应的transition数据: S(t), a(t), logit(t), V(t), r(t), done(t+1)
        """
        raw_policy_data = self.policy_data[policy_id].pop(env_id)  # 单个玩家某个并行环境下的数据
        # 将每个玩家的每个游戏环境下, 单个agent的数据 (过滤掉出生即死亡的agent, 仅有策略数据,没有环境数据)
        policy_data = {agent_id: {key: np.array(value[: len(agent_data['done'])], dtype=np.float32)
                                  for key, value in agent_data.items()  # 每个key对应的value,转成numpy.ndarray类型
                                  }
                       for agent_id, agent_data in raw_policy_data.items() if 'done' in agent_data  # 每个agent
                       }

        for agent_id, agent_data in policy_data.items():  # 校验agent done是否结束
            assert agent_data['done'][-1] == 1

        return policy_data

    def add_policy_data(self, policy_id: int, policy_data):
        """
        添加策略相关的数据
        :param policy_id: 玩家ID
        :param policy_data: 策略相关数据: S(t) => Policy => a(t), V(t), logit(t)
        """
        if not self.policy_data[policy_id]:
            env_buffer = {}
            for env_id, new_data in policy_data.items():
                env_buffer[env_id] = {}
                for agent_id, agent_new_data in new_data.items():
                    env_buffer[env_id][agent_id] = {}
                    for new_key, new_value in agent_new_data.items():
                        env_buffer[env_id][agent_id][new_key] = [copy.deepcopy(new_value)]
            self.policy_data[policy_id] = env_buffer
        else:
            for env_id, env_buffer in self.policy_data[policy_id].items():
                new_data = policy_data[env_id]
                for agent_id, agent_new_data in new_data.items():
                    if agent_id not in env_buffer:
                        env_buffer[agent_id] = {}
                    for new_key, new_value in agent_new_data.items():
                        if new_key not in env_buffer[agent_id]:
                            env_buffer[agent_id][new_key] = []
                        env_buffer[agent_id][new_key].append(copy.deepcopy(new_value))

    def add_env_data(self, policy_id: int, env_data):
        """
        添加执行动作后,环境输出的数据
        :param policy_id: 玩家ID
        :param env_data: 环境输出数据: env.step(a(t)) => r(t), done(t+1)
        """
        for env_id, env_buffer in self.policy_data[policy_id].items():
            new_data = env_data[env_id]
            for agent_id, agent_done, agent_reward in zip(new_data.get('reserved_agent_id', new_data['agent_id']),
                                                          new_data.get('reserved_done', new_data['done']),
                                                          new_data.get('reserved_reward', new_data['reward'])):
                if agent_id not in env_buffer:  # 没有policy data, (next step添加)
                    continue

                if 'done' not in env_buffer[agent_id]:
                    env_buffer[agent_id]['done'] = []
                if 'reward' not in env_buffer[agent_id]:
                    env_buffer[agent_id]['reward'] = []

                env_buffer[agent_id]['done'].append(copy.deepcopy(agent_done))  # t+1 时刻
                env_buffer[agent_id]['reward'].append(copy.deepcopy(agent_reward))  # t时刻

    def reset(self):
        self.policy_data.clear()

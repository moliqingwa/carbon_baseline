from typing import Dict, AnyStr, Union, Tuple, List
from collections import defaultdict
from easydict import EasyDict
import copy

import numpy as np
import gym

from envs.carbon_env import CarbonEnv
from zerosum_env.envs.carbon.helpers import RecrtCenterAction, WorkerAction, Board

BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]

WorkerDirections = np.stack([np.array((0, 0)),
                             np.array((0, 1)),
                             np.array((1, 0)),
                             np.array((0, -1)),
                             np.array((-1, 0))])  # 与WorkerActions相对应


def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret


class CarbonTrainerEnv:
    def __init__(self, cfg: dict):
        self.previous_action = {}  # key: agent_name, value: {'value': [cmd_value...], 'cmd': [cmd_str...]}
        self.agent_cmds = {}

        self.previous_obs = self.current_obs = None  # 记录连续两帧 Observation
        self.previous_opponent_obs = self.current_opponent_obs = None  # 记录对手连续两帧 Observation,仅在selfplay时使用

        self._env = CarbonEnv(cfg)

        self.grid_size = self.configuration.size
        self.max_step = self.configuration.episodeSteps

    @property
    def configuration(self):
        return self._env.env.configuration

    @property
    def act_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(5)

    @property
    def observation_cnn_shape(self) -> Tuple[int, int, int]:
        return 13, 15, 15

    @property
    def observation_vector_shape(self) -> int:
        return 8

    @property
    def observation_dim(self) -> int:
        return 8 + 13 * 15 * 15

    def _get_latest_state(self):
        if self._env.my_index == 0:  # 当前轮次
            my_state, opponent_state = self._env.env.steps[-1]

            for key in set(my_state.observation.keys()) - set(opponent_state.observation.keys()):  # 复制公共属性
                opponent_state.observation[key] = my_state.observation[key]
        else:
            opponent_state, my_state = self._env.env.steps[-1]

            for key in set(opponent_state.observation.keys()) - set(my_state.observation.keys()):  # 复制公共属性
                my_state.observation[key] = opponent_state.observation[key]
        return my_state, opponent_state

    def reset(self, players=None):
        self.previous_action.clear()

        self._env.reset(players)
        my_state, opponent_state = self._get_latest_state()

        self.previous_obs = None
        self.current_obs = Board(my_state.observation, self.configuration)

        my_output = self._output_per_player(my_state, opponent_state, self.current_obs, None,
                                            reset=True)

        if self._env.selfplay:  # 自我对局
            self.previous_opponent_obs = None
            self.current_opponent_obs = Board(opponent_state.observation, self.configuration)

            opponent_output = self._output_per_player(opponent_state, my_state, self.current_opponent_obs, None,
                                                      reset=True)
            return [my_output, opponent_output]
        else:
            return my_output

    def step(self, actions: Tuple[dict, List[dict]]):
        if isinstance(actions, list):
            self.previous_action = {k: v for x in actions for k, v in x.items()}
        else:
            self.previous_action = copy.deepcopy(actions)
            actions = [actions, None]

        commands = []
        for action in actions:
            if action is None:
                command = None
            else:
                command = {agent_name: self.agent_cmds[agent_name][cmd_value].name
                           for agent_name, cmd_value in action.items() if cmd_value != 0}  # 0 is None, no need to send!
            commands.append(command)

        self._env.step(commands)

        my_state, opponent_state = self._get_latest_state()  # 当前轮次

        self.previous_obs = self.current_obs
        self.current_obs = Board(my_state.observation, self.configuration)

        my_output = self._output_per_player(my_state, opponent_state, self.current_obs, self.previous_obs)
        # self._dump_state_and_rewards(my_output, my_state)  # 仅单进程测试使用!

        if self._env.selfplay:
            self.previous_opponent_obs = self.current_opponent_obs
            self.current_opponent_obs = Board(opponent_state.observation, self.configuration)

            opponent_output = self._output_per_player(opponent_state, my_state,
                                                      self.current_opponent_obs, self.previous_opponent_obs)
            return [my_output, opponent_output]
        else:
            return my_output

    def _output_per_player(self, my_state, opponent_state, current_obs: Board, previous_obs: Board, *,
                           reset=False):
        env_done = my_state.status != "ACTIVE"

        if not reset:  # 计算reward
            raw_env_reward, agent_reward_dict = self._calculate_reward(my_state, opponent_state)
            alive_agent_total_reward = sum([v.get('tree', 0) + v.get('carbon', 0)
                                            for v in agent_reward_dict.values()])  # 所有活着的agent的总奖励
            extra_tree_reward = agent_reward_dict.get(None, {}).get('tree', 0)  # 无主之树的奖励
            # 碰撞,被自己树/转化中心直接吸收奖励(碰撞后,agent可能活着,也可能死亡) TODO: 区分哪个agent带来的 ???
            extra_reward = round(raw_env_reward - extra_tree_reward - alive_agent_total_reward, 5)  # env_reward: 可正可负(招聘)!!

        agent_obs, dones, available_actions = self._obs_transform(current_obs, previous_obs)

        output = defaultdict(list)
        for agent_id, obs in agent_obs.items():
            output['agent_id'].append(agent_id)
            output['obs'].append(obs)

            if not reset:
                agent_done = env_done | dones[agent_id]  # 环境结束 或 agent结束
                output['done'].append(agent_done)

                # 计算活着的agent本局的收益
                agent_tree_reward = agent_reward_dict.get(agent_id, {}).get('tree', 0)  # 种树/抢树 带来的奖励(每轮)
                agent_carbon_reward = agent_reward_dict.get(agent_id, {}).get('carbon', 0)  # 捕碳并运回家的奖励
                agent_reward = agent_tree_reward + agent_carbon_reward

                ratio = my_state.observation['step'] / (self.max_step - 1)  # 步数权重(越到后期,权重越高)
                env_reward = raw_env_reward if env_done else extra_reward
                reward = (1 - int(env_done)) * (1 - ratio) * agent_reward + ratio * env_reward
                if 'recrtCenter' in agent_id:  # TODO: hack here to check is the base or not
                    reward = raw_env_reward  # 转化中心因无法产生直接收益,因此使用整体对手的奖励
                elif reward == 0:
                    reward = -0.0034
                output['reward'].append(reward)

            output['info'].append({})
            output['available_actions'].append(available_actions[agent_id])

        if not reset:
            output['env_reward'] = env_reward
        return dict(output)

    def close(self):
        pass

    def _calculate_reward(self, my_state, opponent_state):
        game_end_code = None
        if my_state.status != "ACTIVE":  # 对局结束
            if my_state.reward == opponent_state.reward:  # 两选手分数相同(float)/或者均出错(None) (平局)
                game_end_code = 0
            elif my_state.reward is None:  # 我输,对手赢
                game_end_code = -1
            elif opponent_state.reward is None:  # 我赢,对手输
                game_end_code = 1
            elif my_state.reward > opponent_state.reward:  # 我赢,对手输
                game_end_code = 1
            elif my_state.reward < opponent_state.reward:  # 我输,对手赢
                game_end_code = -1
            else:
                raise Exception("Should not go to here!")

        env_reward = current_cash = my_state.reward  # 选手当前轮次的金额 (注意:游戏结束时,返回的数值不准确,结束时慎用!!!)
        agent_reward_dict = defaultdict(dict)  # key: worker_id, value: {'tree': '种/抢树后获得的收益', 'carbon': '捕碳收益'}
        if len(self._env.env.steps) > 1:
            previous_my_state = self._env.env.steps[-2][self._env.my_index]

            if game_end_code is not None:  # 游戏结束(注意: 游戏结束时,返回的current_cash不准确;未结束时,才准确!!!)
                env_reward = game_end_code * self.max_step
            elif current_cash is not None:
                env_reward = current_cash - previous_my_state.reward

            # 计算每个agent 种(抢)树和捕碳的每轮收益
            trees_dict = my_state.observation["trees"]
            tree_without_owner_reward = sum([tree_reward for tree_owner, tree_reward in
                                             trees_dict.values() if tree_owner is None])  # 无主之树的收益(owner已消失)
            agent_reward_dict[None]['tree'] = tree_without_owner_reward

            # 上一轮次
            _, _, previous_my_workers, previous_my_trees = previous_my_state["observation"]["players"][
                self._env.my_index]
            # 当前轮次
            _, my_bases, my_workers, my_trees = my_state.observation["players"][self._env.my_index]
            my_base_pos = next(iter(my_bases.values()))
            for worker_id, (pos, carbon, type_) in my_workers.items():  # 当前轮次
                tree_reward = 0  # 种树/抢树 收益
                if worker_id in previous_my_workers:
                    tree_reward += sum([tree_reward for tree_id, (tree_owner, tree_reward)
                                        in trees_dict.items() if tree_owner == worker_id])
                agent_reward_dict[worker_id]['tree'] = self._normalize_reward(tree_reward)

                # 捕碳 收益
                if pos == my_base_pos:
                    carbon_reward = previous_my_workers.get(worker_id, [None, 0])[1]  # 运回基地收益 (>= 0)
                else:
                    delta_carbon = my_workers.get(worker_id, [None, 0])[1] - \
                                   previous_my_workers.get(worker_id, [None, 0])[1]
                    carbon_reward = min(delta_carbon, 0)  # 身上碳被吸收 (惩罚项); 捕碳中,无奖励
                agent_reward_dict[worker_id]['carbon'] = self._normalize_reward(carbon_reward)

        return self._normalize_reward(env_reward), agent_reward_dict

    def _obs_transform(self, current_obs: Board, previous_obs: Board = None):
        # 加入对手agent上一轮次的动作
        opponent_cmds = self._guess_opponent_previous_actions(previous_obs, current_obs)
        self.previous_action.update({k: v.value if v is not None else 0
                                     for k, v in opponent_cmds.items()})

        available_actions = {}
        my_player_id = current_obs.current_player_id

        carbon_feature = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for point, cell in current_obs.cells.items():
            if cell.carbon > 0:
                carbon_feature[point.x, point.y] = cell.carbon / self.configuration.maxCellCarbon

        step_feature = current_obs.step / (self.max_step - 1)
        base_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        collector_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        planter_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        worker_carbon_feature = np.zeros_like(carbon_feature, dtype=np.float32)
        tree_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # trees, me: +; opponent: -.
        action_feature = np.zeros((self.grid_size, self.grid_size, self.act_space.n), dtype=np.float32)

        my_base_distance_feature = None
        distance_features = {}

        my_cash, opponent_cash = current_obs.current_player.cash, current_obs.opponents[0].cash
        for base_id, base in current_obs.recrtCenters.items():
            is_myself = base.player_id == my_player_id

            base_x, base_y = base.position.x, base.position.y

            base_feature[base_x, base_y] = 1.0 if is_myself else -1.0
            base_distance_feature = self._distance_feature(base_x, base_y) / (self.grid_size - 1)
            distance_features[base_id] = base_distance_feature

            action_feature[base_x, base_y] = one_hot_np(self.previous_action.get(base_id, 0), self.act_space.n)
            if is_myself:
                available_actions[base_id] = np.array([1, 1, 1, 0, 0])  # TODO
                self.agent_cmds[base_id] = BaseActions

                my_base_distance_feature = distance_features[base_id]

        for worker_id, worker in current_obs.workers.items():
            is_myself = worker.player_id == my_player_id

            available_actions[worker_id] = np.array([1, 1, 1, 1, 1])  # TODO
            self.agent_cmds[worker_id] = WorkerActions

            worker_x, worker_y = worker.position.x, worker.position.y
            distance_features[worker_id] = self._distance_feature(worker_x, worker_y) / (self.grid_size - 1)

            action_feature[worker_x, worker_y] = one_hot_np(self.previous_action.get(worker_id, 0), self.act_space.n)

            if worker.is_collector:
                collector_feature[worker_x, worker_y] = 1.0 if is_myself else -1.0
            else:
                planter_feature[worker_x, worker_y] = 1.0 if is_myself else -1.0

            worker_carbon_feature[worker_x, worker_y] = worker.carbon
        worker_carbon_feature = np.clip(worker_carbon_feature / self.configuration.maxCellCarbon / 2, -1, 1)

        for tree in current_obs.trees.values():
            tree_feature[tree.position.x, tree.position.y] = tree.age if tree.player_id == my_player_id else -tree.age
        tree_feature /= self.configuration.treeLifespan

        global_vector_feature = np.stack([step_feature,
                                          np.clip(my_cash / 2000., -1., 1.),
                                          np.clip(opponent_cash / 2000., -1., 1.),
                                          ]).astype(np.float32)
        global_cnn_feature = np.stack([carbon_feature,
                                       base_feature,
                                       collector_feature,
                                       planter_feature,
                                       worker_carbon_feature,
                                       tree_feature,
                                       *action_feature.transpose(2, 0, 1),  # dim: 5 x 15 x 15
                                       ])  # dim: 11 x 15 x 15

        dones = {}
        local_obs = {}
        previous_worker_ids = set() if previous_obs is None else set(previous_obs.current_player.worker_ids)
        worker_ids = set(current_obs.current_player.worker_ids)
        new_worker_ids, death_worker_ids = worker_ids - previous_worker_ids, previous_worker_ids - worker_ids
        obs = previous_obs if previous_obs is not None else current_obs
        total_agents = obs.current_player.recrtCenters + \
                       obs.current_player.workers + \
                       [current_obs.workers[id_] for id_ in new_worker_ids]  # 基地 + prev_workers + new_workers
        for my_agent in total_agents:
            if my_agent.id in death_worker_ids:  # 死亡的agent, 直接赋值为0
                local_obs[my_agent.id] = np.zeros(self.observation_dim, dtype=np.float32)
                available_actions[my_agent.id] = np.array([1, 1, 1, 1, 1])  # TODO
                dones[my_agent.id] = True
            else:  # 未死亡的agent
                cnn_feature = np.stack([*global_cnn_feature,
                                        my_base_distance_feature,
                                        distance_features[my_agent.id],
                                        ])  # dim: 2925 (13 x 15 x 15)
                if not hasattr(my_agent, 'is_collector'):  # 转化中心
                    agent_type = [1, 0, 0]
                else:  # 工人
                    agent_type = [0, int(my_agent.is_collector), int(my_agent.is_planter)]
                vector_feature = np.stack([*global_vector_feature,
                                           *agent_type,
                                           my_agent.position.x / self.grid_size,
                                           my_agent.position.y / self.grid_size,
                                           ]).astype(np.float32)  # dim: 8
                local_obs[my_agent.id] = np.concatenate([vector_feature, cnn_feature.reshape(-1)])
                dones[my_agent.id] = False

        return local_obs, dones, available_actions

    def _guess_opponent_previous_actions(self, previous_board: Board, board: Board) -> Dict:
        """
        基于连续两帧Board信息,猜测对手采用的动作(已经消失的agent,因无法准确估计,故忽略!)

        :return:  字典, key为agent_id, value为Command或None
        """
        return_value = {}

        previous_workers, workers = {}, {}
        if previous_board is not None:
            previous_workers = {w.id: w for w in previous_board.opponents[0].workers}
        if board is not None:
            workers = {w.id: w for w in board.opponents[0].workers}

        base_cmd = BaseActions[0]
        total_worker_ids = set(previous_workers.keys()) | set(workers.keys())  # 对手的worker id列表
        for worker_id in total_worker_ids:
            previous_worker, worker = previous_workers.get(worker_id, None), workers.get(worker_id, None)
            if previous_worker is not None and worker is not None:  # (连续两局存活) 移动/停留 动作
                prev_pos = np.array([previous_worker.position.x, previous_worker.position.y])
                curr_pos = np.array([worker.position.x, worker.position.y])

                # 计算所有方向的可能位置 (防止越界问题)
                next_all_positions = ((prev_pos + WorkerDirections) + self.grid_size) % self.grid_size
                dir_index = (next_all_positions == curr_pos).all(axis=1).nonzero()[0].item()
                cmd = WorkerActions[dir_index]

                return_value[worker_id] = cmd
            elif previous_worker is None and worker is not None:  # (首次出现) 招募 动作
                if worker.is_collector:
                    base_cmd = BaseActions[1]
                elif worker.is_planter:
                    base_cmd = BaseActions[2]
            else:  # Agent已消失(因无法准确推断出动作), 忽略
                pass

        return_value[board.opponents[0].recrtCenter_ids[0]] = base_cmd
        return return_value

    # obtain the distance of current position to other positions
    def _distance_feature(self, x, y):
        distance_y = (np.ones((self.grid_size, self.grid_size)) * np.arange(self.grid_size)).astype(np.float32)
        distance_x = distance_y.T
        delta_distance_x = abs(distance_x - x)
        delta_distance_y = abs(distance_y - y)
        offset_distance_x = self.grid_size - delta_distance_x
        offset_distance_y = self.grid_size - delta_distance_y
        distance_x = np.where(delta_distance_x < offset_distance_x,
                              delta_distance_x, offset_distance_x)
        distance_y = np.where(delta_distance_y < offset_distance_y,
                              delta_distance_y, offset_distance_y)
        distance_map = distance_x + distance_y

        return distance_map

    def _normalize_reward(self, reward):
        return np.clip(reward / self.max_step, -1, 1)

    def _dump_state_and_rewards(self, my_output, my_state):
        """
        保存游戏整个state和reward到文本文件中,方便事后分析使用 (仅单进程环境下使用)

        :param my_output:
        :param my_state:
        :return:
        """
        with open("rewards.txt", 'a', encoding='utf-8') as f:
            f.write("\n")
            f.write(f"=== STEP: {self.current_obs.step} ===\n")
            f.write(self.current_obs.__str__())
            f.write("\n")
            rewards = {k: round(v, 3) for k, v in zip(my_output['agent_id'], my_output['reward'])}
            f.write(f"{rewards}\n")
            if any([r > 0 for r in rewards.values()]):
                print(my_state.observation.step)

        if all(my_output['done']):
            import json
            with open("episode.txt", "w", encoding='utf-8') as f:
                for step in self._env.env.steps:
                    f.write(json.dumps(step) + "\n")

            import os
            try:
                os.rename("rewards.txt", "rewards_archive.txt")
            except:
                pass


from collections import defaultdict
from typing import Dict, Union, Tuple, List

from easydict import EasyDict
import gym
import numpy as np

from envs.carbon_env import CarbonEnv
from envs.obs_parser import WorkerDirections, WorkerActionsByName, ObservationParser
from zerosum_env.envs.carbon.helpers import Board, Point


def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret


class CarbonTrainerEnv:
    def __init__(self, cfg: dict):
        self.previous_obs = self.current_obs = None  # 记录连续两帧 Observation
        self.previous_opponent_obs = self.current_opponent_obs = None  # 记录对手连续两帧 Observation,仅在selfplay时使用
        self.previous_commands = []

        self._env = CarbonEnv(cfg)

        self.grid_size = self.configuration.size
        self.max_step = self.configuration.episodeSteps

        self.observation_parser = ObservationParser(grid_size=self.configuration.size,
                                                    max_step=self.configuration.episodeSteps,
                                                    max_cell_carbon=self.configuration.maxCellCarbon,
                                                    tree_lifespan=self.configuration.treeLifespan,
                                                    action_space=self.act_space.n)

    @property
    def configuration(self):
        """
        Return full configuration of carbon env.
        """
        return self._env.env.configuration

    @property
    def act_space(self) -> gym.spaces.Discrete:
        """
        Return action space for each agent (default is 5).
        """
        return gym.spaces.Discrete(5)

    def _get_latest_state(self) -> Tuple:
        """
        Return two player's state information.
        The current player's state is in the first position, the opponent's is in the second position.
        """
        if self._env.my_index == 0:  # 当前轮次
            my_state, opponent_state = self._env.env.steps[-1]

            for key in set(my_state.observation.keys()) - set(opponent_state.observation.keys()):  # 复制公共属性
                opponent_state.observation[key] = my_state.observation[key]
        else:
            opponent_state, my_state = self._env.env.steps[-1]

            for key in set(opponent_state.observation.keys()) - set(my_state.observation.keys()):  # 复制公共属性
                my_state.observation[key] = opponent_state.observation[key]
        return my_state, opponent_state

    def reset(self, players: Union[None, List] = None) -> Union[EasyDict, List[EasyDict]]:
        """
        Reset the env.
        :param players: If you want to play with 'random' player inside the game, you can use the default value (None).
            Otherwise, you should set to your two players' functions.

        :return my_output, opponent_output: Dict output for one player, List output for self-play (the first one is
            current player, another is the opponent player).
        """

        self._env.reset(players)
        my_state, opponent_state = self._get_latest_state()

        self.previous_obs = None
        self.current_obs = Board(my_state.observation, self.configuration)
        self.previous_commands.clear()

        my_output = self._parse_observation_and_reward(my_state, opponent_state,
                                                       self.current_obs, None,
                                                       env_reset=True)

        if self._env.selfplay:  # 自我对局, 返回对手状态
            self.previous_opponent_obs = None
            self.current_opponent_obs = Board(opponent_state.observation, self.configuration)

            opponent_output = self._parse_observation_and_reward(opponent_state, my_state,
                                                                 self.current_opponent_obs, None,
                                                                 env_reset=True)
            return [my_output, opponent_output]
        else:
            return my_output

    def step(self, commands: Tuple[Dict[str, str], List[Dict[str, str]]]):
        if isinstance(commands, dict):
            commands = [commands, None]
        self.previous_commands = commands

        self._env.step(commands)

        my_state, opponent_state = self._get_latest_state()  # 当前轮次

        self.previous_obs = self.current_obs
        self.current_obs = Board(my_state.observation, self.configuration)

        my_output = self._parse_observation_and_reward(my_state, opponent_state, self.current_obs, self.previous_obs)

        if self._env.selfplay:
            self.previous_opponent_obs = self.current_opponent_obs
            self.current_opponent_obs = Board(opponent_state.observation, self.configuration)

            opponent_output = self._parse_observation_and_reward(opponent_state, my_state,
                                                                 self.current_opponent_obs, self.previous_opponent_obs)
            return [my_output, opponent_output]
        else:
            return my_output

    def _parse_observation_and_reward(self, my_state, opponent_state: Union[None, Board],
                                      current_obs: Board, previous_obs: Union[None, Board], *,
                                      env_reset=False):
        """
        计算Observation特征以及各个Agent的reward
        """
        # 解析observation生成observation特征
        agent_obs, dones, available_actions = self.observation_parser.obs_transform(current_obs, previous_obs)

        output = defaultdict(list)
        for agent_id, obs in agent_obs.items():
            output['agent_id'].append(agent_id)
            output['obs'].append(obs)

            output['info'].append({})
            output['available_actions'].append(available_actions[agent_id])

        if not env_reset:  # 若游戏未被重置, 则添加reward等相关信息
            raw_env_reward, agent_reward_dict = self._calculate_reward(my_state, opponent_state, current_obs,
                                                                       previous_obs)
            output['env_reward'] = raw_env_reward  # 游戏全局 reward

            env_done = my_state.status != "ACTIVE"  # 游戏是否结束
            for agent_id in agent_obs:  # 添加agent数据 done(t+1)
                agent_done = env_done | dones[agent_id]  # 环境结束 或 agent结束
                output['done'].append(agent_done)
                output['reward'].append(agent_reward_dict[agent_id])  # 单agent的reward

        return EasyDict(output)

    def close(self):
        pass

    def _calculate_reward(self, my_state, opponent_state,
                          current_obs: Board, previous_obs: Board,
                          normalize=False) -> Tuple[float, Dict[str, float]]:
        """
        基于当前轮次和前一个轮次,估算玩家reward和玩家各个agent的reward.

        :return env_reward, agent_rewards: my_state的游戏奖励以及my_state中我方各个agent的奖励
        """
        assert previous_obs is not None  # S(t), a(t) -> r(t), S(t+1)

        # 游戏结束状态
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

        # 工人信息
        my_player = current_obs.players[current_obs.current_player_id]  # 本轮次
        my_base_position = my_player.recrtCenters[0].position
        my_workers = {worker.id: worker for worker in my_player.workers}
        previous_my_player = previous_obs.players[previous_obs.current_player_id]  # 上一轮次
        previous_my_workers = {worker.id: worker for worker in previous_my_player.workers}

        # 通树的信息,计算每个agent种树/抢树的花费
        current_trees = {tree.id: tree for tree in current_obs.current_player.trees}
        previous_trees = {tree.id: tree for tree in previous_obs.current_player.trees}
        new_tree_ids = list(set(current_trees.keys()) - set(previous_trees.keys()))
        worker_plant_cost = defaultdict(float)  # 种/抢树需要额外的花费
        if new_tree_ids:  # 计算种树/抢树的花费
            plant_cost = self.configuration.plantCost  # 默认种树价格
            plant_cost_ratio = self.configuration.plantCostInflationRatio
            plant_cost_base = self.configuration.plantCostInflationBase
            alive_tree_count = len(current_obs.trees)
            plant_market_price = plant_cost + plant_cost_ratio * (plant_cost_base ** alive_tree_count)  # 市场价格
            is_first_tree = not previous_trees  # 首次种树价格: plant_cost, 其余树价格: plant_market_price
            seize_cost = self.configuration.seizeCost  # 抢树价格
            for tree_id in new_tree_ids:
                tree = current_trees[tree_id]
                tree_owner, _ = my_state.observation.trees[tree_id]
                if tree.age == 1:  # 种树
                    worker_plant_cost[tree_owner] += plant_cost if is_first_tree else plant_market_price
                else:  # 抢树
                    worker_plant_cost[tree_owner] += seize_cost

        # 下面计算单个agent的reward
        max_reward = self.max_step
        agent_reward_dict = {}  # 单个agent的reward, key: worker_id (含已死亡), value: 智能体reward
        if game_end_code is not None:  # 游戏结束(注意: 游戏结束时,返回的current_cash不准确;未结束时,才准确!!!)
            env_reward = game_end_code * max_reward
        else:  # 游戏未结束
            env_reward = my_player.cash - previous_my_player.cash  # 选手金额的变化

        agent_accumulate_reward = 0  # 各agent的奖励总和

        # 计算每个agent 种(抢)树和捕碳的每轮收益
        tree_owner_reward = dict()  # key: worker_id, value: float
        trees_dict = my_state.observation["trees"]
        for tree_owner, tree_reward in trees_dict.values():
            if tree_owner not in tree_owner_reward:
                tree_owner_reward[tree_owner] = 0

            tree_owner_reward[tree_owner] += tree_reward

            agent_accumulate_reward += tree_reward

        # 当前轮次，所有工人的reward
        for worker in my_player.workers:
            is_fresher = worker.id not in previous_my_workers  # 是否为新招募成员

            # 计算种(抢)树收益
            if worker.is_planter:
                tree_reward = tree_owner_reward.get(worker.id, 0)  # 树收益
                tree_reward = -1 if tree_reward == 0 else tree_reward
            else:  # 捕碳员
                tree_reward = tree_owner_reward.get(worker.id, 0)

            # 计算捕碳收益
            carbon_reward = 0
            if worker.is_collector:
                # 上一轮携带的CO2量
                previous_carbon = 0 if is_fresher else previous_my_workers[worker.id].carbon
                current_carbon = worker.carbon  # 当前轮携带的CO2量

                if not is_fresher and worker.position == my_base_position:  # 非新招募人员,且在转化中心位置处
                    carbon_reward = previous_carbon  # 运回基地收益 (>= 0)
                    agent_accumulate_reward += carbon_reward
                else:
                    carbon_reward = current_carbon - previous_carbon  # +/-/0, 正值:携带CO2增多,负值:被对方树吸收,0:游走
                    carbon_reward = carbon_reward / 100. if carbon_reward >= 0 else carbon_reward  # 捕碳增加量 / 100
                    carbon_reward = min(carbon_reward, 0.5)  # 身上碳减少 (惩罚项); 捕碳,最高奖励 +0.5
                if carbon_reward == 0:  # 捕碳无收益
                    carbon_reward = -1

            worker_reward = tree_reward + carbon_reward - worker_plant_cost[worker.id]  # 树净化收益 + 捕碳收益 - 树价格
            agent_reward_dict[worker.id] = worker_reward

        # 碰撞,被自己树/转化中心直接吸收奖励(碰撞后,agent可能活着,也可能死亡). 记录一下，暂时无法区分
        env_extra_reward = max(round(env_reward - agent_accumulate_reward, 2), 0)  # 排除各agent自己的奖励

        # 已经死亡的agent
        death_agent_ids = {id_ for id_ in previous_my_workers.keys() if id_ not in agent_reward_dict}
        agent_reward_dict.update({id_: -max_reward for id_ in death_agent_ids})
        # death_agent_reward = max(env_extra_reward, 0)
        # agent_reward_dict.update({id_: -max_reward if death_agent_reward == 0 else death_agent_reward
        #                           for id_ in death_agent_ids})

        # 基于上一轮次的位置+动作,计算新位置是否有碰撞,若碰撞,则惩罚
        is_my_base_recruit = False  # 是否有招募新工人
        position_agent_dict = defaultdict(set)
        no_action_agent_ids = set()
        if my_player.id < len(self.previous_commands):
            previous_my_command = self.previous_commands[my_player.id]
            if previous_my_command:
                is_my_base_recruit = my_player.recrtCenters[0].id in previous_my_command
                for worker_id, worker in previous_my_workers.items():
                    if worker_id in previous_my_command:  # 有移动动作
                        dx, dy = WorkerDirections[WorkerActionsByName[previous_my_command[worker_id]].value].tolist()
                        new_position = worker.position + Point(dx, dy)
                        position_agent_dict[new_position].add(worker_id)
                    else:  # 停留动作
                        no_action_agent_ids.add(worker_id)
                        position_agent_dict[worker.position].add(worker_id)
        # 转化中心的奖励
        if is_my_base_recruit and my_base_position in position_agent_dict:  # 转化中心招募,导致碰撞
            for base in my_player.recrtCenters:
                agent_reward_dict[base.id] = -max_reward
        else:
            for base in my_player.recrtCenters:
                agent_reward_dict[base.id] = env_reward

        for agent_ids in position_agent_dict.values():
            if len(agent_ids) > 1:  # 人员移动, 发生碰撞
                for agent_id in agent_ids:
                    if agent_id not in no_action_agent_ids:  # 仅仅惩罚移动的人员
                        agent_reward_dict[agent_id] = -max_reward

        if normalize:
            agent_reward_dict = {k: self._normalize_reward(v)
                                 for k, v in agent_reward_dict.items()}
        return env_reward, agent_reward_dict

    def _normalize_reward(self, reward):
        return np.clip(reward / self.max_step, -1, 1)

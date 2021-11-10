import gym
import copy
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
from zerosum_env.envs.carbon.helpers import RecrtCenterAction, WorkerAction, Board

''' 转化中心动作
'''
BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

''' worker动作
'''
WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]

''' worker方向
'''
WorkerDirections = np.stack([np.array((0, 0)),
                             np.array((0, 1)),
                             np.array((1, 0)),
                             np.array((0, -1)),
                             np.array((-1, 0))])  # 与WorkerActions相对应

def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret

class Controller:
    def __init__(self, configuration):
        self.previous_action = {}  # key: agent_name, value: {'value': [cmd_value...], 'cmd': [cmd_str...]}
        self.agent_cmds = {}

        self.previous_obs = self.current_obs = None  # 记录连续两帧 Observation

        self.configuration = configuration
        self.grid_size = self.configuration.size
        self.max_step = self.configuration.episodeSteps

    @property
    def act_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(5)

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

    def _output_per_player(self, current_obs: Board, previous_obs: Board, *,
                            reset=False):
            agent_obs, dones, available_actions = self._obs_transform(current_obs, previous_obs)

            output = defaultdict(list)
            for agent_id, obs in agent_obs.items():
                output['agent_id'].append(agent_id)
                output['obs'].append(obs)

                if not reset:
                    agent_done = dones[agent_id]  # 环境结束 或 agent结束
                    output['done'].append(agent_done)

                output['info'].append({})
                output['available_actions'].append(available_actions[agent_id])
            return dict(output)

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
        return commands


def agent(observation, configuration):
    board = Board(observation, configuration)
    controller = Controller(configuration)
    env_output = controller._output_per_player(board, None)
    actions = {}
    for agent_id, obs, available_actions in zip(env_output['agent_id'],
                                                env_output['obs'],
                                                env_output['available_actions']):
                cmd_value = random.choice(list(map(lambda v: v[0],
                                                   filter(lambda v: v[1] == 1,
                                                          enumerate(available_actions)))))
                actions[agent_id] = cmd_value
    return controller.step(actions)[0]


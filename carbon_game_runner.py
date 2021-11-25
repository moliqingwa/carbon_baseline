from typing import Dict, List, Tuple, Any, Union
import os
from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch

from utils.utils import synthesize
from utils.trajectory_buffer import TrajectoryBuffer
from utils.replay_buffer import ReplayBuffer

from algorithms.base_policy import BasePolicy
from algorithms.learner_policy import LearnerPolicy
from algorithms.learner_partner_policy import LeanerPartnerPolicy


class CarbonGameRunner:
    """
    Runner class to perform training of learner policy, evaluation and data collection.
    """
    def __init__(self, cfg: EasyDict):
        self._cfg = cfg
        self.env = cfg.env
        self.episodes = cfg.main_config.runner.episodes
        self.episode_length = cfg.main_config.runner.episode_length
        self.n_threads = cfg.main_config.envs.n_threads
        self.gamma = cfg.main_config.runner.gamma
        self.use_gae = cfg.main_config.runner.use_gae
        self.gae_lambda = cfg.main_config.runner.gae_lambda
        self.device = cfg.main_config.runner.device
        self.selfplay = cfg.main_config.runner.selfplay
        self.save_interval = cfg.main_config.runner.save_interval
        self.tb_writer = cfg.tb_writer

        self.start_episode = 0
        self.save_model_dir = cfg.run_dir / "models"

        self._env_output = None
        self._trajectory_buffer = TrajectoryBuffer()
        self._replay_buffer = ReplayBuffer(cfg.main_config.runner.buffer_size, cfg.main_config.runner.device)

        self.learner_policy = LearnerPolicy(cfg)  # 待训练的策略

        # 下面为selfplay的相关参数
        self.partner_policy = LeanerPartnerPolicy(cfg, self.save_model_dir)  # 陪练机器人
        self.policies = [self.learner_policy, self.partner_policy]

        # 收集的训练/统计相关信息
        self._env_returns = defaultdict(float)  # env_id, returns

    def run(self):
        """
        Collect training data, perform training updates, and evaluate policy.
        """
        self._env_output = self.env.reset(self.selfplay)

        self._trajectory_buffer.reset()

        best_model_threshold = None  # 筛选最佳策略使用
        for episode in range(self.start_episode, self.episodes):
            for policy in self.policies:  # 重置策略
                policy.policy_reset(episode, self.episodes)

            # collect transitions of whole game. S(0), a(0), r(0), dones(1) -> ... -> S(T), r(T-1), dones(T)
            experiences, collect_logs = self.collect_full_episode()
            for experience_data in experiences:
                self._replay_buffer.append(experience_data)

            # PPO training
            train_logs = self.learner_policy.train(self._replay_buffer)
            self._replay_buffer.reset()  # drop training data

            # save state
            v = collect_logs['env_return']
            if best_model_threshold is None or v >= best_model_threshold:  # save best model
                self.save(episode, is_best=True)
                best_model_threshold = v

            if episode % self.save_interval == 0 or episode == self.episodes - 1:  # save model
                self.save(episode)

            # console log
            log_value = f"E {episode}/{self.episodes} | " \
                        f"agent {collect_logs['alive_agent_count']:.0f}/{collect_logs['accumulate_agent_count']:.0f} | " \
                        f"w/d/l {collect_logs['win_count']:.1f}/{collect_logs['draw_count']:.1f}/" \
                        f"{collect_logs['lose_count']:.1f} | " \
                        f"S {collect_logs['step_duration']:.0f} | " \
                        f"r::MμσmM {collect_logs['agent_reward_median']:.3f} {collect_logs['agent_reward_mean']:.3f} " \
                        f"{collect_logs['agent_reward_std']:.3f} " \
                        f"{collect_logs['agent_reward_min']:.3f} {collect_logs['agent_reward_max']:.3f} | " \
                        f"R {collect_logs['env_return']:.3f} || " \
                        f"V {train_logs['value']:.3f} | " \
                        f"aL {train_logs['actor_loss']:.3f} | vL {train_logs['critic_loss']:.3f} | " \
                        f"∇:ac {train_logs['actor_grad_norm']:.3f} {train_logs['critic_grad_norm']:.3f} | " \
                        f"H {train_logs['entropy']:.3f} | A {train_logs['advantage']:.3f} | " \
                        f"kl {train_logs['approx_kl']:.3f} | r {train_logs['ratio']:.3f}"
            print(log_value)

            # tensorboard recording
            if self.tb_writer is not None:
                for field, value in collect_logs.items():
                    self.tb_writer.add_scalar(field, value, episode)
                for field, value in train_logs.items():
                    self.tb_writer.add_scalar(field, value, episode)

    def collect_full_episode(self) -> Tuple[List[List[Dict[str, Dict]]], Dict[str, float]]:
        """
        collect full transitions and statistical logs during the full episode.

        :return return_data: (List[List[Dict[str, Dict]]]) full transitions of the agents appeared in the game.
        :return collect_logs: (Dict[str, float]) the statistical data of the transitions data
        """
        return_data = []
        collect_logs = defaultdict(list)
        for step in range(self.episode_length):
            experience_data, collect_log = self._collect()

            if experience_data:  # add to replay buffer
                return_data.append(experience_data)

                for key, value in collect_log.items():
                    collect_logs[key].extend(value)
        collect_logs = {k: np.mean(v) for k, v in collect_logs.items()}
        return return_data, collect_logs

    def _collect(self) -> Tuple[List[Dict[str, Dict]], Dict[str, List]]:
        """
        Collect one step's transition data (environment output and policy output).
        If the environments are finished, then return transitions and statistical data of total agents
        in the whole episode.

        :return return_data: (List[Dict[str, Dict]]) the transitions of total agents in entire games or empty
            if no games end.
        :return collect_log: (Dict[str, List]) the statistical data of the transitions data
        """
        env_outputs = self._env_output if self.selfplay else [self._env_output]  # policy first, then env

        policy_outputs = []
        for policy_id, env_output in enumerate(env_outputs):
            current_policy = self.policies[policy_id]
            policy_output = self.policy_actions_values(current_policy, env_output)  # 策略输出结果

            if current_policy.can_sample_trajectory():  # 添加以作为训练数据
                self._trajectory_buffer.add_policy_data(policy_id, policy_output)

            policy_outputs.append(policy_output)
        policy_outputs = {key: [d[key] for d in policy_outputs] for key in policy_outputs[0]}  # env first, then policy

        # a(t) -> r(t), S(t+1), done(t+1)
        env_commands = self.to_env_commands(policy_outputs)
        raw_env_output = self.env.step(env_commands)
        env_outputs = raw_env_output if self.selfplay else [raw_env_output]

        for policy_id, env_output_ in enumerate(env_outputs):
            current_policy = self.policies[policy_id]
            if current_policy.can_sample_trajectory():  # 添加以作为训练数据
                self._trajectory_buffer.add_env_data(policy_id, env_output_)

            # 统计环境奖励(仅训练策略)
            if current_policy == self.learner_policy:
                for env_id, env_out in enumerate(env_output_):
                    self._env_returns[env_id] += env_out['env_reward']

        return_data, collect_log = [], defaultdict(list)

        done_env_ids = [env_id for env_id, env_output_ in enumerate(env_outputs[0])  # 选取第一个玩家,检查游戏结束状态
                        if all(env_output_['done'])]
        for env_id in done_env_ids:  # 若游戏结束,收集所有agent的transition数据,并返回
            transitions = defaultdict(dict)
            for policy_id, env_output_ in enumerate(env_outputs):
                current_policy = self.policies[policy_id]
                if not current_policy.can_sample_trajectory():
                    continue

                policy_data = self._trajectory_buffer.get_transitions(policy_id, env_id)

                agent_accumulate_reward, max_step = [], 0
                for agent_id, trajectory_data in policy_data.items():
                    returns = self.compute_returns(trajectory_data, next_value=0, use_gae=self.use_gae)
                    transitions[agent_id] = trajectory_data
                    transitions[agent_id].update(returns)  # 添加R(t),Advantage(t)到transition中

                    agent_accumulate_reward.append(sum(trajectory_data['reward']))
                    max_step = max(max_step, len(trajectory_data['reward']))
                return_data.append(transitions)

                if current_policy == self.learner_policy:  # 仅收集训练策略的统计数据
                    collect_log['env_return'].append(self._env_returns[env_id])  # 环境结束,奖励总和
                    self._env_returns[env_id] = 0.0

                    collect_log['accumulate_agent_count'].append(len(policy_data))
                    collect_log['alive_agent_count'].append(len(env_output_[env_id].get('reserved_agent_id',
                                                                                        env_output_[env_id]['agent_id'])))
                    for k, v in synthesize(agent_accumulate_reward).items():
                        collect_log[f'agent_reward_{k}'].append(v)
                    collect_log['step_duration'].append(max_step)
                    is_win = env_output_[env_id]['env_reward'] > 0
                    is_draw = env_output_[env_id]['env_reward'] == 0
                    collect_log['win_count'].append(1 if is_win else 0)  # +1: win
                    collect_log['draw_count'].append(1 if is_draw else 0)  # draw
                    collect_log['lose_count'].append(1 if not is_win and not is_draw else 0)  # -1: lose

        self._env_output = raw_env_output
        return return_data, collect_log

    def to_env_commands(self, policy_outputs: Dict[int, List[Dict[str, EasyDict]]]) -> List[Union[Dict, List[Dict]]]:
        """
        Extract policy outputs' action value and turn to environment acceptable action command.
        :param policy_outputs: Policy outputs of each environments.
        :return policy_commands: Commands that game environment can accept.
        """
        env_commands = []
        for env_id in range(self.n_threads):  # for each env
            env_policy_outputs = policy_outputs[env_id]  #

            policy_commands = []
            for output in env_policy_outputs:  # for each policy's output
                commands = LearnerPolicy.to_env_commands({agent_id: agent_value.action.item()
                                                          for agent_id, agent_value in output.items()})
                policy_commands.append(commands)

            if len(policy_commands) == 1:  # not self-play, just send the first player (no need to send list)
                policy_commands = policy_commands[0]

            env_commands.append(policy_commands)
        return env_commands

    def policy_actions_values(self, policy: BasePolicy,
                              env_output: List[Dict[str, Any]]) -> Dict[int, Dict[str, EasyDict]]:
        """
        Return actions and values predicted by policy according to environment outputs.

        :param policy: (BasePolicy) current policy instance
        :param env_output: (List[Dict[str, Any]]) environment output related to the current policy
        :return: policy_output: (Dict[int, Dict[str, EasyDict]]) policy output for each environments and agents
        """
        agent_ids, obs, available_actions = zip(*[(output.get('reserved_agent_id', output['agent_id']),
                                                   output.get('reserved_obs', output['obs']),
                                                   output.get('reserved_available_actions', output['available_actions']))
                                                  for output in env_output])
        flatten_obs = [value for env_obs in obs for value in env_obs]
        flatten_obs_tensor = torch.from_numpy(np.stack(flatten_obs))
        flatten_available_actions = np.concatenate(available_actions)

        flatten_action, flatten_log_prob = policy.get_actions(flatten_obs_tensor, flatten_available_actions)  # a(t)
        flatten_value = self.learner_policy.get_values(flatten_obs_tensor)  # V(t)

        policy_output = defaultdict(dict)
        c = 0
        for env_id, agent_ids_per_env in enumerate(agent_ids):  # for each env
            for agent_id in agent_ids_per_env:
                policy_output[env_id][agent_id] = EasyDict(dict(
                    obs=flatten_obs[c],
                    action=flatten_action[c],
                    log_prob=flatten_log_prob[c],
                    value=flatten_value[c],
                    available_actions=flatten_available_actions[c],
                ))
                c += 1
        return policy_output

    def compute_returns(self, trajectory: Dict[str, List[Any]], next_value=0, use_gae=False):
        """
        Compute returns and advantages either as discounted sum of rewards, or using GAE.
        :param trajectory: (dict) Agent trajectory data of full steps.
        :param next_value: (float) value predictions for the step after the last episode step.
        :param use_gae: (bool) Use use generalized advantage estimation or not (default True).
        """
        episode_len = len(trajectory['value'])
        gae = 0

        advantages = np.zeros(episode_len)
        returns = np.zeros(episode_len)
        for t in reversed(range(episode_len)):
            next_mask = int(1 - trajectory['done'][t])

            if use_gae:
                next_value = trajectory['value'][t + 1] if t < episode_len - 1 else next_value
                delta = trajectory['reward'][t] + self.gamma * next_value * next_mask - trajectory['value'][t]
                advantages[t] = gae = delta + self.gamma * self.gae_lambda * next_mask * gae
                returns[t] = gae + trajectory['value'][t]
            else:
                next_value = trajectory['reward'][t] + self.gamma * next_value * next_mask
                returns[t] = next_value
                advantages[t] = returns[t] - trajectory['value'][t]

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return {"advantage": advantages, "return_": returns}

    def save(self, episode: int, is_best=False):
        """
        Save runner state dict (models, optimizers and episode) into local file.

        :param episode: (int) Indicates the current episode
        :param is_best: (bool optional) Indicates whether it is a best model or not.
        """
        if not self.save_model_dir.exists():
            os.makedirs(str(self.save_model_dir))

        status = {
            "episode": episode,
        }
        status.update(self.learner_policy.state_dict())

        model_name = f"model_best.pth" if is_best else f"model_{episode}.pth"
        torch.save(status, str(self.save_model_dir / model_name))

    def restore(self, model_path: str, strict=True):
        """
        Restore runner state from model path for training.

        :param model_path: (str) The path of the trained model.
        :param strict: (bool optional) whether to strictly enforce the keys of torch models
        """
        if not model_path:
            return

        model_dict = torch.load(str(model_path), map_location=self.device)

        self.start_episode = int(model_dict['episode']) + 1  # specifies the next episode for training

        self.learner_policy.restore(model_dict, strict)

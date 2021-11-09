from typing import Dict, List, Tuple, Any, AnyStr
import os
from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch
import torch.nn.functional as F

from timer import timer

from utils.utils import calculate_gard_norm, synthesize
from utils.trajectory_buffer import TrajectoryBuffer
from utils.replay_buffer import ReplayBuffer

from algorithms.policy import Policy
from algorithms.training_partner_policy import TrainingPartnerPolicy


class CarbonGameRunner:
    def __init__(self, cfg: EasyDict):
        self._cfg = cfg
        self.env = cfg.env
        self.episodes = cfg.main_config.runner.episodes
        self.episode_length = cfg.main_config.runner.episode_length
        self.n_threads = cfg.main_config.envs.n_threads
        self.training_times = cfg.main_config.runner.training_times
        self.gamma = cfg.main_config.runner.gamma
        self.gae_lambda = cfg.main_config.runner.gae_lambda
        self.clip_epsilon = cfg.main_config.runner.clip_epsilon
        self.entropy_coef = cfg.main_config.runner.entropy_coef
        self.value_loss_coef = cfg.main_config.runner.value_loss_coef
        self.actor_max_grad_norm = cfg.main_config.runner.actor_max_grad_norm
        self.critic_max_grad_norm = cfg.main_config.runner.critic_max_grad_norm
        self.save_interval = 10  # TODO
        self.save_model_dir = cfg.run_dir / "models"

        self.env_output = None
        self.trajectory_buffer = TrajectoryBuffer()
        self._replay_buffer = ReplayBuffer(cfg.main_config.runner.replay_buffer)

        self.learner_policy = Policy(cfg)  # 待训练的策略

        # 下面为selfplay的相关参数
        self.selfplay = True
        self.partner_policy = TrainingPartnerPolicy(cfg, self.save_model_dir)  # 陪练机器人
        self.policies = [self.learner_policy, self.partner_policy]
        self.best_model = None
        self.best_model_filename = None

    def run(self):
        self.env_output = self.env.reset(self.selfplay)

        self.trajectory_buffer.reset()

        per_agent_accumulate_reward = None  # 筛选最佳策略使用
        for episode in range(self.episodes):
            self.prep_rollout()

            collect_logs = {}
            with timer() as t:
                for step in range(self.episode_length):
                    new_data, collect_log = self.collect(step)

                    if new_data:  # add to replay buffer
                        self._replay_buffer.append(new_data)

                        if not collect_logs:
                            collect_logs = collect_log
                        else:
                            for key, value in collect_log.items():
                                collect_logs[key].extend(value)
            # collect_logs = {key: [d[key] for d in collect_logs] for key in collect_logs[0]}
            collect_logs['collect_step_duration'] = [t.elapse / self.episode_length]
            collect_logs = {k: np.mean(v) for k, v in collect_logs.items()}

            train_logs = []
            should_train = True  # TODO
            if should_train:
                self.prep_training()

                with timer() as t:
                    for _ in range(self.training_times):
                        batch_size = 256
                        n_iters = int(np.ceil(len(self._replay_buffer) / batch_size))
                        # train_data = self._replay_buffer.sample_batch(batch_size)  # TODO
                        for i in range(n_iters):
                            start = i * batch_size
                            end = min(start + batch_size, len(self._replay_buffer))
                            train_data = self._replay_buffer.sample_batch_by_indices(np.arange(start, end))  # TODO

                            train_log = self.train(train_data)

                            train_logs.append(train_log)
                train_logs = {key: np.mean([d[key] for d in train_logs]) for key in train_logs[0]}
                train_logs['per_train_time'] = t.elapse / self.training_times

                self._replay_buffer.reset()

            # print(collect_logs)
            # print(train_logs)
            log_value = f"E {episode}/{self.episodes} | " \
                        f"C {collect_logs['alive_agent_count']:.0f}/{collect_logs['accumulate_agent_count']:.0f} | " \
                        f"Win {collect_logs['win_count']:.0f}/{collect_logs['draw_count']:.0f}/" \
                        f"{collect_logs['lose_count']:.0f} | " \
                        f"R {collect_logs['per_agent_accumulate_reward']:.3f} || " \
                        f"V {train_logs['value']:.3f} | " \
                        f"aL {train_logs['actor_loss']:.3f} | vL {train_logs['critic_loss']:.3f} | " \
                        f"∇:ac {train_logs['actor_grad_norm']:.3f} {train_logs['critic_grad_norm']:.3f} | " \
                        f"H {train_logs['entropy']:.3f} | A {train_logs['advantage']:.3f} | " \
                        f"kl {train_logs['approx_kl']:.3f} | r {train_logs['ratio']:.3f}"
            print(log_value)
            if per_agent_accumulate_reward is None or collect_logs['per_agent_accumulate_reward'] >= per_agent_accumulate_reward:
                self.save(episode, is_best=True)
                per_agent_accumulate_reward = collect_logs['per_agent_accumulate_reward']

            if episode % self.save_interval == 0 or episode == self.episodes - 1:
                self.save(episode)

    def collect(self, step) -> Tuple[List[Dict[AnyStr, Dict]], Dict[AnyStr, List]]:
        env_outputs = self.env_output if self.selfplay else [self.env_output]

        policy_outputs = []
        for policy_id, env_output in enumerate(env_outputs):
            policy_output = self.get_actions_and_values(policy_id, env_output)  # 策略输出( TODO: opponent use best model)
            if self.policies[policy_id].can_sample_trajectory():
                self.trajectory_buffer.add_policy_data(policy_id, policy_output)
            policy_outputs.append(policy_output)
        policy_outputs = {key: [d[key] for d in policy_outputs] for key in policy_outputs[0]}  # env first, then policy

        env_actions = []
        for env_id in range(self.n_threads):  # for each env
            policy_output = policy_outputs[env_id]  #

            action = [{agent_id: agent_value.action.item() for agent_id, agent_value in output.items()}
                      for output in policy_output]  # each policy: agent_id: command value

            if len(action) == 1:  # not self-play
                action = action[0]
            env_actions.append(action)

        # a(t) -> r(t), S(t+1), done(t+1)
        raw_env_output = self.env.step(env_actions)
        env_outputs = raw_env_output if self.selfplay else [raw_env_output]
        for policy_id, env_output_ in enumerate(env_outputs):
            if self.policies[policy_id].can_sample_trajectory():
                self.trajectory_buffer.add_env_data(policy_id, env_output_)

        return_data, collect_log = [], defaultdict(list)
        done_env_ids = [env_id for env_id, env_output_ in enumerate(env_outputs[0])  # 选取第一个玩家,检查游戏结束状态
                        if all(env_output_['done'])]
        for env_id in done_env_ids:  # 若游戏结束,收集所有agent的数据,做训练
            transitions = defaultdict(dict)
            for policy_id, env_output_ in enumerate(env_outputs):
                if not self.policies[policy_id].can_sample_trajectory():
                    continue

                policy_data = self.trajectory_buffer.get_transitions(policy_id, env_id)

                agent_accumulate_reward, max_step = [], 0
                for agent_id, trajectory_data in policy_data.items():
                    returns = self.compute_returns(trajectory_data,
                                                   next_value=0)  # TODO: use critic to estimate ???
                    transitions[agent_id] = trajectory_data
                    transitions[agent_id].update(returns)  # 添加R(t),Advantage(t)到transition中

                    agent_accumulate_reward.append(sum(trajectory_data['reward']))
                    max_step = max(max_step, len(trajectory_data['reward']))
                return_data.append(transitions)

                if self.policies[policy_id] == self.learner_policy:  # 仅收集训练策略的统计数据
                    collect_log['accumulate_agent_count'].append(len(policy_data))
                    collect_log['alive_agent_count'].append(len(env_output_[env_id].get('reserved_agent_id',
                                                                                        env_output_[env_id]['agent_id'])))
                    collect_log['per_agent_accumulate_reward'].append(np.mean(agent_accumulate_reward))
                    collect_log['env_return'].append(env_output_[env_id]['env_reward'])
                    collect_log['step_duration'].append(max_step)
                    is_win = env_output_[env_id]['env_reward'] > 0
                    is_draw = env_output_[env_id]['env_reward'] == 0
                    collect_log['win_count'].append(1 if is_win else 0)  # +1: win
                    collect_log['draw_count'].append(1 if is_draw else 0)  # draw
                    collect_log['lose_count'].append(1 if not is_win and not is_draw else 0)  # -1: lose

        if 'win_count' in collect_log:
            collect_log['win_count'] = [sum(collect_log['win_count'])]
            collect_log['draw_count'] = [sum(collect_log['draw_count'])]
            collect_log['lose_count'] = [sum(collect_log['lose_count'])]
        self.env_output = raw_env_output
        return return_data, collect_log

    def get_actions_and_values(self, policy_id: int, env_output):
        agent_ids, obs, available_actions = zip(*[(output.get('reserved_agent_id', output['agent_id']),
                                                   output.get('reserved_obs', output['obs']),
                                                   output.get('reserved_available_actions', output['available_actions']))
                                                  for output in env_output])
        flatten_obs = [value for env_obs in obs for value in env_obs]
        flatten_obs_tensor = torch.from_numpy(np.stack(flatten_obs))
        flatten_available_actions = np.concatenate(available_actions)

        policy_output = self.policies[policy_id].get_actions_values(flatten_obs_tensor, flatten_available_actions)

        flatten_action, flatten_log_prob, flatten_value = policy_output  # a(t), V(t)

        output = defaultdict(dict)
        c = 0
        for env_id, agent_ids_per_env in enumerate(agent_ids):
            for agent_id in agent_ids_per_env:
                output[env_id][agent_id] = EasyDict(dict(
                    obs=flatten_obs[c],
                    action=flatten_action[c],
                    log_prob=flatten_log_prob[c],
                    value=flatten_value[c],
                    available_actions=flatten_available_actions[c],
                ))
                c += 1
        return output

    def train(self, batch: EasyDict) -> EasyDict:
        log_prob, dist_entropy, value = self.learner_policy.evaluate_actions(batch.obs, batch.action, batch.available_actions)

        # actor loss
        ratio = torch.exp(log_prob - batch.log_prob)
        advantage = batch.advantage
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        actor_loss = policy_loss - self.entropy_coef * dist_entropy

        # value loss
        experience_value = batch.value
        experience_return = batch.return_
        value = value.reshape_as(batch.value)
        value_clipped = experience_value + torch.clamp(value - experience_value, -self.clip_epsilon, self.clip_epsilon)
        value_loss = F.mse_loss(value, experience_return)
        value_clipped_loss = F.mse_loss(value_clipped, experience_return)
        value_loss = torch.max(value_loss, value_clipped_loss).mean()
        critic_loss = self.value_loss_coef * value_loss

        with torch.no_grad():
            log_ratio = log_prob - batch.log_prob
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

        # train
        self.learner_policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = calculate_gard_norm(self.learner_policy.actor_model.parameters())
        if self.actor_max_grad_norm is not None and self.actor_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.learner_policy.actor_model.parameters(), self.actor_max_grad_norm)
        self.learner_policy.actor_optimizer.step()

        self.learner_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = calculate_gard_norm(self.learner_policy.critic_model.parameters())
        if self.critic_max_grad_norm is not None and self.critic_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.learner_policy.critic_model.parameters(), self.critic_max_grad_norm)
        self.learner_policy.critic_optimizer.step()

        # 返回统计情况
        output = EasyDict({
            "entropy": dist_entropy.mean().item(),
            "policy_loss": policy_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_grad_norm": actor_grad_norm,
            "advantage": advantage.mean().item(),
            "approx_kl": approx_kl.item(),
            "value": value.mean().item(),
            "critic_loss": critic_loss.item(),
            "critic_grad_norm": critic_grad_norm,
            "ratio": ratio.mean().item(),
        })
        return output

    def compute_returns(self, trajectory: Dict[AnyStr, List[Any]], next_value=0, use_gae=True):
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

        return {"advantage": advantages, "return_": returns}

    def prep_training(self):
        self.learner_policy.actor_model.train()
        self.learner_policy.critic_model.train()

    def prep_rollout(self):
        self.learner_policy.actor_model.eval()
        self.learner_policy.critic_model.eval()

        self.partner_policy.policy_reset()

    def save(self, episode: int, is_best=False):
        actor_name = f"actor_best.pth" if is_best else f"actor_{episode}.pth"
        critic_name = f"critic_best.pth" if is_best else f"critic_{episode}.pth"
        if not self.save_model_dir.exists():
            os.makedirs(str(self.save_model_dir))
        torch.save(self.learner_policy.actor_model.state_dict(), str(self.save_model_dir / actor_name))
        torch.save(self.learner_policy.critic_model.state_dict(), str(self.save_model_dir / critic_name))

from collections import defaultdict

from torch.multiprocessing import Process, Pipe
import gym


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            env_output = env.step(data)
            # env_done = env_output.done if isinstance(env_output.done, bool) else all(env_output.done)
            # if env_done:
            #     obs = env.reset()
            conn.send(env_output)
        elif cmd == "reset":
            env_output = env.reset(data)
            conn.send(env_output)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        # self.observation_space = self.envs[0].observation_space
        # self.action_space = self.envs[0].action_space
        self.selfplay = False

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self, selfplay=False):
        self.selfplay = selfplay

        players = [None, None] if self.selfplay else None
        for local in self.locals:
            local.send(("reset", players))
        one_output = self.envs[0].reset(players)
        other_outputs = [local.recv() for local in self.locals]
        total_outputs = [one_output] + other_outputs

        if self.selfplay:  # 拆分出player1, player2的数据
            total_outputs = [[v[i] for v in total_outputs] for i, _ in enumerate(total_outputs[0])]
        # results = {key: [d[key] for d in total_outputs] for key in total_outputs[0]}  # list dict to dict list
        return total_outputs

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        one_output = self.envs[0].step(actions[0])
        total_outputs = [one_output] + [local.recv() for local in self.locals]

        if self.selfplay:  # 拆分出player1, player2的数据
            player1_outputs, player2_outputs = [[v[i] for v in total_outputs] for i, _ in enumerate(total_outputs[0])]
            self._reset_if_env_done(player1_outputs, player2_outputs)
            total_outputs = [player1_outputs, player2_outputs]
        else:
            self._reset_if_env_done(total_outputs, None)

        return total_outputs

    def _reset_if_env_done(self, player1_outputs, player2_outputs=None):
        player2_outputs = [None] * len(player1_outputs) if player2_outputs is None else player2_outputs
        for env_id, (output1, output2) in enumerate(zip(player1_outputs, player2_outputs)):  # 遍历每个并行环境的输出
            env_done = all(output1['done'])
            if env_done:
                players = [None, None] if self.selfplay else None
                if env_id == 0:
                    reset_output = self.envs[0].reset(players)
                else:
                    self.locals[env_id - 1].send(("reset", players))
                    reset_output = self.locals[env_id - 1].recv()

                if isinstance(reset_output, list):
                    # 要替换的数据添加reserved_前缀
                    reserved_output = {f"reserved_{k}": output1[k] for k in reset_output[0].keys()}
                    output1.update(reserved_output)
                    output1.update(reset_output[0])  # 替换掉output中的 agent_id, obs, available_actions

                    reserved_output = {f"reserved_{k}": output2[k] for k in reset_output[1].keys()}
                    output2.update(reserved_output)
                    output2.update(reset_output[1])  # 替换掉output中的 agent_id, obs, available_actions
                else:
                    reserved_output = {f"reserved_{k}": output1[k] for k in reset_output.keys()}
                    output1.update(reserved_output)
                    output1.update(reset_output)  # 替换掉output中的 agent_id, obs, available_actions

    def render(self, mode="human"):
        raise NotImplementedError

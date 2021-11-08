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
            env_output = env.reset()
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
        total_outputs = [one_output] + [local.recv() for local in self.locals]

        # results = {key: [d[key] for d in total_outputs] for key in total_outputs[0]}  # list dict to dict list
        return total_outputs

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        one_output = self.envs[0].step(actions[0])
        total_outputs = [one_output] + [local.recv() for local in self.locals]

        for env_id, outputs in enumerate(total_outputs):
            if not isinstance(outputs, list):  # If not selfplay
                outputs = [outputs]

            for output in outputs:
                if all(output['done']):
                    players = [None, None] if self.selfplay else None
                    if env_id == 0:
                        reset_output = self.envs[0].reset(players)
                    else:
                        self.locals[env_id - 1].send(("reset", players))
                        reset_output = self.locals[env_id - 1].recv()
                    reserved_output = {f"reserved_{k}": output[k] for k in reset_output.keys()}  # 要替换的数据添加reserved_前缀
                    output.update(reserved_output)
                    output.update(reset_output)  # 替换掉output中的 agent_id, obs, available_actions
        # results = {key: [d[key] for d in total_outputs] for key in total_outputs[0]}  # list dict to dict list
        #
        # for env_id, env_dones in enumerate(results['done']):  # 如果游戏结束，重新开始
        #     if all(env_dones):  # 检查游戏是否结束(智能体全部结束)
        #         if env_id == 0:
        #             reset_output = self.envs[0].reset()
        #         else:
        #             self.locals[env_id - 1].send(("reset", None))
        #             reset_output = self.locals[env_id - 1].recv()
        #         for k, v in reset_output.items():  # (obs, agent_ids)
        #             results[k][env_id] = v
        return total_outputs

    def render(self, mode="human"):
        raise NotImplementedError

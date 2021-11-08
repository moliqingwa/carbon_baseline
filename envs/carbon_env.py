import time
import random
import envs.rendering as rendering
import numpy as np
import math
from io import StringIO
from collections import defaultdict

from game.zerosum_env import make, InvalidArgument, Agent
from game.zerosum_env.core import act_agent


class AgentRunner:
    def __init__(self, env, agents):
        self.env = env

        # Generate the players.
        self.agents = [None if agent is None else Agent(agent, self.env) for agent in agents]

    def act(self, none_actions=None):
        if len(self.agents) != len(self.env.states):
            raise InvalidArgument("Number of players must match the state length")
        if none_actions is not None and len(none_actions) != len(self.agents):
            raise InvalidArgument("Number of players must match the none_actions argument's length")

        act_args = [
            (
                agent,
                self.env._Environment__get_shared_state(i),
                self.env.configuration,
                none_actions[i] if none_actions is not None else None,
            )
            for i, agent in enumerate(self.agents)
        ]

        results = list(map(act_agent, act_args))

        # results is a list of tuples where the first element is an agent action and the second is the agent log
        # This destructures into two lists, a list of actions and a list of logs.
        actions, logs = zip(*results)
        return list(actions), list(logs)


class CarbonEnv:
    def __init__(self, cfg: dict):
        self.env = make("carbon",
                        configuration=cfg,
                        debug=True)

        self._runner = None

        self.self_play = False
        self.my_index = None
        self.opponent_index = None

        if self.self_play:
            self.players = [None, None]
        else:
            self.players = [None, "random"]

        self.reset(self.players)

        self.pixel_size = 30
        self.viewer = None
        self.board_lines = []

    def reset(self, players=None):
        players = self.players if players is None else players
        if self.self_play:
            assert players == [None, None]
            self.my_index, self.opponent_index = 0, 1
        else:
            for index, agent in enumerate(players):
                if agent is None:
                    if self.my_index is not None and self.my_index > 0:
                        raise InvalidArgument("Only one agent can be marked 'None'")
                    self.my_index = index
        self.players = players
        self.env.reset(len(self.players))
        self._runner = AgentRunner(self.env, self.players)
        self._advance()
        observation = self.env._Environment__get_shared_state(self.my_index).observation
        return observation

    def _advance(self):
        while not self.env.done and self.env.states[self.my_index].status == "INACTIVE":
            actions, logs = self._runner.act()
            self.env.step(actions, logs)

    def step(self, action):
        actions, logs = self._runner.act(action)
        self.env.step(actions, logs)
        self._advance()

    def get_state(self, index):
        return self.env._Environment__get_shared_state(index)

    def close(self):
        pass

    def render(self, mode='human'):
        raise NotImplementedError("not implemented!")

import gym
from gym_minigrid import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env


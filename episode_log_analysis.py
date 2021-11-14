
import json

from zerosum_env.envs.carbon.helpers import Board
from zerosum_env import make

# 构建一个环境
env = make("carbon", debug=True, configuration={})

total_data = []
with open("episode.txt", 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        total_data.append(data)


def get_step_obs(total_data, step):
    raw_obs = total_data[step][0]['observation']
    next_actions = [v['action'] for v in total_data[step + 1]]
    return Board(raw_obs, env.configuration, next_actions)


step = 118
current_obs = get_step_obs(total_data, step)
next_obs = current_obs.next()

pass

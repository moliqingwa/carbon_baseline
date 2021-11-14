
import glob
import itertools
import json
import random

import torch

from algorithms.eval_policy import EvalPolicy

from zerosum_env import evaluate
from zerosum_env.envs.carbon.helpers import *


if __name__ == "__main__":
    all_paths = glob.glob("scripts/train/runs/run12/models/*.pth")
    all_paths2 = glob.glob("scripts/train/runs/run13/models/*.pth")

    total_paths = all_paths + all_paths2

    p1 = EvalPolicy()
    p2 = EvalPolicy()

    def evaluate_agent(p1: EvalPolicy, p2: EvalPolicy):
        rew, _, _, _ = evaluate(
            "carbon",
            agents=[p1.take_action, p2.take_action],
            configuration={"randomSeed": 1},
            debug=True)
        return rew[0][0], rew[0][1]

    exclude_paths = {
        'scripts/train/runs/run12/models/model_720.pth',
    }
    total_paths = [p for p in total_paths if p not in exclude_paths]

    try:
        with open("paths.txt", 'r') as f:
            total_paths = json.loads(f.read())
    except:
        pass

    while len(total_paths) >= 2:
        print(len(total_paths))
        random.shuffle(total_paths)

        with open("paths.txt", 'w') as f:
            f.write(json.dumps(total_paths))

        for path1, path2 in itertools.product(total_paths, total_paths):
            if path1 == path2:
                continue
            p1.restore(torch.load(path1))
            p2.restore(torch.load(path2))
            r11, r12 = evaluate_agent(p1, p2)
            r21, r22 = evaluate_agent(p2, p1)

            if r11 is None and r12 is not None:  # 2# win
                total_paths.remove(path1)
                break
            elif r11 is not None and r12 is None:  # 1# win
                total_paths.remove(path2)
                break
            elif r11 is None and r12 is None:
                print(r11)
                print(r12)
                pass
            elif r11 > r12 and r21 > r22:  # 1# win
                total_paths.remove(path2)
                break
            elif r11 < r12 and r21 < r22:  # 2# win
                total_paths.remove(path1)
                break

    print(total_paths)

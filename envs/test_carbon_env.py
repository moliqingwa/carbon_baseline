from envs.carbon_env import CarbonEnv


class TestCarbonEnv:
    def test_a(self):
        env = CarbonEnv({})

        env.reset([None, "random"])
        print(env.my_index)  # 输出 0

        env.reset(["random", None])
        print(env.my_index)  # 输出 1
        obs = env.get_state(env.my_index)
        pass


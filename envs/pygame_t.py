import os
import pygame
import numpy as np
import math
import time


class CarbonGymEnv:

    def __init__(self, obs, screen, dir):

        self.obs = obs
        self.screen = screen
        self.dir = dir

        # 图片
        self.bg_img = pygame.image.load(self.dir + "/bg.png")
        self.col1 = pygame.image.load(self.dir + '/col1.png').convert_alpha()
        self.col2 = pygame.image.load(self.dir + '/col2.png').convert_alpha()
        self.house1 = pygame.image.load(self.dir + '/house1.png').convert_alpha()
        self.house2 = pygame.image.load(self.dir + '/house2.png').convert_alpha()
        self.planter1 = pygame.image.load(self.dir + '/planter1.png').convert_alpha()
        self.planter2 = pygame.image.load(self.dir + '/planter2.png').convert_alpha()
        self.tree1 = pygame.image.load(self.dir + '/tree1.png').convert_alpha()
        self.tree2 = pygame.image.load(self.dir + '/tree2.png').convert_alpha()

        self.info1 = pygame.image.load(self.dir + "/info1.png")
        self.info2 = pygame.image.load(self.dir + "/info2.png")
        self.tab1 = pygame.image.load(self.dir + "/tab1.png")
        self.tab2 = pygame.image.load(self.dir + "/tab2.png")
        # 字体
        self.ft = pygame.font.Font("freesansbold.ttf", 13)

        self.pixel_size = 30
        self.grid_size = 15
        self.co_color = (0, 0, 0)

        # 修改图片尺寸
        self.bg_img_new = pygame.transform.scale(self.bg_img, (800, 800))
        self.col1_new = pygame.transform.scale(self.col1, (16, 16))
        self.col2_new = pygame.transform.scale(self.col2, (16, 16))
        self.house1_new = pygame.transform.scale(self.house1, (20, 20))
        self.house2_new = pygame.transform.scale(self.house2, (20, 20))
        self.planter1_new = pygame.transform.scale(self.planter1, (16, 16))
        self.planter2_new = pygame.transform.scale(self.planter2, (16, 16))
        self.tree1_new = pygame.transform.scale(self.tree1, (24, 24))
        self.tree2_new = pygame.transform.scale(self.tree2, (24, 24))
        self.info1_new = pygame.transform.scale(self.info1, (70, 40))
        self.info2_new = pygame.transform.scale(self.info2, (70, 40))
        self.tab1_new = pygame.transform.scale(self.tab1, (40, 40))
        self.tab2_new = pygame.transform.scale(self.tab2, (40, 40))

    # 画背景和框
    def draw_border(self):
        # 背景图和队伍信息图片添加
        self.screen.blit(self.bg_img_new, (0, 0))
        self.screen.blit(self.tab1_new, (15, 200))
        self.screen.blit(self.tab2_new, (15, 450))
        self.screen.blit(self.info1_new, (15, 250))
        self.screen.blit(self.info2_new, (15, 500))

        # 添加坐标
        for i in range(16):
            pygame.draw.line(self.screen, (0, 0, 0), (175, 175 + self.pixel_size * i), (625, 175 + self.pixel_size * i))
            pygame.draw.line(self.screen, (0, 0, 0), (175 + self.pixel_size * i, 175), (175 + self.pixel_size * i, 625))

    # 画碳
    def draw_carbon(self):
        carbonList = self.obs["carbon"]
        carbonNumpy = np.array(carbonList).reshape((self.grid_size, self.grid_size))
        c_color = (0, 0, 0)
        for y in range(1, 16):
            for x in range(1, 16):
                x_new = 175 + self.grid_size * (x * 2 - 1)
                y_new = 175 + self.grid_size * (y * 2 - 1)
                pos = x_new, y_new
                radius = math.sqrt((carbonNumpy[x - 1, y - 1])) * 1.5 / math.sqrt(math.pi)
                pygame.draw.circle(self.screen, c_color, pos, radius, 1)

    # 画工人
    def draw_worker(self, work_v):
        col_x, col_y = divmod(work_v[0], self.grid_size)
        col_x_new = 182 if col_x == 0 else 182 + (col_x - 1) * self.pixel_size
        col_y_new = 182 if col_y == 0 else 182 + (col_y - 1) * self.pixel_size
        return col_x_new, col_y_new

    # 画基地
    def draw_base(self, recrt_x, recrt_y):
        recrt_x_new = 180 if recrt_x == 0 else 180 + (recrt_x - 1) * self.pixel_size
        recrt_y_new = 180 if recrt_y == 0 else 180 + (recrt_y - 1) * self.pixel_size

        return recrt_x_new, recrt_y_new

    # 画树
    def draw_trees(self, tree_v):
        tree_x, tree_y = divmod(tree_v[0], self.grid_size)
        tree_x_new = 178 if tree_x == 0 else 178 + (tree_x - 1) * self.pixel_size
        tree_y_new = 178 if tree_y == 0 else 178 + (tree_y - 1) * self.pixel_size

        return tree_x_new, tree_y_new

    def render(self):

        self.draw_border()
        self.draw_carbon()
        for team_info in self.obs["players"]:
            my_base_list = self.obs["players"][0]
            obse_base_list = self.obs["players"][1]

            if team_info == my_base_list:
                base_list = my_base_list
                house = self.house1_new
                player = "player-0-recrtCenter-0"
                col = self.col1_new
                planter = self.planter1_new
                tree = self.tree1_new
                fund = 100.23
                tree_ct = 0
                plant_ct = 1
                col_ct = 0
                co_ct = 327.47
                ft_y = 295
            else:
                base_list = obse_base_list
                player = "player-1-recrtCenter-0"
                house = self.house2_new
                col = self.col2_new
                planter = self.planter2_new
                tree = self.tree2_new
                fund = 200.23
                tree_ct = 2
                plant_ct = 1
                col_ct = 1
                co_ct = 507.47
                ft_y = 545
            ft_x = 16
            # 队伍详细信息添加
            fund = self.ft.render(f"cash:%.2f" % fund, True, (0, 0, 0))
            tree_ct = self.ft.render("tree:%d" % tree_ct, True, (0, 0, 0))
            plant_ct = self.ft.render("planter:%d" % plant_ct, True, (0, 0, 0))
            col_ct = self.ft.render("collector:%d" % col_ct, True, (0, 0, 0))
            co_ct = self.ft.render("CO2(on the way):%.2f" % co_ct, True, (0, 0, 0))

            self.screen.blit(fund, (ft_x, ft_y))
            self.screen.blit(tree_ct, (ft_x, ft_y + 13))
            self.screen.blit(plant_ct, (ft_x, ft_y + 26))
            self.screen.blit(col_ct, (ft_x, ft_y + 39))
            self.screen.blit(co_ct, (ft_x, ft_y + 52))

            # 基地
            recrt_x, recrt_y = divmod(base_list[1][player], self.grid_size)
            recrt_x_new, recrt_y_new = self.draw_base(recrt_x, recrt_y)
            self.screen.blit(house, (recrt_x_new, recrt_y_new))
            # 工人
            for work_k, work_v in base_list[2].items():
                col_x_new, col_y_new = self.draw_worker(work_v)
                # 捕碳人
                if work_v[2] == 'COLLECTOR':
                    self.screen.blit(col, (col_x_new, col_y_new))
                # 种树人
                else:
                    self.screen.blit(planter, (col_x_new, col_y_new))

            # 画树
            for tree_k, tree_v in base_list[3].items():
                tree_x_new, tree_y_new = self.draw_trees(tree_v)
                self.screen.blit(tree, (tree_x_new, tree_y_new))


if __name__ == "__main__":

    obsList = [{'carbon': [13.525, 0.0, 2.081, 3.121, 19.768, 0.0, 1.04, 5.202, 1.04, 0.0, 19.768, 3.121, 2.081, 0.0,
                           13.525, 0.0, 11.444, 0.0, 6.242, 0.0, 33.293, 0.0, 0.0, 0.0, 33.293, 0.0, 6.242, 0.0, 11.444,
                           0.0, 0.0, 0.0, 30.172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.172, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 2.081, 0.0, 2.081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.646,
                           0.0, 0.0, 0.0, 3.121, 11.444, 3.121, 0.0, 0.0, 0.0, 16.646, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0, 0.0, 0.0, 0.0, 13.525, 0.0, 0.0, 0.0, 0.0, 0.0, 15.606, 0.0, 16.646, 0.0, 0.0, 3.121,
                           0.0, 40.576, 0.0, 3.121, 0.0, 0.0, 16.646, 0.0, 15.606, 0.0, 0.0, 28.091, 8.323, 0.0, 0.0,
                           14.566, 0.0, 14.566, 0.0, 0.0, 8.323, 28.091, 0.0, 0.0, 15.606, 0.0, 16.646, 0.0, 0.0, 3.121,
                           0.0, 40.576, 0.0, 3.121, 0.0, 0.0, 16.646, 0.0, 15.606, 0.0, 0.0, 0.0, 0.0, 0.0, 13.525, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.646, 0.0, 0.0, 0.0, 3.121, 11.444,
                           3.121, 0.0, 0.0, 0.0, 16.646, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.081, 0.0, 2.081, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           30.172, 0.0, 0.0, 0.0, 11.444, 0.0, 6.242, 0.0, 33.293, 0.0, 50.0, 0.0, 33.293, 0.0, 6.242,
                           0.0, 11.444, 0.0, 13.525, 0.0, 2.081, 3.121, 19.768, 0.0, 1.04, 5.202, 1.04, 0.0, 19.768,
                           3.121, 2.081, 0.0, 13.525], 'players': [
        [20, {'player-0-recrtCenter-0': 80}, {'player-0-worker-0': [40, 0, 'PLANTER']}, {'my-tree-01': [47, 52]}],
        [20, {'player-1-recrtCenter-0': 144}, {}, {}]], 'player': 0, 'step': 2, 'trees': {},
                'remainingOverageTime': 60}, {
                   'carbon': [13.525, 0.0, 2.081, 3.121, 19.768, 0.0, 1.04, 5.202, 1.04, 0.0, 19.768, 3.121, 2.081, 0.0,
                              13.525, 0.0, 11.444, 0.0, 6.242, 0.0, 33.293, 0.0, 0.0, 0.0, 33.293, 0.0, 6.242, 0.0,
                              11.444, 0.0, 0.0, 0.0, 30.172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.172, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.081, 0.0, 2.081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 16.646, 0.0, 0.0, 0.0, 3.121, 11.444, 3.121, 0.0, 0.0, 0.0, 16.646, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 13.525, 0.0, 0.0, 0.0, 0.0, 0.0, 15.606, 0.0,
                              16.646, 0.0, 0.0, 3.121, 0.0, 40.576, 0.0, 3.121, 0.0, 0.0, 16.646, 0.0, 15.606, 0.0, 0.0,
                              28.091, 8.323, 0.0, 0.0, 14.566, 0.0, 14.566, 0.0, 0.0, 8.323, 28.091, 0.0, 0.0, 15.606,
                              0.0, 16.646, 0.0, 0.0, 3.121, 0.0, 40.576, 0.0, 3.121, 0.0, 0.0, 16.646, 0.0, 15.606, 0.0,
                              0.0, 0.0, 0.0, 0.0, 13.525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.646,
                              0.0, 0.0, 0.0, 3.121, 11.444, 3.121, 0.0, 0.0, 0.0, 16.646, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 2.081, 0.0, 2.081, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.172, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.172, 0.0, 0.0, 0.0, 11.444, 0.0, 6.242, 0.0, 33.293,
                              0.0, 50.0, 0.0, 33.293, 0.0, 6.242, 0.0, 11.444, 0.0, 13.525, 0.0, 2.081, 3.121, 19.768,
                              0.0, 1.04, 5.202, 1.04, 0.0, 19.768, 3.121, 2.081, 0.0, 13.525], 'players': [
            [20, {'player-0-recrtCenter-0': 80}, {'player-0-worker-0': [47, 0, 'PLANTER']}, {'my-tree-01': [62, 52]}],
            [20, {'player-1-recrtCenter-0': 144}, {}, {}]], 'player': 0, 'step': 2, 'trees': {},
                   'remainingOverageTime': 60}]
    dir = os.getcwd() + "/img"
    # dir = '/Users/zhenwang/code/spdb/carbon-challenge/carbon-rl-baseline/game/zerosum_env/static/src'
    pygame.init()
    # 设置窗口大小
    screen = pygame.display.set_mode((800, 800))

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for obs in obsList:
            gym = CarbonGymEnv(obs, screen, dir)
            gym.render()

            pygame.display.flip()
            time.sleep(1)

import json
from collections import namedtuple
from enum import Enum
from functools import partial
import random
from pathlib import Path

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle
# from gym.envs.classic_control import rendering

# A good example is https://github.com/openai/gym-soccer

""" Maybe a more detailed description of the environment here"""

FPS = 60

VIEWPORT_W = 600
VIEWPORT_H = 600

Rewards = namedtuple('Rewards', ['bump_into_wall',
                                 'walking',
                                 'failed_action',
                                 'completed_task'])
Reward = Rewards(-3., -.1, -1, 100.)


def print_vision_grid(grid):
    # DEBUG only - prints the vision grid in a squared format
    bot = np.flip(np.reshape(grid[0:21], (3, 7)), 0)
    mid = np.hstack((grid[21:24], np.array(-1), grid[24:27]))
    top = np.flip(np.reshape(grid[27:], (3, 7)), 0)
    print(np.vstack((top, mid, bot)))


class Tasks(Enum):
    """
    Tasks values are tuples that consist of the id, and the combination of actions
    required to solve the task.
    """
    MAKE_TEA = 1
    MAKE_SOUP = 2
    MAKE_PASTA = 3
    CLEAN_STOVE = 4
    MAKE_OMELETTE = 5
    MAKE_PANCAKES = 6

    @staticmethod
    def to_binary_list(x, vec_len=5):
        res = [int(i) for i in bin(x)[2:]]
        return np.pad(res, (vec_len - len(res), 0))

    @staticmethod
    def to_dec(x):
        res = 0
        for ele in x:
            res = (res << 1) | ele
        return res


class ObjectColors:
    path = Path(__file__).parents[1] / 'colors_house_objects.json'
    with open(path) as json_file:
        colors = json.load(json_file)

    @staticmethod
    def get_color(object_name):
        return ObjectColors.colors[object_name]


class HouseholdEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        EzPickle.__init__(self)  # TODO: not sure if this is still needed...
        self.seed()  # TODO: not sure if this is still needed...
        self.viewer = None
        # Define our grid map dimensions
        self.map_height = 20
        self.map_width = 20
        self.scale = VIEWPORT_W / self.map_width
        # observation space
        self.robot_pos = (0, 0)
        self.task_to_do = Tasks.to_binary_list(0)
        self.vision_grid = np.zeros(48)
        self.states = {'cabinet_open': 0, 'has_tea': 0, 'has_soup_jar': 0, 'fire_on': 0, 'tap_open': 0,
                       'holding_saucepan': 0, 'saucepan_full': 0, 'heated_up': 0, 'has_boiling_water': 0,
                       'has_cleaning_cloth': 0, 'has_cleaning_product': 0, 'stove_cleaned': 0,
                       'has_pasta': 0, 'has_sauce': 0, 'pasta_drained': 0, 'item_cooked': 0,
                       'has_eggs': 0, 'has_milk': 0, 'has_pancake_mix': 0, 'holding_frying_pan': 0,
                       'whisked': 0, 'flipped': 0}

        self.action_dict = {0: self._move_up,
                            1: self._move_down,
                            2: self._move_left,
                            3: self._move_right,
                            4: self._open_door,
                            5: self._close_door,
                            6: self._get_tea,
                            7: self._get_soup_jar,
                            8: self._get_saucepan,
                            9: self._get_cleaning_cloth,
                            10: self._get_cleaning_product,
                            11: self._get_pasta,
                            12: self._get_sauce,
                            13: self._turn_on_heat,
                            14: self._turn_off_heat,
                            15: self._heat_up,
                            16: self._open_tap,
                            17: self._close_tap,
                            18: self._fill,
                            19: self._rinse_and_dry,
                            20: self._scrub,
                            21: self._drain,
                            22: self._mix,
                            23: self._get_eggs,
                            24: self._get_milk,
                            25: self._get_pancake_mix,
                            26: self._get_frying_pan,
                            27: self._whisk,
                            28: self._flip
                            }
        print("HPC version")
        self.reset()

        # Min-Max values for states
        low = np.zeros(29)  # 2(pos) + 22 + 5 task enc
        high = np.hstack((np.array([19, 19]), np.ones(27)))
        self.action_space = spaces.Discrete(29)
        self.observation_space = spaces.Box(low, high, dtype=np.int)

    def __del__(self):
        pass

    def _generate_house(self):
        path = Path(__file__).parents[1] / 'house_objects.json'
        with open(path) as json_file:
            house_objects = json.load(json_file)
        # All the objects that the robot might collide with, so its easier to see if it can move without colliding
        self.colliding_objects = set()
        for key, values in house_objects.items():
            self.house_objects_id[key] = values[0]
            values = [tuple(x) for x in values[1:]]
            self.house_objects[key] = values
            self.colliding_objects = self.colliding_objects.union(values)
        path = Path(__file__).parents[1] / 'operability.json'
        with open(path) as json_file:
            aux = json.load(json_file)
        self.operability = {}
        for key in aux.keys():
            self.operability[key] = [tuple(i) for i in aux[key]]

    def _move_up(self):
        x, y = self.robot_pos
        new_pos = (x, y + 1)
        return self._move(new_pos, restriction=new_pos[1] >= self.map_height)

    def _move_down(self):
        x, y = self.robot_pos
        new_pos = (x, y - 1)
        return self._move(new_pos, restriction=new_pos[1] < 0)

    def _move_left(self):
        x, y = self.robot_pos
        new_pos = (x - 1, y)
        # Just for reducing search space
        if new_pos[0] < 11:
            return Reward.bump_into_wall
        return self._move(new_pos, restriction=new_pos[0] < 0)

    def _move_right(self):
        x, y = self.robot_pos
        new_pos = (x + 1, y)
        return self._move(new_pos, restriction=new_pos[0] >= self.map_width)

    def _move(self, new_pos, restriction):
        # Check if it collides with an object
        if new_pos in self.colliding_objects:
            return Reward.bump_into_wall
        if restriction:
            return Reward.bump_into_wall
        self.robot_pos = new_pos
        return Reward.walking

    def _open_door(self):
        if (self.robot_pos in self.operability['cabinet']) and not self.states['cabinet_open']:
            self.states['cabinet_open'] = 1
            # print("Door opened.")
        else:
            return Reward.failed_action
        return 0

    def _close_door(self):
        if (self.robot_pos in self.operability['cabinet']) and self.states['cabinet_open']:
            self.states['cabinet_open'] = 0
            # print("Door closed.")
        else:
            return Reward.failed_action
        return 0

    def _get_tea(self):
        if self.robot_pos in self.operability['cabinet'] and not self.states['has_tea'] and self.states['cabinet_open']:
            self.states['has_tea'] = 1
            # print("Got tea.")
        else:
            return Reward.failed_action
        return 0

    def _get_soup_jar(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_soup_jar']
                and self.states['cabinet_open']):
            self.states['has_soup_jar'] = 1
            # print("Got soup jar.")
        else:
            return Reward.failed_action
        return 0

    def _get_cleaning_cloth(self):
        if self.robot_pos in self.operability['sink'] and not self.states['has_cleaning_cloth']:
            self.states['has_cleaning_cloth'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_cleaning_product(self):
        if self.robot_pos in self.operability['sink'] and not self.states['has_cleaning_product']:
            self.states['has_cleaning_product'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_pasta(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_pasta']
                and self.states['cabinet_open']):
            self.states['has_pasta'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_sauce(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_sauce']
                and self.states['cabinet_open']):
            self.states['has_sauce'] = 1
        else:
            return Reward.failed_action
        return 0

    def _scrub(self):
        if (self.robot_pos in self.operability['stove'] and not self.states['stove_cleaned']
                and self.states['has_cleaning_cloth'] and self.states['has_cleaning_product']):
            self.states['stove_cleaned'] = 1
        else:
            return Reward.failed_action
        return 0

    def _rinse_and_dry(self):
        # Not quite elegant but it should work..
        if (Tasks.to_dec(self.task_to_do) == Tasks.CLEAN_STOVE.value and
                self.robot_pos in self.operability['sink'] and self.states['has_cleaning_cloth'] and
                self.states['tap_open'] and self.states['stove_cleaned']):
            self.task_done = True
        else:
            return Reward.failed_action
        return 0

    def _drain(self):
        if (self.robot_pos in self.operability['sink'] and not self.states['pasta_drained']
                and self.states['item_cooked']):
            self.states['pasta_drained'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_saucepan(self):
        if (self.robot_pos in self.operability['sink']) and not self.states['holding_saucepan']:
            self.states['holding_saucepan'] = 1
            # print("Got saucepan")
        else:
            return Reward.failed_action
        return 0

    def _turn_on_heat(self):
        if (self.robot_pos in self.operability['stove']) and not self.states['fire_on']:
            self.states['fire_on'] = 1
            # print("Fire turned on")
        else:
            return Reward.failed_action
        return 0

    def _turn_off_heat(self):
        if self.robot_pos in self.operability['stove'] and self.states['fire_on']:
            self.states['fire_on'] = 0
            # print("Fire turned off")
        else:
            return Reward.failed_action
        return 0

    def _heat_up(self):
        common = self.robot_pos in self.operability['stove'] and self.states['fire_on'] and not self.states['heated_up']
        cond_tea = Tasks.to_dec(self.task_to_do) == Tasks.MAKE_TEA.value and self.states['saucepan_full']
        cond_soup = Tasks.to_dec(self.task_to_do) == Tasks.MAKE_SOUP.value and self.states['saucepan_full']
        cond_pasta = Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PASTA.value and self.states['saucepan_full']
        cond_omelette = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_OMELETTE.value and self.states['whisked']
                         and self.states['holding_frying_pan'])
        cond_pancake = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PANCAKES.value and self.states['whisked']
                        and self.states['holding_frying_pan'])

        if common and (cond_tea or cond_soup or cond_pasta or cond_omelette or cond_pancake):
            self.states['heated_up'] = 1
            if cond_tea or cond_pasta:
                self.states['has_boiling_water'] = 1
            elif cond_soup:
                self.task_done = True
        else:
            return Reward.failed_action
        return 0

    def _fill(self):
        cond_tea = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_TEA.value and
                    self.robot_pos in self.operability['sink'] and self.states['holding_saucepan'] and
                    not self.states['saucepan_full'] and self.states['tap_open'])
        cond_pasta = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PASTA.value and
                      self.robot_pos in self.operability['sink'] and self.states['holding_saucepan'] and
                      not self.states['saucepan_full'] and self.states['tap_open'])
        cond_soup = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_SOUP.value and self.states['holding_saucepan'] and
                     not self.states['saucepan_full'] and self.states['has_soup_jar'])
        if cond_tea or cond_soup or cond_pasta:
            self.states['saucepan_full'] = 1
            # print("Saucepan full")
        else:
            return Reward.failed_action
        return 0

    def _open_tap(self):
        if (self.robot_pos in self.operability['sink']) and not self.states['tap_open']:
            self.states['tap_open'] = 1
        else:
            return Reward.failed_action
        return 0

    def _close_tap(self):
        if (self.robot_pos in self.operability['sink']) and self.states['tap_open']:
            self.states['tap_open'] = 0
        else:
            return Reward.failed_action
        return 0

    def _mix(self):
        if Tasks.to_dec(self.task_to_do) == Tasks.MAKE_TEA.value:
            if self.states['has_tea'] and self.states['has_boiling_water']:
                self.task_done = True
            else:
                return Reward.failed_action
        elif Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PASTA.value:
            # adding the pasta to water
            if ((self.robot_pos in self.operability['stove']) and self.states['has_boiling_water']
                    and self.states['has_pasta'] and not self.states['item_cooked']):
                self.states['item_cooked'] = 1
            # adding the sauce to pasta
            elif self.states['item_cooked'] and self.states['pasta_drained']:
                self.task_done = True
            else:
                return Reward.failed_action

        return 0

    def _get_frying_pan(self):
        if (self.robot_pos in self.operability['sink']) and not self.states['holding_frying_pan']:
            self.states['holding_frying_pan'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_eggs(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_eggs']
                and self.states['cabinet_open']):
            self.states['has_eggs'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_milk(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_milk']
                and self.states['cabinet_open']):
            self.states['has_milk'] = 1
        else:
            return Reward.failed_action
        return 0

    def _get_pancake_mix(self):
        if (self.robot_pos in self.operability['cabinet'] and not self.states['has_pancake_mix']
                and self.states['cabinet_open']):
            self.states['has_pancake_mix'] = 1
        else:
            return Reward.failed_action
        return 0

    def _whisk(self):
        cond_omelette = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_OMELETTE.value
                         and self.states['has_eggs'] and not self.states['whisked'])
        cond_pancakes = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PANCAKES.value
                         and self.states['has_eggs'] and self.states['has_pancake_mix'] and self.states['has_milk']
                         and not self.states['whisked'])
        if cond_omelette or cond_pancakes:
            self.states['whisked'] = 1
        else:
            return Reward.failed_action
        return 0

    def _flip(self):
        cond_omelette = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_OMELETTE.value
                         and self.states['whisked'] and self.states['heated_up'] and self.states['fire_on'])
        cond_pancakes = (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PANCAKES.value
                         and self.states['whisked'] and self.states['heated_up'] and self.states['fire_on'])
        if cond_omelette or cond_pancakes:
            self.states['flipped'] = 1
            self.states['item_cooked'] = 1
            self.task_done = True
        else:
            return Reward.failed_action
        return 0

    def _fill_vision_grid(self):
        x, y = self.robot_pos
        offset = [-3, -2, -1, 0, 1, 2, 3]
        fov = []
        for dy in offset:
            for dx in offset:
                if (dx == 0) and (dy == 0):
                    continue
                x_grid, y_grid = x + dx, y + dy
                if (x_grid >= 0) and (x_grid < self.map_width) and (y_grid >= 0) and (y_grid < self.map_height):
                    # First check on coll. obj. set should be faster than iterating through the dictionary every time
                    if (x_grid, y_grid) in self.colliding_objects:
                        for key, val in self.house_objects.items():
                            if (x_grid, y_grid) in val:
                                obj_id = self.house_objects_id[key]
                                fov.append(obj_id)
                    else:
                        fov.append(0)
                else:
                    fov.append(self.house_objects_id['wall'])  # Everything outside bounds is considered as a wall

        self.vision_grid = np.array(fov)
        # print_vision_grid(self.vision_grid)  # TODO: DEBUG only

    def set_position(self, pos):
        if tuple(pos) not in self.colliding_objects:
            self.robot_pos = tuple(pos)
        else:
            raise Exception("Indicated position is not reachable.")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        reward = self.action_dict[action]()
        done = False

        # Check if all the requirements are fulfilled for the specified task
        if self.task_done:
            # Make tea
            if (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_TEA.value and
                    not self.states['cabinet_open'] and not self.states['fire_on'] and not self.states['tap_open']):
                print("TEA, you just made tea!!!")
                done = True
                reward = Reward.completed_task
            # Make soup
            elif (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_SOUP.value and
                  not self.states['cabinet_open'] and not self.states['fire_on']):
                # TODO: should i put not self.states['tap_open'] ?? it'd be nice
                print("SOUP, you just made soup!!!")
                done = True
                reward = Reward.completed_task
            # Make pasta
            elif (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PASTA.value and
                  not self.states['cabinet_open'] and not self.states['fire_on'] and not self.states['tap_open']):
                print("PASTA, you just made pasta!!!")
                done = True
                reward = Reward.completed_task
            # Clean stove
            elif (Tasks.to_dec(self.task_to_do) == Tasks.CLEAN_STOVE.value and
                  not self.states['tap_open']):
                print("CLEANED, you just cleaned the stove!!!")
                done = True
                reward = Reward.completed_task
            # Make omelette
            elif (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_OMELETTE.value and
                  not self.states['cabinet_open'] and not self.states['fire_on']):
                print("OMELETTE, you just made an omelette!!!")
                done = True
                reward = Reward.completed_task
            # Make pancakes
            elif (Tasks.to_dec(self.task_to_do) == Tasks.MAKE_PANCAKES.value and
                  not self.states['cabinet_open'] and not self.states['fire_on']):
                print("PANCAKES, you just made pancakes!!!")
                done = True
                reward = Reward.completed_task

        # self._fill_vision_grid()  # next state vision grid

        next_state = np.hstack((self.robot_pos, list(self.states.values()), self.task_to_do))
        return next_state, reward, done, {}

    def reset(self):
        self.task_done = False  # Indicates that the task is done, but there might be more actions needed to get the
        self.task_to_do = Tasks.to_binary_list(0)
        # reward (e.g. fire is still on)
        self.house_objects = {}
        self.house_objects_id = {}
        self._generate_house()

        for key in self.states.keys():
            self.states[key] = 0

        spawn = True
        while spawn:
            # Only spawn in the kitchen for the moment
            x = random.randrange(11, self.map_width)
            y = random.randrange(14, self.map_height)
            if (x, y) not in self.colliding_objects:
                self.robot_pos = (x, y)
                spawn = False

        return np.hstack((self.robot_pos, list(self.states.values()), self.task_to_do))

    # def render(self, mode='human'):
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
    #         self.viewer.set_bounds(0, VIEWPORT_W, 0, VIEWPORT_H)  # TODO:necessary? they do it in bipedal and lunar
    #
    #     self._draw_robot()
    #     self._draw_objects()
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_current_task(self, task):
        if not isinstance(task, Tasks):
            raise TypeError("task should be of the class type Tasks")
        self.task_to_do = Tasks.to_binary_list(task.value)
        return np.hstack((self.robot_pos, list(self.states.values()), self.task_to_do))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _draw_objects(self):
        for obj, pos in self.house_objects.items():
            color = ObjectColors.get_color(obj)
            for square in pos:
                self._draw_square(square, color)

    def _draw_square(self, pos, color):
        x, y = pos
        v = [(x * self.scale, y * self.scale), (x * self.scale, (y + 1) * self.scale),
             ((x + 1) * self.scale, (y + 1) * self.scale), ((x + 1) * self.scale, y * self.scale)]
        self.viewer.draw_polygon(v, color=color)

    # def _draw_robot(self):
    #     # self._draw_square(self.robot_pos, color=(0.5, 0.5, 0.5))
    #     x, y = self.robot_pos
    #     # Draw body
    #     v = [((x + 0.1) * self.scale, y * self.scale),
    #          ((x + 0.1) * self.scale, (y + 0.8) * self.scale),
    #          ((x + 0.9) * self.scale, (y + 0.8) * self.scale),
    #          ((x + 0.9) * self.scale, y * self.scale)]
    #     self.viewer.draw_polygon(v, color=(0.701, 0.701, 0.701))
    #     # Draw ears
    #     for dx1, dx2 in ((0, 0.1), (0.9, 1)):
    #         v = [((x + dx1) * self.scale, (y + 0.275) * self.scale),
    #              ((x + dx1) * self.scale, (y + 0.525) * self.scale),
    #              ((x + dx2) * self.scale, (y + 0.525) * self.scale),
    #              ((x + dx2) * self.scale, (y + 0.275) * self.scale)]
    #         self.viewer.draw_polygon(v, color=(0.6, 0.6, 0.6))
    #     # Draw hat
    #     v = [((x + 0.175) * self.scale, (y + 0.8) * self.scale),
    #          ((x + 0.3) * self.scale, (y + .95) * self.scale),
    #          ((x + 0.7) * self.scale, (y + .95) * self.scale),
    #          ((x + 0.825) * self.scale, (y + 0.8) * self.scale)]
    #     self.viewer.draw_polygon(v, color=(0.6, 0.6, 0.6))
    #     # Draw mouth
    #     v = [((x + 0.2) * self.scale, (y + 0.15) * self.scale),
    #          ((x + 0.2) * self.scale, (y + 0.35) * self.scale),
    #          ((x + 0.8) * self.scale, (y + 0.35) * self.scale),
    #          ((x + 0.8) * self.scale, (y + 0.15) * self.scale)]
    #     self.viewer.draw_polygon(v, color=(0.9, 0.9, 0.9))
    #     # teeth
    #     number_teeth = 6
    #     dist = 0.6 / number_teeth
    #     for i in range(1, number_teeth):
    #         p1 = ((x + 0.2 + i * dist) * self.scale, (y + 0.15) * self.scale)
    #         p2 = ((x + 0.2 + i * dist) * self.scale, (y + 0.35) * self.scale)
    #         self.viewer.draw_polyline((p1, p2), color=(0, 0, 0), linewidth=1.75)
    #     # v = [((x + 0.2 + 0.1) * self.scale, (y + 0.15) * self.scale),
    #     #      ((x + 0.2 + 0.1) * self.scale, (y + 0.35) * self.scale)]
    #     # self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
    #
    #     # Draw eyes
    #     for dx in (0.30, 0.7):
    #         t = rendering.Transform(translation=((x + dx) * self.scale, (y + 0.5625) * self.scale))
    #         self.viewer.draw_circle(radius=(0.12 * self.scale), res=10, color=(0.643, 0.039, 0.039)).add_attr(t)

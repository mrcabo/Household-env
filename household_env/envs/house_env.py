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
from gym.envs.classic_control import rendering

# A good example is https://github.com/openai/gym-soccer

""" Maybe a more detailed description of the environment here"""

FPS = 60

VIEWPORT_W = 600
VIEWPORT_H = 600

Rewards = namedtuple('Rewards', ['bump_into_wall',
                                 'walking',
                                 'take_action',
                                 'turn_on_tv'])
Reward = Rewards(-3., -.5, -.5, 300.)


def print_vision_grid(grid):
    # DEBUG only - prints the vision grid in a squared format
    bot = np.flip(np.reshape(grid[0:21], (3, 7)), 0)
    mid = np.hstack((grid[21:24], np.array(-1), grid[24:27]))
    top = np.flip(np.reshape(grid[27:], (3, 7)), 0)
    print(np.vstack((top, mid, bot)))


class Tasks(Enum):
    TURN_ON_TV = 2
    TURN_ON_DISHWASHER = 6

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

        self.reset()

        # Min-Max values for coordinates, order encoding, object id
        # low = np.hstack((np.zeros(2), np.zeros(5), np.zeros(48)))
        # high = np.hstack((np.array([19, 19]), np.ones(5), np.array([5] * 48)))
        low = np.hstack((np.zeros(2), np.zeros(5), np.zeros(4)))
        high = np.hstack((np.array([19, 19]), np.ones(5), np.array([8] * 4)))
        self.action_space = spaces.Discrete(4)
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
        rew = Reward.walking
        if self.robot_pos in self.operability['tv']:
            rew = Reward.turn_on_tv
            print(f"Congrats!! You reached dest. reward: {rew}")
            self.task_done = True
        return rew

    def _add_to_buffer(self, action):
        self.action_buffer.append(action)
        return self._calculate_reward()

    def _calculate_reward(self):
        # task = self.state
        # Reward for turning on TV
        if (Tasks.to_dec(self.task_to_do) == Tasks.TURN_ON_TV.value) and (
                self.robot_pos in self.operability['tv']) and (
                self.action_buffer == [8]):
            self.task_done = True
            print("TV turned ON!!!")
            return Reward.turn_on_tv
        return Reward.take_action

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        reward = self.action_dict[action]()

        self._fill_vision_grid()  # next state vision grid

        done = self.task_done
        n_actions = 4  # number of past actions remembered in the state
        buf = np.pad(np.asarray(self.action_buffer, dtype=int),
                     (0, n_actions - len(self.action_buffer)))
        if len(self.action_buffer) >= n_actions:
            done = True
        next_state = np.hstack((self.robot_pos, self.task_to_do, buf))
        return next_state, reward, done, {}

    def reset(self):
        self.action_buffer = []
        self.task_done = False
        self.house_objects = {}
        self.house_objects_id = {}
        self._generate_house()
        self.action_dict = {0: self._move_up,
                            1: self._move_down,
                            2: self._move_left,
                            3: self._move_right,
                            4: partial(self._add_to_buffer, 4),
                            5: partial(self._add_to_buffer, 5),
                            6: partial(self._add_to_buffer, 6),
                            7: partial(self._add_to_buffer, 7),
                            8: partial(self._add_to_buffer, 8)}
        spawn = True
        while spawn:
            x = random.randrange(0, self.map_width)
            y = random.randrange(0, self.map_height)
            if (x, y) not in self.colliding_objects:
                self.robot_pos = (x, y)
                spawn = False

        return np.hstack((self.robot_pos, self.task_to_do, np.zeros(4, dtype=int)))

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W, 0, VIEWPORT_H)  # TODO:necessary? they do it in bipedal and lunar

        self._draw_robot()
        self._draw_objects()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_current_task(self, task):
        if not isinstance(task, Tasks):
            raise TypeError("task should be of the class type Tasks")
        self.task_to_do = Tasks.to_binary_list(task.value)

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

    def _draw_robot(self):
        # self._draw_square(self.robot_pos, color=(0.5, 0.5, 0.5))
        x, y = self.robot_pos
        # Draw body
        v = [((x + 0.1) * self.scale, y * self.scale),
             ((x + 0.1) * self.scale, (y + 0.8) * self.scale),
             ((x + 0.9) * self.scale, (y + 0.8) * self.scale),
             ((x + 0.9) * self.scale, y * self.scale)]
        self.viewer.draw_polygon(v, color=(0.701, 0.701, 0.701))
        # Draw ears
        for dx1, dx2 in ((0, 0.1), (0.9, 1)):
            v = [((x + dx1) * self.scale, (y + 0.275) * self.scale),
                 ((x + dx1) * self.scale, (y + 0.525) * self.scale),
                 ((x + dx2) * self.scale, (y + 0.525) * self.scale),
                 ((x + dx2) * self.scale, (y + 0.275) * self.scale)]
            self.viewer.draw_polygon(v, color=(0.6, 0.6, 0.6))
        # Draw hat
        v = [((x + 0.175) * self.scale, (y + 0.8) * self.scale),
             ((x + 0.3) * self.scale, (y + .95) * self.scale),
             ((x + 0.7) * self.scale, (y + .95) * self.scale),
             ((x + 0.825) * self.scale, (y + 0.8) * self.scale)]
        self.viewer.draw_polygon(v, color=(0.6, 0.6, 0.6))
        # Draw mouth
        v = [((x + 0.2) * self.scale, (y + 0.15) * self.scale),
             ((x + 0.2) * self.scale, (y + 0.35) * self.scale),
             ((x + 0.8) * self.scale, (y + 0.35) * self.scale),
             ((x + 0.8) * self.scale, (y + 0.15) * self.scale)]
        self.viewer.draw_polygon(v, color=(0.9, 0.9, 0.9))
        # teeth
        number_teeth = 6
        dist = 0.6 / number_teeth
        for i in range(1, number_teeth):
            p1 = ((x + 0.2 + i * dist) * self.scale, (y + 0.15) * self.scale)
            p2 = ((x + 0.2 + i * dist) * self.scale, (y + 0.35) * self.scale)
            self.viewer.draw_polyline((p1, p2), color=(0, 0, 0), linewidth=1.75)
        # v = [((x + 0.2 + 0.1) * self.scale, (y + 0.15) * self.scale),
        #      ((x + 0.2 + 0.1) * self.scale, (y + 0.35) * self.scale)]
        # self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # Draw eyes
        for dx in (0.30, 0.7):
            t = rendering.Transform(translation=((x + dx) * self.scale, (y + 0.5625) * self.scale))
            self.viewer.draw_circle(radius=(0.12 * self.scale), res=10, color=(0.643, 0.039, 0.039)).add_attr(t)

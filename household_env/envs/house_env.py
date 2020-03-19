import json
import random

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


class ObjectColors:
    with open('colors_house_objects.json') as json_file:
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
        self.robot_pos = (None, None)

        self.reset()

        low = np.hstack((np.zeros(2), np.zeros(49)))
        high = np.hstack((np.array([19, 19]), np.array([10] * 49)))
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        print("j")

    def __del__(self):
        pass

    def _generate_house(self):
        with open('house_objects.json') as json_file:
            self.house_objects = json.load(json_file)
        # All the objects that the robot might collide with, so its easier to see if it can move without colliding
        self.colliding_objects = set()
        for values in self.house_objects.values():
            values = [tuple(x) for x in values]
            self.colliding_objects = self.colliding_objects.union(values)
        print(f"Occupied places are {self.colliding_objects}")  # TODO:debug only

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
            print("Tried to bump into object")  # TODO debug
            return -1  # TODO: neg reward for bumping into object
        if restriction:
            print("Tried to bump into wall")  # TODO debug
            return -1  # TODO: neg reward for bumping into wall
        self.robot_pos = new_pos
        return 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        aux = self.action_dict[action]()

        # return np.array(state), reward, done, {}
        return None, aux, False, {}

    def reset(self):
        self._generate_house()
        self.action_dict = {0: self._move_up,
                            1: self._move_down,
                            2: self._move_left,
                            3: self._move_right,
                            4: self._move_up,  # TODO..
                            5: self._move_down,
                            6: self._move_left,
                            7: self._move_right}
        spawn = True
        while spawn:
            x = random.randrange(0, self.map_width)
            y = random.randrange(0, self.map_height)
            if (x, y) not in self.colliding_objects:
                self.robot_pos = (x, y)
                spawn = False

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W, 0, VIEWPORT_H)  # TODO:necessary? they do it in bipedal and lunar

        self._draw_robot()
        self._draw_objects()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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

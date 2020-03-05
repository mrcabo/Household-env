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
    colors = {'TV': (0, 0.9, 0.4),
              'couch': (0.4, 0.2, 0)
              }

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
        self.robot_pos = (0, 0)

        self.reset()

    def __del__(self):
        pass

    def _generate_house(self):
        self.house_objects = {'TV': {(0, 19), (0, 18), (0, 17)},
                              'couch': {(4, 19), (4, 18), (4, 17)}
                              }
        # All the objects that the robot might collide with, so its easier to see if it can move without colliding
        self.colliding_objects = set()
        for values in self.house_objects.values():
            self.colliding_objects = self.colliding_objects.union(values)
        print(f"Occupied places are {self.colliding_objects}")  # TODO:debug only

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass

    def reset(self):
        self._generate_house()

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
        # TODO: Maybe draw a nicer robot? :D
        self._draw_square(self.robot_pos, color=(0.5, 0.5, 0.5))

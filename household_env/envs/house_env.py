import gym
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle
from gym.envs.classic_control import rendering

# A good example is https://github.com/openai/gym-soccer

""" Maybe a more detailed description of the environment here"""

FPS = 60


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

        self.reset()

    def __del__(self):
        pass

    def _generate_house(self):
        self.house_objects = {'TV': {(0, 0), (0, 1), (0, 2)},
                              'couch': {(4, 0), (4, 1), (4, 2)}}

        # All the objects that the robot might collide with, so its easier to see if it can move
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
        screen_width = 600
        screen_height = 600

        scale = screen_width / self.map_width

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# A good example is https://github.com/openai/gym-soccer


class HouseholdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

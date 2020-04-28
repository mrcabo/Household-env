from enum import Enum, auto
from household_env.envs.house_env import Tasks
import gym
import time

tasks_list = [Tasks.TURN_ON_TV, Tasks.TURN_ON_DISHWASHER]

env = gym.make('household_env:Household-v0')
env.reset()
env.set_current_task(tasks_list[0])
env.render()

done = False
while not done:
    x = int(input("Enter your command: "))
    if x < 0 or x > 8:
        print("Input should be between 0 and 8")
        break
    next_state, reward, done, info = env.step(x)  # take a random action
    env.render()
    print(f"next_state: {next_state}")
    print(f"Reward: {reward}")

env.close()

from enum import Enum, auto
from household_env.envs.house_env import Tasks
import gym
import time

tasks_list = [Tasks.TURN_ON_TV, Tasks.MAKE_BED]

env = gym.make('household_env:Household-v0')
env.set_current_task(tasks_list[0])
x = 0
while x != -1:
    state = env.reset()
    env.render()
    done = False
    while not done:
        print(f"Current state: {state}")
        x = int(input("Enter your command: "))
        if x < 0 or x > 8:
            print("Input should be between 0 and 8")
            break
        next_state, reward, done, info = env.step(x)  # take a random action
        state = next_state
        env.render()
        print(f"next_state: {next_state}, Reward: {reward}, Done: {done}")
    print("End of episode\n")

env.close()

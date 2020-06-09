from enum import Enum, auto
from household_env.envs.house_env import Tasks
import gym
import time

tasks_list = [Tasks.MAKE_TEA, Tasks.MAKE_SOUP]

env = gym.make('household_env:Household-v0')
env.set_current_task(Tasks.MAKE_TEA)
x = 0
while x != -1:
    state = env.reset()
    env.render()
    done = False
    while not done:
        print(f"\nCurrent state: {state}")
        print(env.env.states)
        x = int(input("Enter your command: "))
        if x < 0 or x > 15:
            print("Input should be between 0 and 15")
            break
        next_state, reward, done, info = env.step(x)  # take a random action
        state = next_state
        env.render()
        # print(f"Next_state: {next_state}")
        print(f"Reward: {reward}, Done: {done}")
        print(f"Done: {done}")
    print("End of episode\n\n")

env.close()

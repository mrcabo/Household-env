# Household-env

This is a gym environment that represents a robot agent in a household environment for RL purposes.

#### Table of Contents  
[How to run it](#how-to-run-it)  
[Observation space](#observation-space)  
[Action space](#action-space)  
[Tasks](#tasks)


## How to run it

To run an example of the environment first install it and then run the dummy file.
```bash
#!/bin/bash
pip install -e Household-env 
python3 dummy.py
```

## Observation space

The robot has a vision grid of 7x7. The vision grid inputs will return values that represent the content of that cell. 

The tasks encoding is binary instead of label encoding (0: task1, 1: task2...) because there is no ordering for the
 tasks, but the alg. might think there is.
 
 **Note**: We could do the same for the vision grid, but then the observation space would increase a lot (48*3=147 for
  for only 7 representable objects )

Num   | Observation                |  Min   |  Max
------|----------------------------|--------|-------
0     | x_coord_robot              |  0     |  19
1     | y_coord_robot              |  0     |  19
2     | task_encoding              |  0     |  1
...   | ...                        |        |  
6     | task_encoding              |  0     |  1
7     | 1st action                 |  0     |  8
8     | 2nd action                 |  0     |  8
9     | 3rd action                 |  0     |  8
10    | 4th action                 |  0     |  8
11    | 5th action                 |  0     |  8
12    | 6th action                 |  0     |  8
13    | 7th action                 |  0     |  8
(vision not yet)
11    | vision_grid                |  0     |  1
..    | ..                         |  0     |  1
58    | vision_grid                |  0     |  1

Objects will return the following values when within range of the 7x7 vision grid.

Num   | Object
------|---------------
0     | Nothing
1     | wall
2     | TV
3     | couch
4     | bed
5     | fridge
6     | dishwasher
7     | person

## Action space

Only one action can be taken at each time step. The Num of the action to be taken is passed as the argument to the
 `step` function.

Num   | Action                     |  Min   |  Max
------|----------------------------|--------|-------
0     | move_up                    |  0     |  1
1     | move_down                  |  0     |  1
2     | move_left                  |  0     |  1
3     | move_right                 |  0     |  1
4     | (A) extend_arm             |  0     |  1
5     | (B) retract_arm            |  0     |  1
6     | (C) grasp                  |  0     |  1
7     | (D) drop                   |  0     |  1
8     | (E) push                   |  0     |  1

## Tasks

The tasks encoding is binary instead of label encoding (0: task1, 1: task2...) because there is no ordering for the
 tasks, but the alg. might think there is. We currently use 5 units so up to 31 tasks.
 
Num   | Action                     |  Action sequence
------|----------------------------|-----------------
0     | No tasks                   |  -
1     | Turn on TV                 |  E
2     | Clean table                |  ABAB
3     | Clean stove                |  ABAB
4     | Make bed                   |  ACAD
5     | Do laundry                 |  ACBADE
6     | Put dryer                  |  ACBADE
7     | Dishwasher                 |  ACBADEE

# household-env

This is a gym environment that represents a robot agent in a household environment for RL purposes.

## Observation space

The robot has a vision grid of 7x7. The vision grid inputs will return values that represent the content of that cell. 

Num   | Observation                |  Min   |  Max
------|----------------------------|--------|-------
0     | x_coord_robot              |  0     |  1
1     | y_coord_robot              |  0     |  1
2     | order_encoding             |  0     |  1
..    | ..                         |  0     |  1
6     | order_encoding             |  0     |  1
7     | vision_grid                |  0     |  1
..    | ..                         |  0     |  1
55    | vision_grid                |  0     |  1

Objects will return the following values when within range of the 7x7 vision grid.

Num   | Object
------|---------------
0     | Nothing
1     | TV
2     | fridge
3     | couch
4     | person
5     | bed

## Action space

Only one action can be taken at each time step. The Num of the action to be taken is passed as the argument to the
 `step` function.

Num   | Action                     |  Min   |  Max
------|----------------------------|--------|-------
0     | move_up                    |  0     |  1
1     | move_down                  |  0     |  1
2     | move_left                  |  0     |  1
3     | move_right                 |  0     |  1
4     | extend_arm                 |  0     |  1
5     | retract_arm                |  0     |  1
6     | grasp                      |  0     |  1
7     | drop                       |  0     |  1
8     | push                       |  0     |  1

# household-env
This is a gym environment that represents a robot agent in a household environment for RL purposes.

## Action space

Only one action can be taken at each time step. The Num of the action to be taken is passed as the argument to the
 `step` function.

Num   | Observation                |  Min   |  Max 
------|----------------------------|--------|-------
0     | move_up                    |  0     |  1    
1     | move_down                  |  0     |  1    
2     | move_left                  |  0     |  1    
3     | move_right                 |  0     |  1    
4     | extend_arm                 |  0     |  1    
5     | retract_arm                |  0     |  1    
6     | grasp                      |  0     |  1    
7     | push                       |  0     |  1    

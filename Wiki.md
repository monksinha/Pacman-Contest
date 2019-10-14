# # UoM COMP90054 Contest Project

## Youtube Link

TODO

## Overview

The purpose of this project is to implement a Pac Man Autonomous Agent that can play and compete in a tournament.

This wiki contains three parts:

1. Designs and Challenges
2. Approaches and Implementation
   - A* Heuristic Search
   - Monte Carlo Tree Search
3. Conclusion

## 1. Designs and Challenges

Before we started to implement our Pacman agents, we first analyzed the feasibility and difficulty of the implementation of different algorithms according to some limitations of this competition. The limitations include:

1. There is only **1 second** for computation before the agents move.
2. The tournament will run on **random maps**, instead of the provided baseline map.

For limitation 1, that means we cannot use a too complex algorithm that requires much computation and time to decide the next movement. For limitation 2, that means we cannot use any specific methods for a certain map or terrain, e.g. hand-coded movement decision-making when the start position is on the top-right side of the map. 

Fortunately, we also have some advantages:

1. **15 seconds** to do initialization for the agents before the game start. 
2. We can **import additional files** in our directory using ```sys.path.append()```

For these advantages, this means we can scan the map and obtain some key information of the current map for each round before the competition starts, or import our pre-trained model using various frameworks like Tensorflow or PyTorch.
Based on these limitations and advantages, we compared all the algorithms covered in the lectures.

| Algorithm                                | Implementation Difficulty | Computation time | Performance                           | Generalization Ability                |
| ---------------------------------------- | ------------------------- | ---------------- | ------------------------------------- | ------------------------------------- |
| A* Heuristic Search                      | Low                       | Low              | Depend on the heuristic function      | Good                                  |
| Policy Iteration (Model-Based MDP)       | Mid                       | High             | Good (if have enough time to compute) | Bad                                   |
| Monte Carlo Tree Search (Model-Free MDP) | Mid                       | Mid              | Good (if have enough time to compute) | Good (if have enough time to compute) |
| Q-Learning (Reinforcement Learning)      | Hard                      | Low              | Good (if trained long enough)         | Good (if trained long enough)         |
| PDDL                                     | Hard                      | Mid              | Good                                  | Good                                  |

From the table, we can know that:
1. If we have a good heuristic function, the A* Heuristic Search is the best solution for the competition because of its low implementation difficulty, low computation time, and good generalization ability. However, the problem is how to generate a good heuristic function.

2. Q-Learning has good performance and generalization ability, but it requires long-time training while we only have 1 month to finish the whole project. Besides, the implementation difficulty of q-learning is the hardest among all the algorithms. The same as PDDL.

3. Both Model-Based Markov Decision Process and Model-Free Markov Decision Process have good performance, but it is hard to convert the policy from one map to another while using Policy Iteration and so it does not have any generalization ability.

After considering these factors comprehensively,  we decided to use both A* Heuristic Search and Monte Carlo Tree Search (Model-Free MDP).

## 2. Approaches and Implementation

### A* Heuristic Search

### Monte Carlo Tree Search



## 3. Conclusion

##  Team Members

- Zhengyu Chen - [zhengyuc@student.unimelb.edu.au](mailto:zhengyuc@student.unimelb.edu.au) - 991678
- Junrong Su - junrongs@student.unimelb.edu.au - 963294
- Yang Lu - yang.lu1@student.unimelb.edu.au - 985419


# UoM COMP90054 Contest Project
![Semantic description of image](http://ai.berkeley.edu/projects/release/contest/v1/002/capture_the_flag.png)  
*The Project is based on 'Contest: Pacman Capture the Flag', from http://ai.berkeley.edu/contest.html*
# Table of Contents
- 1 &nbsp; Introduction
- 2 &nbsp; Game Analysis
    - 2.1 &nbsp; Basics
    - 2.2 &nbsp; Estimate Enemies Position
    - 3.3 &nbsp; Determine Blind Alley (Dead End)
- 3 &nbsp; Design and Challenges
    - 3.1 &nbsp; Choices of Techniques
    - 3.2 &nbsp; A* Heuristic Search
    - 3.3 &nbsp; Monte Carlo Tree Search
- 4 &nbsp; Experiments and Approach Evaluation
- 5 &nbsp; Improvements and Reflections



**Team Name:**  Greatest-Jagras

**Team Members:**
* Zhengyu Chen - zhengyuc@student.unimelb.edu.au - 991678
* Junrong Su - junrongs@student.unimelb.edu.au - 963294
* Yang Lu - yang.lu1@student.unimelb.edu.au - 985419

**Important Note**: In the project repository, **myTeam.py** is for A* algorithm, **myTeam2.py** is for MCTS algorithm.

# 0. Youtube presentation
Video Link: https://youtu.be/IrAce-rT7-Y

Or Click <a href= "https://youtu.be/IrAce-rT7-Y"> Here </a> For the video

# 1. Introduction
The goal of this project is to implement two autonomous agents which collaborate and try to eat the food in the opponent's territory while defending the food on our own side. See [UC Berkely CS188](http://ai.berkeley.edu/contest.html) for detailed specification of this project.


# 2. Game Analysis
As we started this project, we discussed some important aspects that could affect our designs and implementations.
## 2.1 Basics
1. The state-space can be very large, as some map layout consists of hundreds of grid cells.
2. Actions are deterministic, i.e., when the agent follows the algorithm to move towards a direction, it will not deviate from that direction.
3. The environment (game state) can be observed by agent, but with some noises (probabilistic).
4. Computation time for each action is limited (1 second).
5. It is an adversarial game, which means that when choosing actions for our agents, we need to consider the state of the opponent's agents.


## 2.2 Estimate Enemies Position 

#### Assumption

All agent move one Manhattan distance each turn. (Except rebirth)

#### Basic idea

The basic idea is based on noisy distance and initial positions.  Since every turn the agent could move one Manhattan distance, the possible current positions are bound to be in the previous possible positions expanding one grid, which could be imagined as a circle growing bigger and bigger. On the other side, the noisy distance falls into the range [real distance -6, real distance +6], vice versa. And every time we observe, we could get a possible position falling into a ring. Combining with these two observations, by picking the intersection of them, we can get a rough estimation for our enemies. 

A more advanced and accurate method we used is that we combine the above two with previous observation and the other agent's current and previous observations. Thus, we use a total of four observations with previous possible position to get our estimation. The result shows this method works well. Besides, we assume enemies uniformly appear in these possible areas.

#### Difficulties & Caveats 

- Reborn problem: When enemies are reborn, the previous possible position should be the initial position. Besides, the previous observation should be abandoned. The reason behind is that rebirth breaks our assumption of one Manhattan distance each time. Therefore, we should re-estimate.
- Order of agent action: Assumed that the order is my agent 1, enemy agent 1, my agent 2, enemy agent 2 with index 0, 1, 2, 3 respectively. For my agent 1, its current observation for enemy agent 2 is "accurate" since enemy 2 moves before it, while for enemy 1, its observation is indeed an obsolete one since enemy 1 moves after it. In conclusion, for an agent, it has an "accurate" observation for the enemy move before it while its ally has an "accurate" observation for the enemy move after it.

#### A better way

What can we do more based on this idea? Now we only use current and previous one observations, and the problem is sometimes these observations are so consistent that we get wider and wider range even we do intersection operation. To get more precise, we should utilise all observations since we lost sight of our enemies.

To better estimate enemies' positions, we could have another prior distribution(currently, we use uniform distribution), such as beta distribution which is conjugate prior to help us analyse whether the enemy is chasing us.

## 2.3 Determine Blind Alley

First thing first, in theory, this thing won't bother us if we use a Monte Carlo Tree Search as long as we simulate steps more than twice of the depth of alley and we would get back to the entrance at the right time. However, that takes huge computing resources and is impractical in our case due to limited computing time and hardware. Now, let's cut to the chase.

#### Why this matters?

Assume that we are at the entrance of a blind alley and the distance with our enemy is six. According to the rules, both of us don't have sight to know each other. The minimum steps for the enemy to this entrance is six and if we choose to eat food whose depth at least three in this alley, then bad thing happens. We are stuck in this alley since the enemy is already waiting at the entrance. In another case, if we choose to eat food with depth two, then we need four steps from this entrance back to this entrance. Even if the distance between me and my enemy is five, I still have one spare step to run away.

#### How to find blind alley ahead?

 A simple implementation is, for all legal positions, we get its successors and the end of the alley should have only one successor pointing to the entrance. Then we continuously get the successors of the previous successors which points to the entrance. These positions form the main part of the alley. The last is about when we stop the loop. Indeed, the entrance of this problem is a vertex of a cycle(graph theory). Therefore, we should find all cycles in the enemy's territory.

#### What's next?

After we find all blind alleys, we can count the maze distance between inner food and the entrance. Every time, when we decide to eat a certain food inside an alley, we should promise that the cost/steps of starting from the current position to that food and then come back to the entrance should less than the maze distance between our enemy with that entrance. The discrepancy is the distance between us when we come back. 

#### More thoughts about cycle

Once we are in a cycle, we can run a circle around our enemies without being eaten unless we are double-teamed. Under a double-team situation, we still have a chance to survive if we make it go to another cycle without considering the capsules. 

#### Difficulties

- Not familiar with graph theory and graph algorithm, thus it's hard for us to fully implement the above idea.

- The blind alleys we calculate aren't fully correct, we didn't deal with the fork situation which means our entrance could be part of a blind alley.


# 3 Design and Challenges
## 3.1 Choices of Techniques
For the candidate techniques, we analysed that:
1. Heuristic Search suits large state-space model and requires low computation time. In addition, we have already implemented some search algorithms in Project 1, it seems reasonable to start our design using a heuristic search.
2. Implementing Classical Planning (PDDL) seems to be hard, and we would have to spend extra time familiarizing ourself with writing PDDL predicate if we choose this technique.
3. For policy iteration and value iteration, it seems that the state space could be too large to determine the value functions V(s).
4. For Reinforcement Learning such as TD, we need to maintain Q-function using Q-table for each action in each state, therefore, like value iterations, this technique may not scale well to large game states.
5. Game-Theoretic Methods, not covered yet when we started the project.
6. Monte Carlo Tree Search, combined with UCB, is a widely adopted and successful approach nowadays. Unlike other methods, it calculates action online at each state. We are tempted to experiment with this algorithm for our project.

Given the above considerations, we decided to implement **A\* Heuristic Search** and **Monte Carlo Tree Search** for the project.
  
## 3.2 A* Heuristic Search
#### 'Friendly' Agent

This agent is implemented based on section 2, it is an improved version based on our previous attacker 'Positive'. Besides, it comprises several components as following

- Picking an entrance to the enemy's territory based on current observation. We mask the entrances in about 3 Manhattan distance of the enemies. Meanwhile, we assign a counter to each entrance. Every step we increase one for those counters and we penalize those masked entrances by five. Before we pick the entrance, we calculate the maze distance from each entrance to its nearest food combining with the counter value to choose the best entrance.

  However, there are still some problems with this approach

  - It's hard to choose suitable "mask distance" for the enemies due to several factors, such as map size, the shape of the boundary, enemies' strategy or action mode to some extent.
  - The math part isn't suitable as well. The adjacent entrances have a similar distance to the nearest food. They are more likely to be masked at the same time as well. Therefore, the adjacent entrances have almost the same value and under some circumstances, the best entrance will alternatively be in this adjacent entrances, which makes agent stuck in these positions turning around and around.
  - Once the enemies keep a distance from the boundary, this approach won't work and we could be stuck in the loop of go ahead and come back.

- The second approach for picking an entrance is by using a heuristic, which is more "artificial intelligence". The heuristic is penalizing the position beside the enemies, thus we find a detour to a certain food.

-  Calculating the minimum maze distance of enemy to my exit path comparing with my distance to that intersection to estimate the probability of surviving. This doesn't work according to experiments due to two factors

  - Enemies don't estimate our positions so they won't find the best solution to eat us which makes this calculation meaningless.
  - In a small size map, a rough estimation doesn't help a lot because the range of noisy is fixed and it's hard to be useful for us. Based on a terrible estimation to calculate the probability of surviving isn't a good idea, which makes the agent wastes some steps to "escape" and return to eat.

- Masking some exits which are near enemy and enemy can reach that exit ahead of us. This thought is based on the situation that the enemy is in our territory and can come back to chase us.

- Masking the food near the enemy by calculating the distance between enemies and food.

  Before the agent invades enemies' territory, it will find a way, which could be a detour to the nearest food as picking an entrance to some extent. Then we use a heuristic to avoid being into the range of one Manhattan distance of enemy and get to a certain food. Every time when we will eat food in a blind alley, we estimate the distance between us and our enemies, then decide to eat or not.

  We go home when we are being chased, or the time limits or the distance to the next food is large than the distance to go home when we are carrying the food.
  
Five Heuristic used (see **myTeam.py** for detailed implementations)
``` python
1.    def manhattanHeuristic(self, pos, goal):
        "The Manhattan distance heuristic for a PositionSearchProblem"
        
2.    def noCrossingHeuristic(self, pos, goal):
        "The heuristic for not crossing the boundary"
        
3.    def DetectOpponentGhostsHeuristic(self, pos, goal):
        "The heuristic for detecting opponent ghost in enemy territory"
        
4.    def DetectOpponentPacmansHeuristic(self, pos, goal):
        "The heuristic for detecting opponent pacman in our home territory"
        
5.   def changeGateHeuristic(self, pos, goal):
        "the heurisitic for changing entry point to the enemy territory"
```




## 3.3 Monte Carlo Tree Search
#### - Design
The idea is that at each state, we build a search tree by simulating the game for a number of iterations. And then we select the next game state by comparing the UCB value of all successor states.

As there are many variations of Monte Carlo Tree Search, we follow the algorithms introduced <a href="https://www.youtube.com/watch?v=UXW2yZndl7U">here</a>

During a simulation, we compute the reward for agents by summing up features*weights value.

Recall the four steps in Monte Carlo Tree Search (see **myTeam2.py** for detailed implementations)
```python
1.    def selection(self, currNode):
        '''
        select node based on the ucb value, keeps searching until reach the leaf node
        '''    
2.    def expansion(self, currNode):
        '''
        expand the node, add children, and randomly select a child
        '''    
3.    def simulation(self, currNode, discount = 0.9):
        '''
        simulate game by randomly choosing next action until we reach the step limit
        '''    
4.    def backPropagation(self, currNode, reward):
        '''
        update reward for the node and its parents
        '''
```
This part of logic is contained in the base class McstAgent, which is subclassed by both Offensive Agent and Defensive Agent. 

In our implementation, the main difference between attacker and defender is the features and weights defined for a variety of scenarios.

#### - Attacker
features:
- onAttack: our Pacman agent is invading enemy's territory
- foodsLeft: number of foods left in enemy territory
- distanceToFood: distance to the closet food
- distanceToCapsule: distance to the closet capsule
- distanceToGhost: distance to the closet ghost
- distanceToBoundary: distance to the boundary of the home side and the enemy side
- scaredTime: the scared time of the enemy ghost
- stop: next action is stay put
- reverse: next action is going backwards
- distanceToStart: distance to the starting position
- isDead: our Pacman agent is dead

We have some general weight settings, for example, we give negative weights to **stop** and **reverse** as we discourage both actions during the search. We give  **dead** a very high penalty so that the agent would not bump into the opponent ghost. We give **distanceToGhost** a high positive weight so that the agent is discouraged from getting close to the ghost, and give **distanceToFood** a negative weight so that agent will be likely to get close to the food.

In addition, we change the weights for some features in different scenarios. For example, when the enemy ghost is scared, we set the weight for **foodsLeft** and **onAttack** to encourage our agent to eat food. Otherwise if the enemy is not scared, we set the weight for **distanceToCapsule**. When the agent has carried some number of foods, we set the weight for **distanceToBoundary** and **distanceToStart** to encourage the agent to take food home.

#### - Defender
features:
- onDefense: our ghost is in defence state
- numInvaders: number of invaders observed in our territory
- invaderDistance: distance to the closet invader
- stop: next action is stay put
- reverse: next action is going backwards
- foodDefending: number of foods left in our territory
- distanceToBoundary: distance to the boundary of the home side and the enemy side

Likewise, the defender has some general weights setting as well as some case-specific weights setting. For example, we give negative weights to **invaderDistance** so that our agent will be discouraged from getting far from the invader. When the agent does not observe invaders, we encourage moves towards the boundary, otherwise, we penalise moves that will lead to a decrease of our defending food.

#### - Challenge
1. Limited computation time. 
At the beginning, we set an iteration limit of 100, i.e, build the tree by simulating 100 times. However, it seems that the computation cannot be completed within 1 second on the server, so we have to instead impose a time limit. By doing so, the search tree may not be complete, hence the value calculated for each node may be somewhat biased. 

2. Definition for reward
This is a critical part of the simulation, and we find it difficult to devise a good and meaningful way to represent a reward for each step. In our experiment, we use the sum of features*weights to denote reward. We also tried to use the change of the value and threshold of value, but the result is not very good.

3. Simulation of the Opponent's Behaviour
In this type of adversarial game, normally we should also simulate opponent's behaviour, we tried to implement in this way, but it seems to be quite complex, so in the end, we only consider simulating our own agent's behaviour.


# 4. Experiment and Approach Evaluation
For each algorithm, we collect 5 game results against staff teams and analyse their performance.
## 4.1 A* Heuristic Search
| A*                | Result 1     | Result 2        | Result 3     | Result 4   | Result 5   |
|-----------------  |:-------------|:--------------- |:-------------|:-----------|:-----------|
| Game Date Time    |15 Oct 10:00  |15 Oct 12:00     |15 Oct 2:00   |15 Oct 4:00 |15 Oct 6:00 |   
| staff_team_basic  | Win          | Win             | Win          | Win        | Win        |
| staff_team_medium | Win          | Win             | Lose         | Win        | Lose       |
| staff_team_top    | Win          | Lose            | Win          | Win        | Win        |
| staff_team_super  | Lose         | Win             | Lose         | Win        | Tie        |

The result suggests that our A* agents have a moderate performance. Sometimes, we can even beat staff_team_super. We also noticed that the performance is not stable, which means that our implementation does not generalize well so that it has advantages in some game settings while perform bad in others.

From some game replays, we find out that in those games we lose, the main reason is that our defender does not defend the right gate from invading or our attacker cannot choose a better entry point for attacking. This indicates that our strategy for choosing the entrance is still problematic to some extent.  

## 4.2 Monte Carlo Tree Search
| MCTS              | Result 1     | Result 2        | Result 3     | Result 4   | Result 5  |
|-----------------  |:-------------|:--------------- |:-------------|:-----------|:----------|
| Game Date Time    |9 Oct 16:00   |9 Oct 18:00      |9 Oct 20:00   |9 Oct 22:00 |10 Oct 8:00 |   
| staff_team_basic  | Tie          | Win             | Lose         | Tie        |  Tie       |
| staff_team_medium | Lose         | Lose            | Lose         | Lose       |  Tie       |
| staff_team_top    | Lose         | Lose            | Lose         | Lose       |  Tie       |
| staff_team_super  | Lose         | Lose            | Lose         | Lose       |  Tie       |

The result indicates that our MCTS agent is merely as good as basic staff team, and the performance is not what we expected.

Clearly, the design is somewhat flawed, we reckon the most controversial part is how to define the reward for the agent and how to do the simulations. We found out that our agents sometimes move back and forth, and it seems that they don't have a specific goal during the game. This may be caused by the wrong assignment of value and weights, so that the real reward for each state is not correctly shown to the agent. It is also possible that the algorithm becomes less accurate when the search tree is not built completely. Moreover, by omitting the opponent's future actions, the simulation may be less useful in terms of reflecting the possible future game states.

We tried to modify this technique, but the result is still not very good. Given the time constraint, we decided to put more effort into the A* Heuristic Algorithm.


# 5. Improvements and Reflections
By the end of this project, although we did not achieve a satisfying result in the competition, we do have gained a better understanding of different AI planning techniques. We discussed some aspects which can be improved in the future, including general improvements as well as method-specific improvements.

## General Improvements
1. More flexible strategies
Currently, one of our agents is responsible for eating food on the far side, the other is defending our own food. However, this strategy is too restrictive and we may lose a good chance to invade the opponent's territory. After looking at some gameplays, we observe some teams use two-attacker strategy, and other teams let the defender attack if there is no threat in the homeland. By adopting these kinds of strategies, the agent can be more flexible and is likely to outperform one-attacker-one-defender strategy.

2. Collaboration between two agents
It turns out that our two agents are acting on their own without considering the state of the other team agent. It might be better if we can let the two agents working as a group rather than as individuals.

## Technique Specific Improvements
#### A* Heuristic Search
1. Improve the selection of entry points for both attacker and defender to avoid deadlock.
2. Our current agent does not consider eating capsule proactively, as we do not know if it is worth giving priority to capsule over food. We need to think further about the logic of eating capsule.
3. Consider other suitable heuristics.
4. Add more condition judgement.

#### Monte Carlo Tree Search
1. Redesign the weights and features or use other ways to evaluate rewards
2. Add in the simulation of the opponent's actions
3. Reuse the built search tree if possible so that we can make better use of limited computation time.


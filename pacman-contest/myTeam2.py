# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys

from game import Directions
from game import Actions
import game

import math
from util import nearestPoint

TIME_LIMIT = 0.9
ITERATION_LIMIT = 100

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


#########################
# Monte Carlo Tree Node #
#########################
class TreeNode:
    def __init__(self, gameState, simulationStep= 6):
        '''
        simulationStep: how far we want to simulate in MCT
        visitTime: how many times the node is visited
        reward: accumulated reward for the node
        '''
        self.gameState = gameState
        self.simluationStep = simulationStep
        self.visitTime = 0
        self.reward = 0.0
        self.parent = None
        self.children = None

    def __str__(self):
        str = 'reward: %f, visited: %d times' %(self.reward, self.visitTime)
        return str

    def __eq__(self, other):
        if other == None: return False
        if not self.gameState == other.gameState: return False
        if not self.simluationStep == other.simluationStep: return False
        if not self.visitTime == other.visitTime: return False
        if not self.reward == other.reward: return False
        if not self.parent == other.parent: return False
        if not self.children == other.children: return False
        return True

    def setParent(self, parent):
        self.parent = parent

    def addChildren(self, child):
        # self.children.append(child)
        child.setParent(self)
        if self.children is not None:
            self.children.append(child)
        else:
            self.children = [child]

    def update(self, reward):
        self.reward += reward
        self.visitTime += 1

    def visited(self):
        return self.visitTime > 0

    def avgReward(self):
        return self.reward/float(self.visitTime)

    def getPrevState(self):
        if self.parent is not None:
            return self.parent.gameState
        else:
            return None


#####################
# Agents Base Class #
#####################
class MctsAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        self.midX = gameState.data.layout.width//2-1 if self.red else gameState.data.layout.width//2

        self.boundary = [(self.midX ,y) for y in range(gameState.data.layout.height) if not gameState.hasWall(self.midX, y)]

        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        '''
        choose the best action according to the Monte Carlo Tree path
        '''
        start = time.time()
        rootNode = self.mctSearch(gameState)
        path = self.generatePath(rootNode)
        nextNode = path[1]
        nextState = nextNode.gameState
        action = nextState.getAgentState(self.index).configuration.direction
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        print(action)
        return action

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getSuccessors(self, gameState):
        '''
        Finds all possible successor states for the current state
        '''
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        successors = [self.getSuccessor(gameState, action) for action in actions]
        return successors

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        pass

    def getWeights(self, gameState, action):
        pass


    ####################
    # Helper Functions #
    ####################
    def getGhostDistance(self, gameState):
        '''
        :return: the distances from current agent to the enemy ghosts
        '''
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostList = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghostList) > 0:
            distances = [self.getMazeDistance(myPos, ghost) for ghost in ghostList]
            return distances
        else:
            return None

    def getInvaderDistance(self, gameState):
        '''
        :return: the distances from current agent to enemy pacmans
        '''
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaderList = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaderList) >0:
            distances = [self.getMazeDistance(myPos, invader) for invader in invaderList]
            return distances
        else:
            return None

    def getFoodDistance(self, gameState):
        '''
        :return: the distances from current agent to all Food in enemy territory
        '''
        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        if len(foodList)>0:
            distances = [self.getMazeDistance(myPos, food) for food in foodList]
            return distances
        else:
            return None

    def getCapsuleDistance(self, gameState):
        '''
        :return: the distances from current agent to all Capsules in enemy territory
        '''
        myPos = gameState.getAgentPosition(self.index)
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            distances = [self.getMazeDistance(myPos, capsule) for capsule in capsuleList]
            return distances
        else:
            return None

    def getNumOfFoods(self, gameState):
        '''
        :return: the current number of food in enemy territory
        '''
        return self.getFood(gameState).count()

    def getNumOfFoodsDefending(self, gameState):
        '''
        :return: the current number of food in our territory
        '''
        return self.getFoodYouAreDefending(gameState).count()

    def getDistanceToBoundary(self, gameState):
        '''
        :return: the closet distance to the boundary
        '''
        myPos = gameState.getAgentPosition(self.index)
        minDistance = min([self.getMazeDistance(myPos, point) for point in self.boundary])
        return minDistance

    def getDistanceToStart(self, gameState):
        '''
        :return: the distance to the starting point of our agent
        '''
        myPos = gameState.getAgentPosition(self.index)
        distance = self.getMazeDistance(myPos,self.start)
        return distance

    def getEnemyScaredTimer(self, gameState):
        '''
        :return: if enemy ghost is scared, return the scared time left, else return 0
        '''
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        timer = []
        for ghost in enemyGhosts:
            timer.append(ghost.scaredTimer)
        if len(timer) > 0:
            return min(timer)
        else:
            return 0

    def isDead(self, prevState, currState):
        '''
        :return: True if the current agent is eaten by enemy ghost
        '''
        currPos = currState.getAgentPosition(self.index)
        prevPos = prevState.getAgentPosition(self.index)
        distance = self.getMazeDistance(currPos, prevPos)
        if currPos is self.start and distance > 10:
            return True
        else:
            return False

    def getCurrentAction(self, gameState):
        '''
        :return: action from previous state to current state
        '''
        return gameState.getAgentState(self.index).configuration.direction

    def getReverseAction(self,gameState):
        '''
        :return: action from current state to previous state
        '''
        return Directions.REVERSE[self.getCurrentAction(gameState)]

    ##############################
    # Monte Carlo Tree functions #
    ##############################
    # https://www.youtube.com/watch?v=UXW2yZndl7U, algorithm implemented based on idea from this video
    def mctSearch(self, gameState):
        '''
        build the search tree in a fixed number of iteration
        '''
        start = time.time()
        rootNode = TreeNode(gameState)
        iterations = ITERATION_LIMIT
        while True:
            # if exceeds either the time limit or iteration limit, stop building the tree
            current = time.time()
            if current - start > TIME_LIMIT or iterations == 0:
                break
            currNode = self.selection(rootNode)
            if currNode.visited() or currNode == rootNode:
                # print("expansion")
                currNode = self.expansion(currNode)
            currReward = self.simulation(currNode)
            self.backPropagation(currNode, currReward)
            iterations -= 1

        return rootNode

    def generatePath(self,currNode):
        '''
        generate the path of the Search Tree, based on the ucb value
        '''
        path = [currNode]
        while currNode.children is not None:
            children = currNode.children
            ucbValues = [self.ucbValue(child) for child in children]
            maxValue = max(ucbValues)
            candidateNodes = [child for child, value in zip(children, ucbValues) if value == maxValue]
            currNode = random.choice(candidateNodes)
            path.append(currNode)
        return path


    def ucbValue(self, currNode, rho = 1.0, Q0 = math.inf):
        '''
        rho: hyperparameter to balance between exploration and exploitation
        Q0:  default value if the node is not visited
        '''
        if currNode.visited():
            confidenceInterval = math.sqrt(rho * math.log(currNode.parent.visitTime) / currNode.visitTime)
            ucbValue = currNode.avgReward() + confidenceInterval
        else:
            ucbValue = Q0
        return ucbValue

    def selection(self, currNode):
        '''
        select node based on the ucb value, keeps searching down the tree until we reach the leaf node
        '''
        # while len(currNode.children) > 0:
        while currNode.children is not None:
            children = currNode.children
            ucbValues = [self.ucbValue(child) for child in children]
            maxValue = max(ucbValues)
            candidateNodes = [child for child, value in zip(children, ucbValues) if value == maxValue]
            currNode = random.choice(candidateNodes)
            # print(currNode)
        return currNode


    def expansion(self, currNode):
        '''
        expand the node, add children, and randomly select a child
        '''
        successors = self.getSuccessors(currNode.gameState)
        for successor in successors:
            child = TreeNode(successor)
            currNode.addChildren(child)
        currNode = random.choice(currNode.children)
        return currNode


    def simulation(self, currNode, discount = 0.9):
        '''
        simulate game by randomly choosing next action until we reach the step limit
        '''
        currState = currNode.gameState
        totalRewards = 0
        step = currNode.simluationStep
        while step > 0:
            actions = currState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            reverse = self.getReverseAction(currState)
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            nextAction = random.choice(actions)
            power = currNode.simluationStep - step
            totalRewards += discount**power * self.evaluate(currState, nextAction)
            successor = self.getSuccessor(currState, nextAction)
            currState = successor
            step -= 1
        return totalRewards


    def backPropagation(self, currNode, reward):
        '''
        update reward for the node and its parents
        '''
        while currNode is not None:
            currNode.update(reward)
            currNode = currNode.parent




###################
# Offensive Agent #
###################
class OffensiveAgent(MctsAgent):
    # def chooseAction(self, gameState):


    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        # myPos = myState.getPosition()
        reverse = self.getReverseAction(gameState)
        foodDistance = self.getFoodDistance(successor)
        capsuleDistance = self.getCapsuleDistance(successor)
        ghostDistance = self.getGhostDistance(successor)

        if myState.isPacman:
            features['onAttack'] = 1

        features['foodsLeft'] = self.getNumOfFoods(successor)
        if foodDistance is not None:
            features['distanceToFood'] = min(foodDistance)

        if capsuleDistance is not None:
            features['distanceToCapsule'] = min(capsuleDistance)

        if ghostDistance is not None:
            features['distanceToGhost'] = min(ghostDistance)

        features['distanceToBoundary'] = self.getDistanceToBoundary(successor)

        features['scaredTime'] = self.getEnemyScaredTimer(successor)

        features['stop'] = 1 if action == Directions.STOP else 0

        features['reverse'] = 1 if action == reverse else 0

        features['distanceToStart'] = self.getDistanceToStart(successor)

        # high penalty for death
        features['isDead'] = 1 if self.isDead(gameState, successor) else 0


        return features


    def getWeights(self, gameState, action):
        """
        Returns a counter of weights for the state
        """
        weights = util.Counter()
        # basic weights
        weights['stop'] = -20
        weights['reverse'] = -20
        weights['isDead'] = -1000
        weights['distanceToGhost'] = 100
        weights['distanceToFood'] = -5


        if self.getEnemyScaredTimer(gameState) > 10:
            weights['foodsLeft'] = -50
            weights['onAttack'] = 100

        elif gameState.getAgentState(self.index).numCarrying <= 3:
            weights['foodsLeft'] = -50
            weights['distanceToCapsule'] = -5
            weights['scaredTime'] = 10
            weights['onAttack'] = 100

        else:
            weights['distanceToBoundary'] = -20
            weights['distanceToStart'] = -20
            weights['onAttack'] = - 100

        return weights


###################
# Defensive Agent #
###################
class DefensiveAgent(MctsAgent):
    # def chooseAction(self, gameState):


    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        reverse = self.getReverseAction(gameState)
        invaderDistance = self.getInvaderDistance(successor)

        if not myState.isPacman:
            features['onDefense'] = 1

        if invaderDistance is not None:
            features['numInvaders'] = len(invaderDistance)
            features['invaderDistance'] = min(invaderDistance)

        if action == Directions.STOP:
            features['stop'] = 1
        if action == reverse:
            features['reverse'] = 1

        features['foodDefending'] = self.getNumOfFoodsDefending(successor)
        features['distanceToBoundary'] = self.getDistanceToBoundary(successor)

        return features

    def getWeights(self, gameState, action):
        """
        Returns a counter of weights for the state
        """
        weights = util.Counter()
        weights['stop'] = -20
        weights['reverse'] = -20
        weights['invaderDistance'] = -10
        weights['onDefense'] = 100

        if self.getGhostDistance(gameState) is None:
            weights['distanceToBoundary'] = -10
            weights['distanceToStart'] = 10

        else:
            weights['foodDefending'] = 50
            weights['numInvaders'] = -1000

        return weights

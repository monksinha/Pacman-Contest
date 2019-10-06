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
import game
# import numpy as np
import math
from util import nearestPoint

NUM_ITERATIONS = 100

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
    def __init__(self, gameState, simulationStep=10):
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



#####################
# Agents Base Class #
#####################
class MctsAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        midLine = [(gameState.data.layout.width//2-1, y) for y in range(0, gameState.data.layout.height)] if self.red \
            else [(gameState.data.layout.width//2, y) for y in range(0, gameState.data.layout.height)]

        self.midPoints = [(x,y) for (x,y) in midLine if not gameState.hasWall(x,y)]

        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        start = time.time()
        currNode = self.mctSearch(gameState, NUM_ITERATIONS)
        avgRewards = []
        for child in currNode.children:
            avgReward = child.avgReward()
            avgRewards.append(avgReward)
        maxReward = max(avgRewards)
        candidateNodes = [child for child, reward in zip(currNode.children,avgRewards) if reward == maxReward]
        nextNode = random.choice(candidateNodes)
        nextState = nextNode.gameState
        action = nextState.getAgentState(self.index).configuration.direction
        print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        print(action)
        return action


    #####################
    # Getter Functions  #
    #####################
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
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getGhostDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghostList = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghostList) > 0:
            distances = [self.getMazeDistance(myPos, ghost) for ghost in ghostList]
            return distances
        else:
            return None

    def getInvaderDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaderList = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaderList) >0:
            distances = [self.getMazeDistance(myPos, invader) for invader in invaderList]
            return distances
        else:
            return None

    def getFoodDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        if len(foodList)>0:
            distances = [self.getMazeDistance(myPos, food) for food in foodList]
            return distances
        else:
            return None

    def getCapsuleDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            distances = [self.getMazeDistance(myPos, capsule) for capsule in capsuleList]
            return distances
        else:
            return None

    def getNumOfFoods(self, gameState):
        return self.getFood(gameState).count()

    def getNumOfFoodsDefending(self, gameState):
        return self.getFoodYouAreDefending(gameState).count()

    def getDistanceToMid(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        minDistance = min([self.getMazeDistance(myPos, point) for point in self.midPoints])
        return minDistance

    def getDistanceToStart(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        distance = self.getMazeDistance(myPos,self.start)
        return distance

    def getEnemyScaredTimer(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        timer = []
        for ghost in enemyGhosts:
            timer.append(ghost.scaredTimer)
        if len(timer) > 0:
            return min(timer)
        else:
            return 0

    #############################
    # Monte Carlo Tree functions#
    #############################
    def mctSearch(self, gameState, iteration):
        rootNode = TreeNode(gameState)

        for t in range(iteration):
            currNode = self.selection(rootNode)
            if currNode.visited() or currNode == rootNode:
                # print("expansion")
                currNode = self.expansion(currNode)
            currReward = self.simulation(currNode)
            self.backPropagation(currNode, currReward)

        return rootNode

    def ucbValue(self, currNode, rho = 1.0, Q0 = math.inf):
        if currNode.visited():
            confidence = math.sqrt(rho * math.log(currNode.parent.visitTime) / currNode.visitTime)
            ucbValue = currNode.reward + confidence
        else:
            ucbValue = Q0
        return ucbValue

    def selection(self, currNode):
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
        successors = self.getSuccessors(currNode.gameState)
        for successor in successors:
            child = TreeNode(successor)
            currNode.addChildren(child)
        currNode = random.choice(currNode.children)
        return currNode


    def simulation(self, currNode, discount = 0.9):
        gameState = currNode.gameState
        totalRewards = 0
        step = currNode.simluationStep
        while step > 0:
            actions = gameState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            nextAction = random.choice(actions)
            power = currNode.simluationStep - step
            totalRewards += discount**power * self.evaluate(gameState, nextAction)
            successor = self.getSuccessor(gameState, nextAction)
            gameState = successor
            step -= 1
        return totalRewards


    def backPropagation(self, currNode, reward):
        while currNode is not None:
            currNode.update(reward)
            currNode = currNode.parent




###################
# Offensive Agent #
###################
class OffensiveAgent(MctsAgent):
    # def chooseAction(self, gameState):



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        # myPos = myState.getPosition()
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
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
        features['distanceToMid'] = self.getDistanceToMid(successor)
        features['scaredTime'] = self.getEnemyScaredTimer(successor)
        features['stop'] = 1 if action == Directions.STOP else 0
        features['reverse'] = 1 if action == reverse else 0
        features['distanceToStart'] = self.getDistanceToStart(successor)

        return features


    def getWeights(self, gameState, action):
        if gameState.getAgentState(self.index).numCarrying <= 3:
            return {'foodsLeft': -100,
                    'distanceToFood': -50,
                    'distanceToCapsule': -50,
                    'distanceToGhost': 1000,
                    'scaredTime': 100,
                    'stop': -100,
                    'reverse': -100,
                    'onAttack': 100
                    }
        else:
            return {'distanceToMid': - 50,
                    'distanceToStart': -50,
                    'distanceToGhost': 1000,
                    'stop': -100,
                    'reverse': -100,
                    'onAttack': -100
                    }


###################
# Defensive Agent #
###################
class DefensiveAgent(MctsAgent):
    # def chooseAction(self, gameState):



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
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
        features['distanceToMid'] = self.getDistanceToMid(successor)

        return features

    def getWeights(self, gameState, action):
        return {'foodDefending': 100,
                'numInvaders': -1000,
                'onDefense': 100,
                'invaderDistance': -50,
                'stop': -100,
                'reverse': -100,
                'distanceToMid': -50
                }

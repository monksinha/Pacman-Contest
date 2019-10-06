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
    def __init__(self, gameState, count, reward, parent=None, children=[]):
        self.gameState = gameState
        self.count = count
        self.reward = reward
        self.parent = parent
        self.children = children

    def setParent(self, parent):
        self.parent = parent

    def addChildren(self, child):
        self.children.append(child)
        child.setParent(self)

    def update(self, reward):
        self.reward += reward
        self.count += 1

    def visited(self):
        return self.count > 0

#####################
# Agents Base Class #
#####################
class MctsAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)

        midLine = [(gameState.data.layout.width/2-1, y) for y in range(0, gameState.data.layout.height)] if self.red \
            else [(gameState.data.layout.width/2, y) for y in range(0, gameState.data.layout.height)]

        self.midPoints = [point for point in midLine if not gameState.hasWall(point)]

        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        pass

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
        enemyGhosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
        minDistance = min([self.getMazeDistance(myPos, food) for food in enemyGhosts])
        return minDistance

    def getFoodDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        return minDistance

    def getCapsuleDistance(self, gameState):
        myPos = gameState.getAgentState(self.index)
        capsuleList = self.getCapsules(gameState).asList()
        minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
        return minDistance

    def getNumOfFoods(self, gameState):
        return len(self.getFood(gameState).count())

    def getDistanceToMid(self, gameState):
        myPos = gameState.getAgentState(self.index)
        minDistance = min([self.getMazeDistance(myPos, point) for point in self.midPoints])
        return minDistance

    def mctSearch(self, gameState, iteration):
        depth = 20
        rootNode = TreeNode(gameState,0, 0.0)

        for t in range(iteration):
            currNode = self.selection(rootNode)
            if currNode.visited():
                currNode = self.expansion(currNode)
            currReward = self.simulation(currNode, depth, gameState)
            self.backPropagation(currNode,currReward)

        return rootNode


    def getEnemyScaredTimer(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyGhosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
        timer = []
        for ghost in enemyGhosts:
            timer.append(gameState.getAgentState(ghost.index).scaredTimer)
        return min(timer)


    def selection(self, currNode):
        while len(currNode.children)>0:
            children = currNode.children
            ucbValues = [self.ucbValue(child) for child in children]
            maxValue = max(ucbValues)
            candidateNodes = [child for child, value in zip(children,ucbValues) if value == maxValue]
            currNode = random.choice(candidateNodes)
        return currNode


    def expansion(self, currNode):
        successors = self.getSuccessors(currNode.gameState)
        for successor in successors:
            currNode.addChildren(TreeNode(successor, 0, 0.0))
        currNode = currNode.children[0]
        return currNode


    def simulation(self, currNode, depth, gameState, discount = 0.9):
        totalRewards = 0
        while depth > 0:
            actions = gameState.getLegalActions(self.index)
            nextAction = random.choice(actions)
            totalRewards = totalRewards * discount + self.evaluate(gameState, nextAction)
            successor = self.getSuccessor(gameState, nextAction)
            gameState = successor
            depth -= 1
        return totalRewards


    def backPropagation(self, currNode, reward):
        while currNode is not None:
            currNode.update(reward)
            currNode = currNode.parent

    def ucbValue(self, currNode, rho = 1.0, Q0 = math.inf):
        if currNode.visited():
            confidence = math.sqrt(rho * math.log(currNode.parent.count) / currNode.count)
            ucbValue = currNode.reward + confidence
        else:
            ucbValue = Q0
        return ucbValue


###################
# Offensive Agent #
###################
class OffensiveAgent(MctsAgent):
    def chooseAction(self, gameState):
        pass


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # myState = successor.getAgentState(self.index)
        # myPos = myState.getPosition()
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        features['successorScore'] = -self.getNumOfFoods(successor)
        features['distanceToFood'] = self.getFoodDistance(successor)
        features['distanceToCapsule'] = self.getCapsuleDistance(successor)
        features['distanceToGhost'] = self.getGhostDistance(successor)
        features['distanceToMid'] = self.getDistanceToMid(successor)
        features['scaredTime'] = self.getEnemyScaredTimer(successor)
        features['stop'] = 1 if action == Directions.STOP else 0
        features['reverse'] = 1 if action == reverse else 0


    def getWeights(self, gameState, action):
        if gameState.getAgentState(self.index).numCarrying < 10:
            return {'successorScore' : 20,
                    'distanceToFood': -5,
                    'distanceToCapsule': -10,
                    'distanceToGhost': 50,
                    'stop': -100,
                    'reverse':  -2
                    }
        else:
            return {'distanceToMid': -10,
                    'distanceToGhost': 50,
                    'stop': -100,
                    'reverse': -2
                    }


###################
# Defensive Agent #
###################
class DefensiveAgent(MctsAgent):
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
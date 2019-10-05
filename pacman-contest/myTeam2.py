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
        enemy_ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
        minDistance = min([self.getMazeDistance(myPos, food) for food in enemy_ghosts])
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

class OffensiveAgent(MctsAgent):



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # myState = successor.getAgentState(self.index)
        # myPos = myState.getPosition()

        features['successorScore'] = -self.getNumOfFoods(successor)
        features['distanceToFood'] = self.getFoodDistance(successor)
        features['distanceToCapsule'] = self.getCapsuleDistance(successor)
        features['distanceToGhost'] = self.getGhostDistance(successor)
        features['distanceToMid'] = self.getDistanceToMid(successor)


    def getWeights(self, gameState, action):
        if gameState.getAgentState(self.index).numCarrying < 10:
            return {'successorScore' : 20,
                    'distanceToFood': -5,
                    'distanceToCapsule': -10,
                    'distanceToGhost': -50,
                    }
        else:
            return {'distanceToMid': -10,
                    'distanceToGhost': -50
                    }





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
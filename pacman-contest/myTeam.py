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
import util, math
from game import Directions, Actions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='Rush', second='Guard'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.home = gameState.getAgentState(self.index).getPosition()
        self.food_list = len(self.getFood(gameState).asList())
        self.defending_food_list = len(self.getFoodYouAreDefending(gameState).asList())
        self.wall_list = gameState.getWalls().asList()
        self.last_eaten_food = None
        self.eaten_foods = None

    def getIntervals(self, gameState):
        intervals = [((gameState.data.layout.width / 2) - (1 if self.red else 0), y) for y in range(0, gameState.data.layout.height)]
        return [interval for interval in intervals if interval not in self.wall_list]

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(o) for o in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) == 0:
            return None
        else:
            return invaders

    def getGuardians(self, gameState):
        enemies = [gameState.getAgentState(o) for o in self.getOpponents(gameState)]
        guardians = [a for a in enemies if a.getPosition() != None and not a.isPacman]
        return None if len(guardians) == 0 else guardians

    def getNearbyFoods(self, gameState):
        foods = [food for food in self.getFood(gameState).asList()]
        foodDistance = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a) for a in foods]
        nearby_foods = [f for f, d in zip(foods, foodDistance) if d == min(foodDistance)]
        return None if not nearby_foods else nearby_foods[0]

    def getcloseCapsule(self, gameState):
        capsules = self.getCapsules(gameState)
        if len(capsules) == 0:
            return None
        else:
            capsuleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), c) for c in capsules]
            closeCapsules = [c for c, d in zip(self.getCapsules(gameState), capsuleDis) if d == min(capsuleDis)]
            return closeCapsules[0]

    def getSuccessors(self, currentPosition):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.wall_list:
                nextPosition = (nextx, nexty)
                successors.append((nextPosition, action))
        return successors

    def simpleHeuristic(self, gameState, thisPosition):
        return 0

    def DetectOpponentsHeuristic(self, gameState, thisPosition):
        heuristics = []
        ghosts = self.getGuardians(gameState)
        if ghosts == None:
            return 0
        else:
            for o in ghosts:
                if self.getMazeDistance(thisPosition, o.getPosition()) < 3:
                    d = self.getMazeDistance(thisPosition, o.getPosition())
                    heuristics.append(math.pow((d - 5), 4))
                else:
                    heuristics.append(0)
            return max(heuristics)

    def AStarSearch(self, gameState, goal, heuristic):
        start = self.getCurrentObservation().getAgentState(self.index).getPosition()
        openSet = util.PriorityQueue()
        openSet.push((start, []), 0)
        visitedNodes = []
        while not openSet.isEmpty():
            node, trace = openSet.pop()
            if node == goal:
                if len(trace) == 0:
                    return 'Stop'

                return trace[0]
            if node not in visitedNodes:
                visitedNodes.append(node)
                successors = self.getSuccessors(node)
                for successor in successors:
                    cost = len(trace + [successor[1]]) + heuristic(gameState, successor[0])
                    if successor not in visitedNodes:
                        openSet.push((successor[0], trace + [successor[1]]), cost)
        if goal != self.home:
            return self.AStarSearch(gameState, self.home, self.DetectOpponentsHeuristic)
        return 'Stop'


class Rush(ReflexCaptureAgent):

    def chooseAction(self, gameState):

        closeCapsule = self.getcloseCapsule(gameState)
        foods = self.getFood(gameState).asList()
        nearby_foods = self.getNearbyFoods(gameState)
        middleLines = self.getIntervals(gameState)
        middleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), mi) for mi in
                     middleLines]
        closeMiddle = [m for m, d in zip(middleLines, middleDis) if d == min(middleDis)]
        middle = closeMiddle[0]
        guardians = self.getGuardians(gameState)
        invaders = self.getInvaders(gameState)


        #  if on the home side and get scared run towrads home
        if gameState.getAgentState(self.index).scaredTimer > 0 and invaders != None and not gameState.getAgentState(
                self.index).isPacman:
            for invader in invaders:
                if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                        invader.getPosition()) <= 2:
                    return self.AStarSearch(gameState, self.home, self.DetectOpponentsHeuristic)

        # if carrying foods exceed a threshold, run back mid
        if self.getScore(gameState) < 0:
            if gameState.getAgentState(self.index).numCarrying + self.getScore(gameState) > 0:
                return self.AStarSearch(gameState, middle, self.DetectOpponentsHeuristic)
            if gameState.getAgentState(self.index).numCarrying > 10:
                return self.AStarSearch(gameState, middle, self.DetectOpponentsHeuristic)

        # if time left is short or less than 2 foods remain, run back mid
        if gameState.data.timeleft < 200 or len(foods) < 3 or gameState.getAgentState(self.index).numCarrying > 28:
            if gameState.getAgentState(self.index).numCarrying > 0:
                return self.AStarSearch(gameState, middle, self.DetectOpponentsHeuristic)

        # if opponents' defender scared,run towards food
        if guardians != None:
            for defender in guardians:
                if defender.scaredTimer > 0:
                    if defender.scaredTimer > 10:
                        return self.AStarSearch(gameState, nearby_foods, self.simpleHeuristic)
                    else:
                        return self.AStarSearch(gameState, nearby_foods, self.DetectOpponentsHeuristic)

        # if there is capsule and opponent ghost not nearby, run towards capsule, otherwise, run back mid
        if closeCapsule != None:
            if guardians != None:
                for d in guardians:
                    if self.getMazeDistance(d.getPosition(), closeCapsule) < 2:
                        return self.AStarSearch(gameState, middle, self.DetectOpponentsHeuristic)
                return self.AStarSearch(gameState, closeCapsule, self.DetectOpponentsHeuristic)

        #  no capsule left, carried some amount of food, run back
        if closeCapsule == None:
            if guardians != None and gameState.getAgentState(self.index).numCarrying != 0:
                return self.AStarSearch(gameState, middle, self.DetectOpponentsHeuristic)

        return self.AStarSearch(gameState, nearby_foods, self.DetectOpponentsHeuristic)

class Guard(ReflexCaptureAgent):
    # check if any food is eaten currently
    def IsEating(self):
        if self.getPreviousObservation() is not None and len(
                self.getFoodYouAreDefending(self.getCurrentObservation()).asList()) < len(
                self.getFoodYouAreDefending(self.getPreviousObservation()).asList()):
            return True
        else:
            return False

    # get the closest food eaten
    def getEaten(self):
        defendLeft = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        lastDefend = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
        eaten = [left for left in lastDefend if left not in defendLeft]
        eatenDis = [self.getMazeDistance(self.getCurrentObservation().getAgentState(self.index).getPosition(), eat) for
                    eat in eaten]
        closeEaten = [e for e, d in zip(eaten, eatenDis) if d == min(eatenDis)]
        self.eaten_foods = closeEaten[0]
        return closeEaten[0]

    # check if any food has been eaten
    def beginEaten(self):
        if len(self.getFoodYouAreDefending(self.getCurrentObservation()).asList()) < self.defending_food_list:
            return True
        else:
            return False


    def chooseAction(self, gameState):
        invaders = self.getInvaders(gameState)

        middleLines = self.getIntervals(gameState)
        middleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), mi) for mi in
                     middleLines]
        closeMiddle = [m for m, d in zip(middleLines, middleDis) if d == min(middleDis)]
        middle = closeMiddle[0]

        #update the current foodlist
        for index in self.getOpponents(gameState):
            if self.getPreviousObservation() is not None:
                if gameState.getAgentState(index).numReturned > self.getPreviousObservation().getAgentState(
                        index).numReturned:
                    self.defending_food_list = self.defending_food_list - (gameState.getAgentState(
                        index).numReturned - self.getPreviousObservation().getAgentState(index).numReturned)

        # if come to the mid or get to the last eaten point, run back home
        if gameState.getAgentState(self.index).getPosition() == middle or gameState.getAgentState(
                self.index).getPosition() == self.eaten_foods:
            return self.AStarSearch(gameState, self.home, self.simpleHeuristic)

        # our defender scared, run back home
        if gameState.getAgentState(self.index).scaredTimer > 0 and invaders != None:
            for invader in invaders:
                if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                        invader.getPosition()) <= 2:
                    return self.AStarSearch(gameState, self.home, self.DetectOpponentsHeuristic)

        # chase the closet opponent
        if invaders != None:
            invadersDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for
                           a in invaders]
            minDIs = min(invadersDis)
            target = [a.getPosition() for a, v in zip(invaders, invadersDis) if v == minDIs]
            return self.AStarSearch(gameState, target[0], self.simpleHeuristic)

        # go to the recently eaten food
        if self.beginEaten():
            if self.IsEating():
                eaten = self.getEaten()
                self.eaten_foods = eaten
                return self.AStarSearch(gameState, eaten, self.simpleHeuristic)
            else:
                return self.AStarSearch(gameState, self.eaten_foods, self.simpleHeuristic)

        return self.AStarSearch(gameState, middle, self.simpleHeuristic)
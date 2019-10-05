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

import logging

MIN_CARRYING = 1


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='Positive', second='Negative'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def InitLogger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n%(message)s\n')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def Log(self, *iterables):
        for iter in iterables:
            self.logger.debug('{}'.format(iter))

    def registerInitialState(self, gameState):
        self.InitLogger()
        # self.Log(gameState)

        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getAgentState(self.index).getPosition()
        self.opponent_food_number = len(self.getFood(gameState).asList())
        self.self_food_number = len(self.getFoodYouAreDefending(gameState).asList())
        self.walls = gameState.getWalls().asList()
        # self.Log(self.start_position, self.opponent_food_number, self.self_food_number, self.walls)

        self.layout_height = gameState.data.layout.height
        self.layout_width = gameState.data.layout.width
        self.mid_points = []
        for y in range(0, self.layout_height):
            point = ((self.layout_width // 2) - (1 if self.red else 0), y)
            if point not in self.walls:
                self.mid_points.append(point)
        # self.Log(self.mid_points)

        self.last_eaten_food = None
        self.eaten_foods = None

    def GetNearbyOpponentPacmans(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        nearby_opponent_pacmans = [opponent for opponent in opponents if
                                   opponent.isPacman and opponent.getPosition() != None]
        # self.Log(nearby_opponent_pacmans)
        return nearby_opponent_pacmans

    def GetNearbyOpponentGhosts(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
        nearby_opponent_ghosts = [opponent for opponent in opponents if
                                  not opponent.isPacman and opponent.getPosition() != None]
        # self.Log(nearby_opponent_ghosts)
        return nearby_opponent_ghosts

    def GetNearestObject(self, objects, distances):
        min_distance = 999999
        nearest_object = None
        for i in range(len(distances)):
            distance = distances[i]
            if min_distance > distance:
                min_distance = distance
                nearest_object = objects[i]
        return nearest_object

    def GetNearestFood(self, gameState):
        agent_position = gameState.getAgentState(self.index).getPosition()
        remaining_foods = [food for food in self.getFood(gameState).asList()]
        remaining_foods_distances = [self.getMazeDistance(agent_position, food) for food in remaining_foods]
        nearest_food = self.GetNearestObject(remaining_foods, remaining_foods_distances)
        # self.Log(nearest_food)
        return nearest_food

    def GetNearestCapsule(self, gameState):
        agent_position = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        capsules_distances = [self.getMazeDistance(agent_position, capsule) for capsule in capsules]
        nearest_capsules = self.GetNearestObject(capsules, capsules_distances)
        # self.Log(nearest_capsules)
        return nearest_capsules

    def GetSuccessors(self, position):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            next_position = (int(x + dx), int(y + dy))
            if next_position not in self.walls:
                successors.append((next_position, action))
        return successors

    def nullHeuristic(self, gameState, thisPosition):
        return 0

    def DetectOpponentGhostsHeuristic(self, gameState, thisPosition):
        heuristics = [0]
        ghosts = self.GetNearbyOpponentGhosts(gameState)
        for ghost in ghosts:
            heuristics.append(999999 if self.getMazeDistance(thisPosition, ghost.getPosition()) < 2 else 0)  # < 2
        return max(heuristics)

    class Node:
        def __init__(self, states, path, cost=0):
            self.states = states
            self.path = path
            self.cost = cost

    def waStarSearch(self, gameState, goal, heuristic=nullHeuristic):
        start_position = self.getCurrentObservation().getAgentState(self.index).getPosition()
        weight = 2
        heap = util.PriorityQueue()
        heap.push(self.Node([start_position], [], 0), 0)
        visited = {start_position: 0}
        while not heap.isEmpty():
            node = heap.pop()
            states = node.states
            state = states[-1]
            path = node.path
            cost = node.cost
            if state == goal:
                return path[0] if path else 'Stop'
            for successor in self.GetSuccessors(state):
                successor_state = successor[0]
                successor_direction = successor[1]
                successor_cost = cost + heuristic(gameState, successor_state)
                if successor_state not in visited or visited[successor_state] > successor_cost:
                    visited[successor_state] = successor_cost
                    successor_states = states[:] + [successor_state]
                    successor_path = path[:] + [successor_direction]
                    heap.push(self.Node(successor_states, successor_path, successor_cost),
                              successor_cost + weight * heuristic(gameState, successor_state))
        return 'Stop'


class Positive(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        mid_distances = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), mid_point) for
                         mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)

        if not gameState.getAgentState(self.index).isPacman and nearby_pacmans and gameState.getAgentState(
                self.index).scaredTimer == 0:
            return self.waStarSearch(gameState, nearby_pacmans[0].getPosition(), self.DetectOpponentGhostsHeuristic)

        if nearby_ghosts:
            for ghost in nearby_ghosts:
                if ghost.scaredTimer > 0:
                    return self.waStarSearch(gameState, ghost.getPosition(), self.DetectOpponentGhostsHeuristic)
            if nearest_capsule:
                return self.waStarSearch(gameState, nearest_capsule, self.DetectOpponentGhostsHeuristic)
            else:
                return self.waStarSearch(gameState, nearest_mid_point, self.DetectOpponentGhostsHeuristic)

        if gameState.getAgentState(self.index).numCarrying >= MIN_CARRYING:
            return self.waStarSearch(gameState, nearest_mid_point, self.DetectOpponentGhostsHeuristic)

        return self.waStarSearch(gameState, nearest_food, self.DetectOpponentGhostsHeuristic)


class Negative(ReflexCaptureAgent):
    def IsEating(self):
        if self.getPreviousObservation() is not None and len(
                self.getFoodYouAreDefending(self.getCurrentObservation()).asList()) < len(
            self.getFoodYouAreDefending(self.getPreviousObservation()).asList()):
            return True
        else:
            return False

    def getEaten(self):
        defendLeft = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        lastDefend = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
        eaten = [left for left in lastDefend if left not in defendLeft]
        eatenDis = [self.getMazeDistance(self.getCurrentObservation().getAgentState(self.index).getPosition(), eat) for
                    eat in eaten]
        closeEaten = [e for e, d in zip(eaten, eatenDis) if d == min(eatenDis)]
        self.eaten_foods = closeEaten[0]
        return closeEaten[0]

    def beginEaten(self):
        if len(self.getFoodYouAreDefending(self.getCurrentObservation()).asList()) < self.self_food_number:
            return True
        else:
            return False

    def chooseAction(self, gameState):
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)

        self.mid_points = self.mid_points
        mid_distances = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), mi) for mi in
                         self.mid_points]
        nearest_mid_point = [m for m, d in zip(self.mid_points, mid_distances) if d == min(mid_distances)]
        nearest_mid_point = nearest_mid_point[0]
        for index in self.getOpponents(gameState):
            if self.getPreviousObservation() is not None:
                if gameState.getAgentState(index).numReturned > self.getPreviousObservation().getAgentState(
                        index).numReturned:
                    self.self_food_number = self.self_food_number - (gameState.getAgentState(
                        index).numReturned - self.getPreviousObservation().getAgentState(index).numReturned)

        if gameState.getAgentState(self.index).getPosition() == nearest_mid_point or gameState.getAgentState(
                self.index).getPosition() == self.eaten_foods:
            return self.waStarSearch(gameState, self.start_position, self.nullHeuristic)

        if gameState.getAgentState(self.index).scaredTimer > 0 and nearby_pacmans != None:
            for invader in nearby_pacmans:
                if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                        invader.getPosition()) <= 2:
                    return self.waStarSearch(gameState, self.start_position, self.DetectOpponentGhostsHeuristic)

        if nearby_pacmans:
            invadersDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for
                           a in
                           nearby_pacmans]
            minDIs = min(invadersDis) if invadersDis else 0
            target = [a.getPosition() for a, v in zip(nearby_pacmans, invadersDis) if v == minDIs]
            return self.waStarSearch(gameState, target[0], self.nullHeuristic)

        if self.beginEaten():
            if self.IsEating():
                eaten = self.getEaten()
                self.eaten_foods = eaten
                return self.waStarSearch(gameState, eaten, self.nullHeuristic)
            else:
                return self.waStarSearch(gameState, self.eaten_foods, self.nullHeuristic)

        return self.waStarSearch(gameState, nearest_mid_point, self.nullHeuristic)

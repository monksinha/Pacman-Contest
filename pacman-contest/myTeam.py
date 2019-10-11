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
from capture import SONAR_NOISE_RANGE
import logging

MIN_CARRYING = 2


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='Positive', second='Negative'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.display = 'updateDistributions'
        self.start_position = self.opponent_food_list = self.food_list = self.walls = self.layout_height \
            = self.layout_width = self.mid_points = self.eaten_foods = self.logger = self.nearest_eaten_food \
            = self.opponents_index = self.distributions = None
        self.opt_reborn_poss = {}

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
        self.opponent_food_list = self.getFood(gameState).asList()
        self.food_list = self.getFoodYouAreDefending(gameState).asList()
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

        self.eaten_foods = []
        self.nearest_eaten_food = None
        # --------------------------------------------
        self.opponents_index = self.getOpponents(gameState)
        self.opt_init_pos = {}
        # TODO check
        self.opt_init_pos[self.opponents_index[0]] = p1 = (
            self.layout_width - 1 - self.start_position[0], self.layout_height - 1 - self.start_position[1])
        self.opt_init_pos[self.opponents_index[1]] = p2 = (
            self.layout_width - 1 - self.start_position[0], self.layout_height - self.start_position[1])

        self.opt_reborn_poss[self.opponents_index[0]] = [pos for pos, _ in self.GetSuccessors(p1)]
        self.opt_reborn_poss[self.opponents_index[1]] = [pos for pos, _ in self.GetSuccessors(p2)]
        self.distributions = [util.Counter() for i in range(4)]

        legalPosition = gameState.getWalls().deepCopy()
        for col in range(self.layout_width):
            legalPosition[col] = [not x for x in legalPosition[col]]
        self.legalPosition = legalPosition.asList()
        self.prePossiblePosition = [util.Counter() for i in range(4)]
        for i in range(4):
            for pos in self.legalPosition:
                self.prePossiblePosition[i][pos] = 0
        for pos, _ in self.GetSuccessors(self.opt_init_pos[self.opponents_index[0]]):
            self.prePossiblePosition[self.opponents_index[0]][pos] = 1
        for pos, _ in self.GetSuccessors(self.opt_init_pos[self.opponents_index[1]]):
            self.prePossiblePosition[self.opponents_index[1]][pos] = 1
            # --------------------------------------------

    def GetNearbyOpponentPacmans(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.opponents_index]
        nearby_opponent_pacmans = [opponent for opponent in opponents if
                                   opponent.isPacman and opponent.getPosition() != None]
        # self.Log(nearby_opponent_pacmans)
        return nearby_opponent_pacmans

    def GetNearbyOpponentGhosts(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.opponents_index]
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

    def waStarSearch(self, goalState, heuristic=nullHeuristic):
        gameState = self.getCurrentObservation()
        start_position = gameState.getAgentState(self.index).getPosition()
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
            if state == goalState:
                return path[0] if path else 'Stop'
            for successor in self.GetSuccessors(state):
                successor_state = successor[0]
                successor_direction = successor[1]
                successor_cost = cost + heuristic(gameState, successor_state)
                if successor_state not in visited or visited[successor_state] > successor_cost:
                    visited[successor_state] = successor_cost
                    successor_states = states + [successor_state]
                    successor_path = path + [successor_direction]
                    heap.push(self.Node(successor_states, successor_path, successor_cost),
                              successor_cost + weight * heuristic(gameState, successor_state))
        return 'Stop'

    def updateDistribution(self):
        if self.getPreviousObservation() is None:
            return self.prePossiblePosition
        pre_distances = self.getPreviousObservation().getAgentDistances()
        pre_position = self.getPreviousObservation().getAgentPosition(self.index)

        cur_distances = self.getCurrentObservation().getAgentDistances()
        cur_position = self.getCurrentObservation().getAgentPosition(self.index)

        delta = (SONAR_NOISE_RANGE - 1) / 2

        for i in range(2):
            op_idx = self.opponents_index[i]
            for pos in self.legalPosition:
                self.distributions[op_idx][pos] = 0
                if cur_distances[op_idx] - delta <= util.manhattanDistance(cur_position, pos) \
                        and util.manhattanDistance(pre_position, pos) <= pre_distances[op_idx] + delta + 1:
                    self.distributions[op_idx][pos] += 0.5
                if pre_distances[op_idx] - delta - 1 <= util.manhattanDistance(pre_position, pos) \
                        and util.manhattanDistance(cur_position, pos) <= cur_distances[op_idx] + delta:
                    self.distributions[op_idx][pos] += 0.5

            for pos in self.legalPosition:
                if self.distributions[op_idx][pos] != 1:
                    self.distributions[op_idx][pos] = 0

            cur_possible = util.Counter()
            for pos in self.prePossiblePosition[op_idx].keys():
                if self.prePossiblePosition[op_idx][pos] == 1:
                    cur_possible[pos] = 1
                    for succ in self.GetSuccessors(pos):
                        cur_possible[succ[0]] = 1
            isSeen = False
            for pos in cur_possible.keys():
                if self.distributions[op_idx][pos] == 0:
                    cur_possible[pos] = 0
                elif cur_possible[pos] != 0:
                    isSeen = True
            if isSeen:
                self.distributions[op_idx] = util.Counter()
                self.prePossiblePosition[op_idx] = util.Counter()
                self.distributions[op_idx] = cur_possible.copy()
                self.prePossiblePosition[op_idx] = cur_possible.copy()
            else:
                self.distributions[op_idx] = util.Counter()
                self.prePossiblePosition[op_idx] = util.Counter()
                for rebornPos in self.opt_reborn_poss[op_idx]:
                    self.prePossiblePosition[op_idx][rebornPos] = 1
                    self.distributions[op_idx][rebornPos] = 1


class Positive(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        current_position = gameState.getAgentState(self.index).getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)

        self.updateDistribution()
        self.displayDistributionsOverPositions(self.distributions)

        if not gameState.getAgentState(self.index).isPacman and nearby_pacmans and gameState.getAgentState(
                self.index).scaredTimer == 0:
            return self.waStarSearch(nearby_pacmans[0].getPosition(), self.DetectOpponentGhostsHeuristic)

        if nearby_ghosts:
            for ghost in nearby_ghosts:
                if ghost.scaredTimer > 0:
                    return self.waStarSearch(ghost.getPosition(), self.DetectOpponentGhostsHeuristic)
            if nearest_capsule:
                return self.waStarSearch(nearest_capsule, self.DetectOpponentGhostsHeuristic)
            else:
                return self.waStarSearch(nearest_mid_point, self.DetectOpponentGhostsHeuristic)

        if gameState.getAgentState(self.index).numCarrying >= MIN_CARRYING:
            return self.waStarSearch(nearest_mid_point, self.DetectOpponentGhostsHeuristic)

        return self.waStarSearch(nearest_food, self.DetectOpponentGhostsHeuristic)


class Negative(ReflexCaptureAgent):
    def GetFoodEatenByOpponent(self):
        current_state = self.getCurrentObservation()
        current_position = current_state.getAgentState(self.index).getPosition()
        previous_state = self.getPreviousObservation()
        if previous_state:
            current_foods = self.getFoodYouAreDefending(current_state).asList()
            previous_foods = self.getFoodYouAreDefending(previous_state).asList()
            eaten_foods = [food for food in previous_foods if food not in current_foods]
            # self.Log(eaten_foods)
            if eaten_foods:
                eaten_foods_distance = [self.getMazeDistance(current_position, food) for food in eaten_foods]
                self.nearest_eaten_food = self.GetNearestObject(eaten_foods, eaten_foods_distance)
                if self.nearest_eaten_food != None:
                    self.eaten_foods.append(self.nearest_eaten_food)

    def IsOpponentEating(self):
        current_foods = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        return len(current_foods) < len(self.food_list)

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
        # self.Log(eaten)
        eatenDis = [self.getMazeDistance(self.getCurrentObservation().getAgentState(self.index).getPosition(), eat) for
                    eat in eaten]
        closeEaten = [e for e, d in zip(eaten, eatenDis) if d == min(eatenDis)]
        self.eaten_foods = closeEaten[0]
        return closeEaten[0]

    # check if any food has been eaten
    def beginEaten(self):
        if len(self.getFoodYouAreDefending(self.getCurrentObservation()).asList()) < len(self.food_list):
            return True
        else:
            return False

    def chooseAction(self, gameState):
        current_state = gameState.getAgentState(self.index)
        current_position = current_state.getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)
        nearby_pacmans_distance = [self.getMazeDistance(current_position, pacman.getPosition()) for pacman in
                                   nearby_pacmans]
        nearest_pacman = self.GetNearestObject(nearby_pacmans, nearby_pacmans_distance)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        self.GetFoodEatenByOpponent()
        # self.Log(nearby_pacmans)
        # self.Log(nearest_pacman)

        if self.nearest_eaten_food != None and current_position == self.nearest_eaten_food:
            return self.waStarSearch(nearest_mid_point, self.nullHeuristic)

        if current_position == nearest_mid_point or current_position == self.eaten_foods:
            return self.waStarSearch(self.start_position, self.nullHeuristic)

        if nearby_pacmans:
            if current_state.scaredTimer > 0:
                for pacman in nearby_pacmans:
                    if self.getMazeDistance(current_position, pacman.getPosition()) <= 1:
                        return self.waStarSearch(self.start_position, self.DetectOpponentGhostsHeuristic)
            if nearby_pacmans is not None:
                return self.waStarSearch(nearby_pacmans, self.DetectOpponentGhostsHeuristic)
            return self.waStarSearch(nearest_mid_point, self.nullHeuristic)

        # self.Log(self.nearest_eaten_food)
        if self.nearest_eaten_food is not None:
            return self.waStarSearch(self.nearest_eaten_food, self.DetectOpponentGhostsHeuristic)

        return self.waStarSearch(nearest_mid_point, self.nullHeuristic)

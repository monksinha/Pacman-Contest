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
from capture import SONAR_NOISE_RANGE, SCARED_TIME
import logging

MIN_CARRYING = 2


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='Negative', second='Friendly'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    _instances = []
    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.teammate_index = 2 if index == 0 else 0 if index == 2 else 1 if index == 3 else 3

        self.display = 'updateDistributions'
        self.start_position = self.opponent_food_list = self.food_list = self.walls = self.layout_height \
            = self.layout_width = self.mid_points = self.eaten_foods = self.logger = self.nearest_eaten_food \
            = self.opponents_index = self.distributions = None
        self.opt_reborn_poss = {}
        self.opt_init_pos = {}

        self._instances.append(self)

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
        # if self.index == self.getTeam(gameState)[0]:
        #     self.teammate_idx =self.getTeam(gameState)[1]
        # else:
        #     self.teammate_idx =self.getTeam(gameState)[0]
        self.teammate = self._instances[self.teammate_index//2]

        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getInitialAgentPosition(self.index)
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
        self.opt_init_pos[self.opponents_index[0]] = p1 = gameState.getInitialAgentPosition(self.opponents_index[0])
        self.opt_init_pos[self.opponents_index[1]] = p2 = gameState.getInitialAgentPosition(self.opponents_index[1])

        self.opt_reborn_poss[self.opponents_index[0]] = [pos for pos, _ in self.GetSuccessors(p1)]
        self.opt_reborn_poss[self.opponents_index[1]] = [pos for pos, _ in self.GetSuccessors(p2)]
        self.distributions = [util.Counter() for i in range(4)]

        legalPosition = gameState.getWalls().deepCopy()
        for col in range(self.layout_width):
            legalPosition[col] = [not x for x in legalPosition[col]]
        self.legalPosition = legalPosition.asList()
        # init start position possibility
        self.prePossiblePosition = [util.Counter() for i in range(4)]
        for idx in self.opponents_index:
            self.initStartPositionPossibility(idx)

        # --------------------------------------------

    def initStartPositionPossibility(self, index):
        self.prePossiblePosition[index] = util.Counter()
        self.prePossiblePosition[index][self.opt_init_pos[index]] = 1
        for pos, _ in self.GetSuccessors(self.opt_init_pos[index]):
            self.prePossiblePosition[index][pos] = 1

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

    def manhattanHeuristic(self, pos, goal):
        "The Manhattan distance heuristic for a PositionSearchProblem"
        xy1 = pos
        xy2 = goal
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def DetectOpponentGhostsHeuristic(self, thisPosition, goal):
        heuristics = []
        heuristics.append(self.manhattanHeuristic(thisPosition, goal))
        ghosts = self.GetNearbyOpponentGhosts(self.getCurrentObservation())
        for ghost in ghosts:
            if self.getMazeDistance(thisPosition, ghost.getPosition()) < 2:
                heuristics.append(999999)
        return max(heuristics)

    class Node:
        def __init__(self, states, path, cost=0):
            self.states = states
            self.path = path
            self.cost = cost

    def waStarSearch(self, goal, heuristic=nullHeuristic):
        gameState = self.getCurrentObservation()
        start_position = gameState.getAgentState(self.index).getPosition()
        weight = 2
        heap = util.PriorityQueue()
        heap.push(self.Node([start_position], [], 0), 0)
        visited = {start_position: 0}
        while not heap.isEmpty():
            node = heap.pop()
            states = node.states
            pos = states[-1]
            path = node.path
            cost = node.cost
            if pos == goal:
                return path[0] if path else 'Stop'
            for successor in self.GetSuccessors(pos):
                successor_state = successor[0]
                successor_direction = successor[1]
                successor_cost = cost + heuristic(successor_state, goal)
                if successor_state not in visited or visited[successor_state] > successor_cost:
                    visited[successor_state] = successor_cost
                    successor_states = states + [successor_state]
                    successor_path = path + [successor_direction]
                    heap.push(self.Node(successor_states, successor_path, successor_cost),
                              successor_cost + weight * heuristic(successor_state, goal))
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
            op_position = self._instances[self.index//2].getCurrentObservation().getAgentPosition(op_idx)
            if op_position is not None:
                self.distributions[op_idx] = util.Counter()
                self.prePossiblePosition[op_idx] = util.Counter()
                self.distributions[op_idx][op_position] = 1
                self.prePossiblePosition[op_idx][op_position] = 1
                if op_position == cur_position:
                    self.initStartPositionPossibility(op_idx)
                    self.distributions[op_idx] = self.prePossiblePosition.copy()
                continue
            for pos in self.legalPosition:
                self.distributions[op_idx][pos] = 0
                if cur_distances[op_idx] - delta <= util.manhattanDistance(cur_position, pos) \
                        <= cur_distances[op_idx] + delta and pre_distances[op_idx] - delta - 1 <= \
                        util.manhattanDistance(pre_position, pos) <= pre_distances[op_idx] + delta + 1:
                    self.distributions[op_idx][pos] = 1

            cur_possible = util.Counter()
            for pos in self.prePossiblePosition[op_idx].keys():
                if self.prePossiblePosition[op_idx][pos] == 1:
                    cur_possible[pos] = 1
                    for succ in self.GetSuccessors(pos):
                        cur_possible[succ[0]] = 1
            for pos in cur_possible.keys():
                if util.manhattanDistance(pos, cur_position) <= 5:
                    cur_possible[pos] = 0

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
        return self.distributions


class Positive(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        self.Log(self.teammate)
        current_position = gameState.getAgentState(self.index).getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)

        # self.displayDistributionsOverPositions(self.updateDistribution())

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

    def chooseAction(self, gameState):
        # self.Log(teammate_state)
        self.Log(self.teammate)

        teammate_state = gameState.getAgentState(self.teammate_index)

        # self.displayDistributionsOverPositions(self.updateDistribution())
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
                return self.waStarSearch(nearby_pacmans[0].getPosition(), self.DetectOpponentGhostsHeuristic)
            return self.waStarSearch(nearest_mid_point, self.nullHeuristic)

        # self.Log(self.nearest_eaten_food)
        if self.nearest_eaten_food is not None:
            return self.waStarSearch(self.nearest_eaten_food, self.DetectOpponentGhostsHeuristic)

        return self.waStarSearch(nearest_mid_point, self.nullHeuristic)


class Friendly(ReflexCaptureAgent):
    def __init__(self, gameState):
        ReflexCaptureAgent.__init__(self, gameState)
        self.mTerritory = {}
        self.invincible_state = (False, 0)

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.territory()

    def territory(self):
        x0, _ = self.start_position
        for pos in self.legalPosition:
            x, _ = pos
            # TODO check self.layout_width // 2
            if abs(x - x0) < self.layout_width // 2 - 1:
                self.mTerritory[pos] = True
            else:
                self.mTerritory[pos] = False

    def isInvade(self, op_idx):
        count = 0
        for pos in self.distributions[op_idx].keys():
            if self.distributions[op_idx][pos] != 0:
                if self.mTerritory[pos]:
                    count += 1
                else:
                    count -= 1
        return True if count > 0 else False

    def chooseAction(self, gameState):
        current_position = gameState.getAgentState(self.index).getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)
        capsules = self.getCapsules(self.getPreviousObservation() if self.getPreviousObservation() else gameState)
        carry_points = self.getCurrentObservation().getAgentState(self.index).numCarrying

        self.displayDistributionsOverPositions(self.updateDistribution())

        # print(self.isInvade(self.opponents_index[0]))

        def evalution():
            nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
            if carry_points > 5:
                return 'escape', nearest_mid_point
            if nearby_ghosts:
                isInv, leftTime = self.invincible_state
                d = []
                chase = []
                for g in nearby_ghosts:
                    dis = self.getMazeDistance(current_position, g.getPosition())
                    d.append(dis)
                    if isInv and leftTime > dis:
                        chase.append((g.getPosition(), dis))

                if chase != []:
                    chase.sort(key=lambda x: x[1])
                    return 'chase', chase[0][0]
                else:
                    if min(d) >= 8:
                        return 'eat_more', nearest_food
                    if 5 < min(d) < 8:
                        return 'sneak', certainFood()
                return 'escape', nearest_mid_point

            if min(eval_dist()) >= 8:
                return 'eat_more', nearest_food
            if 5 <= min(eval_dist()) < 8:
                return 'sneak', certainFood()

            return 'eat_more', nearest_food

        def eval_dist():
            dists = []
            cur_position = self.getCurrentObservation().getAgentState(self.index).getPosition()

            def nPP(idx):
                curMin = 9999
                distribution = self.distributions[idx]
                for pos in distribution.keys():
                    if distribution[pos] != 0:
                        curMin = min(self.getMazeDistance(cur_position, pos), curMin)
                return curMin

            for i in self.opponents_index:
                dists.append(nPP(i))
            return dists

        def nearestExit():
            return nearest_mid_point

        def certainFood():
            x = lambda f: self.getMazeDistance(f, current_position)
            food = accessibleFood()
            dists = []
            if food == []:
                return current_position
            for f in food:
                dists.append((f, x(f)))

            dists.sort(key=lambda x: x[1])
            return dists[0][0]

        def accessibleFood():
            food = self.getFood(gameState).asList()
            accessible_food = []
            for f in food:
                isRisky = False
                for op in self.opponents_index:
                    for pos in self.distributions[op]:
                        if self.distributions[op][pos] != 0 and util.manhattanDistance(pos, f) < 5:
                            isRisky = True
                if not isRisky:
                    accessible_food.append(f)
            return accessible_food

        def update_Invincible():
            isInvincible, leftTime = self.invincible_state
            if current_position in capsules:
                self.invincible_state = (True, SCARED_TIME)
            elif isInvincible:
                if leftTime - 1 > 0:
                    self.invincible_state = (True, leftTime - 1)
                else:
                    self.invincible_state = (False, 0)

        strategy, goal = evalution()
        update_Invincible()
        print(self.invincible_state)

        if strategy == 'eat_more':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'escape':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'sneak':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'chase':
            return self.waStarSearch(goal, self.manhattanHeuristic)

        #
        # if nearby_ghosts:
        #     for ghost in nearby_ghosts:
        #         if ghost.scaredTimer > 0:
        #             return self.waStarSearch(ghost.getPosition(), self.DetectOpponentGhostsHeuristic)
        #     if nearest_capsule:
        #         return self.waStarSearch(nearest_capsule, self.DetectOpponentGhostsHeuristic)
        #     else:
        #         return self.waStarSearch(nearest_mid_point, self.DetectOpponentGhostsHeuristic)
        #
        # if gameState.getAgentState(self.index).numCarrying >= MIN_CARRYING:
        #     return self.waStarSearch(nearest_mid_point, self.DetectOpponentGhostsHeuristic)
        #
        # return self.waStarSearch(nearest_food, self.DetectOpponentGhostsHeuristic)
        #
        #

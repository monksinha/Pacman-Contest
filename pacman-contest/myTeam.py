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

def createTeam(firstIndex, secondIndex, isRed, first='Friendly', second='Negative'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    _instances = []
    distributions = [util.Counter() for i in range(4)]
    prePossiblePosition = [util.Counter() for i in range(4)]

    def __init__(self, index, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.teammate_index = 2 if index == 0 else 0 if index == 2 else 1 if index == 3 else 3

        self.display = 'updateDistributions'
        self.start_position = self.opponent_food_list = self.food_list = self.walls = self.layout_height \
            = self.layout_width = self.mid_points = self.eaten_foods = self.logger = self.nearest_eaten_food \
            = self.opponents_index = None
        self.opt_reborn_poss = {}
        self.opt_init_pos = {}
        self.mTerritory = {}
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
        # if self.index == self.getTeam(gameState)[0]:
        #     self.teammate_idx =self.getTeam(gameState)[1]
        # else:
        #     self.teammate_idx =self.getTeam(gameState)[0]
        self.teammate = self._instances[self.teammate_index // 2]

        self.destination = None

        CaptureAgent.registerInitialState(self, gameState)
        self.start_position = gameState.getInitialAgentPosition(self.index)
        self.opponent_food_list = self.getFood(gameState).asList()
        self.food_list = self.getFoodYouAreDefending(gameState).asList()
        self.walls = gameState.getWalls().asList()

        self.layout_height = gameState.data.layout.height
        self.layout_width = gameState.data.layout.width
        self.mid_points = []
        for y in range(0, self.layout_height):
            point = ((self.layout_width // 2) - (1 if self.red else 0), y)
            if point not in self.walls:
                self.mid_points.append(point)

        self.eaten_foods = []
        self.eaten_foods_distance = []
        self.nearest_eaten_food = None
        self.opt_mid_points = []
        for y in range(0, self.layout_height):
            point = ((self.layout_width // 2) - (0 if self.red else 1), y)
            if point not in self.walls:
                self.opt_mid_points.append(point)

        # --------------------------------------------
        self.pre_op_idx = (4 + self.index - 1) % 4
        self.next_op_idx = (4 + self.index + 1) % 4

        self.opponents_index = self.getOpponents(gameState)
        self.opt_init_pos[self.opponents_index[0]] = p1 = gameState.getInitialAgentPosition(self.opponents_index[0])
        self.opt_init_pos[self.opponents_index[1]] = p2 = gameState.getInitialAgentPosition(self.opponents_index[1])

        self.opt_reborn_poss[self.opponents_index[0]] = [pos for pos, _ in self.GetSuccessors(p1)] + [p1]
        self.opt_reborn_poss[self.opponents_index[1]] = [pos for pos, _ in self.GetSuccessors(p2)] + [p2]
        # self.distributions = [util.Counter() for i in range(4)]

        legalPosition = gameState.getWalls().deepCopy()
        for col in range(self.layout_width):
            legalPosition[col] = [not x for x in legalPosition[col]]
        self.legalPosition = legalPosition.asList()
        # init start position possibility
        # self.prePossiblePosition = [util.Counter() for i in range(4)]
        for idx in self.opponents_index:
            self.initStartPositionPossibility(idx)

        self.midX = gameState.data.layout.width // 2 - 1 if self.red else gameState.data.layout.width // 2
        self.territory()
        # --------------------------------------------

    def initStartPositionPossibility(self, index):
        # self.prePossiblePosition[index] = util.Counter()
        self.prePossiblePosition[index][self.opt_init_pos[index]] = 1
        for pos, _ in self.GetSuccessors(self.opt_init_pos[index]):
            self.prePossiblePosition[index][pos] = 1

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
                    count += self.distributions[op_idx][pos]
                else:
                    count -= self.distributions[op_idx][pos]
        return True if count > 0 else False

    def GetNearbyOpponentPacmans(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.opponents_index]
        nearby_opponent_pacmans = [opponent for opponent in opponents if
                                   opponent.isPacman and opponent.getPosition() != None]
        return nearby_opponent_pacmans

    def GetNearbyOpponentGhosts(self, gameState):
        opponents = [gameState.getAgentState(opponent) for opponent in self.opponents_index]
        nearby_opponent_ghosts = [opponent for opponent in opponents if
                                  not opponent.isPacman and opponent.getPosition() != None]
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

    def GetFurthestObject(self, objects, distances):
        max_distance = 0
        furthest_object = None
        for i in range(len(distances)):
            distance = distances[i]
            if max_distance < distance:
                max_distance = distance
                furthest_object = objects[i]
        return furthest_object

    def GetNearestFood(self, gameState):
        agent_position = gameState.getAgentState(self.index).getPosition()
        remaining_foods = [food for food in self.getFood(gameState).asList()]
        remaining_foods_distances = [self.getMazeDistance(agent_position, food) for food in remaining_foods]
        nearest_food = self.GetNearestObject(remaining_foods, remaining_foods_distances)
        return nearest_food

    def GetNearestCapsule(self, gameState):
        agent_position = gameState.getAgentState(self.index).getPosition()
        capsules = self.getCapsules(gameState)
        capsules_distances = [self.getMazeDistance(agent_position, capsule) for capsule in capsules]
        nearest_capsules = self.GetNearestObject(capsules, capsules_distances)
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

    def noCrossingHeuristic(self, pos, goal):
        heuristics = []
        heuristics.append(self.manhattanHeuristic(pos, goal))
        (x, y) = pos
        if (self.red and x > self.midX) or (not self.red and x < self.midX):
            heuristics.append(99)
        return max(heuristics)

    def DetectOpponentGhostsHeuristic(self, pos, goal):
        heuristics = []
        heuristics.append(self.manhattanHeuristic(pos, goal))
        ghosts = self.GetNearbyOpponentGhosts(self.getCurrentObservation())
        for ghost in ghosts:
            if self.getMazeDistance(pos, ghost.getPosition()) < 5:
                heuristics.append(999)
        return max(heuristics)

    def DetectOpponentPacmansHeuristic(self, pos, goal):
        heuristics = []
        heuristics.append(self.manhattanHeuristic(pos, goal))
        pacmans = self.GetNearbyOpponentPacmans(self.getCurrentObservation())
        for pacman in pacmans:
            if self.getMazeDistance(pos, pacman.getPosition()) < 2:
                heuristics.append(999999)
        return max(heuristics)

    def changeGateHeuristic(self, pos, goal):
        heuristics = []
        heuristics.append(self.manhattanHeuristic(pos, goal))
        if pos in self.opt_mid_points:
            heuristics.append(99)
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
                successor_cost = cost + 1
                if successor_state not in visited or visited[successor_state] > successor_cost:
                    visited[successor_state] = successor_cost
                    successor_states = states + [successor_state]
                    successor_path = path + [successor_direction]
                    heap.push(self.Node(successor_states, successor_path, successor_cost),
                              successor_cost + weight * heuristic(successor_state, goal))
        return 'Stop'

    def waStarSearchFullPath(self, goal, heuristic=nullHeuristic):
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
                return states if path else []
            for successor in self.GetSuccessors(pos):
                successor_state = successor[0]
                successor_direction = successor[1]
                successor_cost = cost + 1
                if successor_state not in visited or visited[successor_state] > successor_cost:
                    visited[successor_state] = successor_cost
                    successor_states = states + [successor_state]
                    successor_path = path + [successor_direction]
                    heap.push(self.Node(successor_states, successor_path, successor_cost),
                              successor_cost + weight * heuristic(successor_state, goal))
        return []

    def updateDistribution(self):
        if self.getPreviousObservation is None:
            return ReflexCaptureAgent.prePossiblePosition
        my_cur_gs = self.getCurrentObservation()
        my_pre_gs = self.getPreviousObservation() if self.getPreviousObservation() is not None else my_cur_gs
        tm_cur_gs = self.teammate.getCurrentObservation() if self.teammate.observationHistory != [] else my_cur_gs
        tm_pre_gs = self.teammate.getPreviousObservation() if len(self.teammate.observationHistory) > 1 else tm_cur_gs
        pre_distances = my_pre_gs.getAgentDistances()
        pre_position = my_pre_gs.getAgentPosition(self.index)
        cur_distances = my_cur_gs.getAgentDistances()
        cur_position = my_cur_gs.getAgentPosition(self.index)

        pre_distances1 = tm_pre_gs.getAgentDistances()
        pre_position1 = tm_pre_gs.getAgentPosition(self.teammate_index)
        cur_distances1 = tm_cur_gs.getAgentDistances()
        cur_position1 = tm_cur_gs.getAgentPosition(self.teammate_index)

        delta = (SONAR_NOISE_RANGE - 1) / 2

        pre_op_idx = (4 + self.index - 1) % 4
        next_op_idx = (4 + self.index + 1) % 4
        for op_idx in [pre_op_idx, next_op_idx]:
            op_position = my_cur_gs.getAgentPosition(op_idx)
            if op_position is not None:
                ReflexCaptureAgent.distributions[op_idx] = util.Counter()
                ReflexCaptureAgent.prePossiblePosition[op_idx] = util.Counter()
                ReflexCaptureAgent.distributions[op_idx][op_position] = 1
                ReflexCaptureAgent.prePossiblePosition[op_idx][op_position] = 1
                if op_position == cur_position:
                    self.initStartPositionPossibility(op_idx)
                    ReflexCaptureAgent.distributions[op_idx] = ReflexCaptureAgent.prePossiblePosition[op_idx].copy()
                continue

            for pos in self.legalPosition:
                ReflexCaptureAgent.distributions[op_idx][pos] = 0
                if op_idx == pre_op_idx:
                    if cur_distances[op_idx] - delta <= util.manhattanDistance(cur_position, pos) \
                            <= cur_distances[op_idx] + delta and pre_distances[op_idx] - delta - 1 <= \
                            util.manhattanDistance(pre_position, pos) <= pre_distances[op_idx] + delta + 1 \
                            and cur_distances1[op_idx] - delta - 1 <= util.manhattanDistance(cur_position1, pos) \
                            <= cur_distances1[op_idx] + delta + 1 and pre_distances1[op_idx] - delta - 2 <= \
                            util.manhattanDistance(pre_position1, pos) <= pre_distances1[op_idx] + delta + 2:
                        ReflexCaptureAgent.distributions[op_idx][pos] = 1
                else:
                    if cur_distances[op_idx] - delta - 1 <= util.manhattanDistance(cur_position, pos) \
                            <= cur_distances[op_idx] + delta + 1 and pre_distances[op_idx] - delta - 2 <= \
                            util.manhattanDistance(pre_position, pos) <= pre_distances[op_idx] + delta + 2 \
                            and cur_distances1[op_idx] - delta <= util.manhattanDistance(cur_position1, pos) \
                            <= cur_distances1[op_idx] + delta and pre_distances1[op_idx] - delta - 1 <= \
                            util.manhattanDistance(pre_position1, pos) <= pre_distances1[op_idx] + delta + 1:
                        ReflexCaptureAgent.distributions[op_idx][pos] = 1

            cur_possible = util.Counter()
            for pos in ReflexCaptureAgent.prePossiblePosition[op_idx].keys():
                if ReflexCaptureAgent.prePossiblePosition[op_idx][pos] != 0:
                    cur_possible[pos] = 1
                    if op_idx == pre_op_idx:
                        for p, _ in self.GetSuccessors(pos):
                            cur_possible[p] = 1
            for pos in cur_possible.keys():
                if util.manhattanDistance(pos, cur_position) <= 5:
                    cur_possible[pos] = 0
                # if util.manhattanDistance(pos, cur_position1) <= 5:
                #     cur_possible[pos] = 0

            isSeen = False
            for pos in cur_possible.keys():
                if ReflexCaptureAgent.distributions[op_idx][pos] == 0:
                    cur_possible[pos] = 0
                elif cur_possible[pos] != 0:
                    isSeen = True
            if isSeen:
                ReflexCaptureAgent.distributions[op_idx] = util.Counter()
                ReflexCaptureAgent.prePossiblePosition[op_idx] = util.Counter()
                ReflexCaptureAgent.distributions[op_idx] = cur_possible.copy()
                ReflexCaptureAgent.prePossiblePosition[op_idx] = cur_possible.copy()
            else:
                ReflexCaptureAgent.distributions[op_idx] = util.Counter()
                ReflexCaptureAgent.prePossiblePosition[op_idx] = util.Counter()
                for rebornPos in self.opt_reborn_poss[op_idx]:
                    ReflexCaptureAgent.prePossiblePosition[op_idx][rebornPos] = 1
                    ReflexCaptureAgent.distributions[op_idx][rebornPos] = 1
                    for p, _ in self.GetSuccessors(rebornPos):
                        ReflexCaptureAgent.prePossiblePosition[op_idx][p] = 1
                        ReflexCaptureAgent.distributions[op_idx][p] = 1
        for i in range(4):
            l = len([k for k in ReflexCaptureAgent.distributions[i] if ReflexCaptureAgent.distributions[i][k] != 0])
            ReflexCaptureAgent.distributions[i].divideAll(l) if l != 0 else None
        return ReflexCaptureAgent.distributions

    def UpdateFoodList(self, gameState):
        pos = gameState.getAgentState(self.index).getPosition()
        current_food_list = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        self.nearest_eaten_food = None
        if len(current_food_list) > len(self.food_list):
            self.food_list = current_food_list
            return

        eaten_foods = [food for food in self.food_list if food not in current_food_list]
        eaten_foods_distance = [self.getMazeDistance(pos, food) for food in eaten_foods]
        self.food_list = current_food_list
        if eaten_foods:
            self.nearest_eaten_food = self.GetNearestObject(eaten_foods, eaten_foods_distance)
        # self.Log(self.nearest_eaten_food)


class Positive(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        current_position = gameState.getAgentState(self.index).getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)

        self.displayDistributionsOverPositions(self.updateDistribution())

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

    def chooseAction(self, gameState):
        teammate_state = gameState.getAgentState(self.teammate_index)

        self.displayDistributionsOverPositions(self.updateDistribution())

        current_state = gameState.getAgentState(self.index)
        current_position = current_state.getPosition()
        mid_distances = [self.getMazeDistance(current_position, mid_point) for mid_point in self.mid_points]
        nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        furthest_mid_point = self.GetFurthestObject(self.mid_points, mid_distances)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)
        nearby_pacmans_distance = [self.getMazeDistance(current_position, pacman.getPosition()) for pacman in
                                   nearby_pacmans]
        nearest_pacman = self.GetNearestObject(nearby_pacmans, nearby_pacmans_distance)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        self.UpdateFoodList(gameState)

        if nearby_pacmans:
            self.destination = nearby_pacmans[0].getPosition()
            if current_state.scaredTimer > 0:
                for pacman in nearby_pacmans:
                    if self.getMazeDistance(current_position, pacman.getPosition()) <= 2:
                        self.destination = self.start_position
            return self.waStarSearch(self.destination, self.DetectOpponentPacmansHeuristic)

        elif self.nearest_eaten_food is not None:
            return self.waStarSearch(self.nearest_eaten_food, self.noCrossingHeuristic)

        else:
            for op in self.opponents_index:
                _min = 999
                goal = current_position
                if self.isInvade(op):
                    for p in self.distributions[op].keys():
                        if self.distributions[op][p] != 0 and p in self.mTerritory:
                            dis = self.getMazeDistance(p, current_position)
                            if dis < _min:
                                goal = p
                                _min = dis
                if goal != current_position:
                    return self.waStarSearch(goal, self.noCrossingHeuristic)
        if self.destination is None:
            self.destination = nearest_mid_point
            return self.waStarSearch(nearest_mid_point, self.noCrossingHeuristic)

        elif current_position == self.destination:
            self.destination = furthest_mid_point

        return self.waStarSearch(self.destination, self.noCrossingHeuristic)


class Friendly(ReflexCaptureAgent):
    def __init__(self, gameState):
        ReflexCaptureAgent.__init__(self, gameState)
        self.mTerritory = {}
        self.invincible_state = (False, 0)
        self.entrance = []

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.territory()
        for x, y in self.mid_points:
            op_x = x + 1 if self.red else x - 1
            if (op_x, y) in self.opt_mid_points:
                self.entrance.append((x, y))

        self.weight_gate = util.Counter()
        for pos in self.entrance:
            self.weight_gate[pos] = 1

        roadEnd = []
        self.lane = {}
        self.lane_end_start = {}
        for pos in self.legalPosition:
            l = len(self.GetSuccessors(pos))
            if l == 1:
                roadEnd.append(pos)
        for p in roadEnd:
            road = [p]
            while True:
                succ = self.GetSuccessors(road[-1])
                if len(succ) > 2:
                    break
                else:
                    for i in succ:
                        if i[0] not in road:
                            road.append(i[0])
            self.lane[p] = road
            self.lane_end_start[p] = road[-1]

        # print(self.lane)

        # print(self.lane_food)

    def update_lane_food(self):
        self.lane_food = {}
        gameState = self.getCurrentObservation()
        pos = gameState.getAgentState(self.index).getPosition()
        remaining_foods = [food for food in self.getFood(gameState).asList()]
        for f in remaining_foods:
            self.lane_food[f] = 0
            for lane in self.lane.keys():
                if f in self.lane[lane]:
                    self.lane_food[f] = max(self.getMazeDistance(f, self.lane_end_start[lane]), self.lane_food[f])

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
                    count += self.distributions[op_idx][pos]
                else:
                    count -= self.distributions[op_idx][pos]
        return True if count > 0 else False

    def updateGate(self):
        gate = util.Counter()
        _max = 0

        def cal_nearest_food(pos):
            _min = 999
            for f in self.getFood(self.getCurrentObservation()).asList():
                dist = self.getMazeDistance(f, pos)
                _min = min(_min, dist)
            return _min

        for pos in self.entrance:
            d = cal_nearest_food(pos)
            gate[pos] = d
            _max = max(_max, d)
        for pos in gate.keys():
            gate[pos] = _max / gate[pos]
        gate.normalize()
        return gate

    def chooseAction(self, gameState):

        def show_lane():
            showLane = util.Counter()
            for i in self.lane.keys():
                for p in self.lane[i]:
                    showLane[p] = 1
            self.displayDistributionsOverPositions([showLane])

        def show_legalpos():
            show_lp = util.Counter()
            for i in self.lane_food:
                if self.lane_food[i]!=0:
                    show_lp[i] = 1
            self.displayDistributionsOverPositions([show_lp])

        # show_legalpos()
        # show_lane()

        def bestExit():
            markedExit = []
            best = None
            _min = 999
            for pos in self.entrance:
                flag = True
                for op in self.opponents_index:
                    distri = self.distributions[op]
                    for enemyPos in distri.keys():
                        if distri[enemyPos] != 0:
                            if self.getMazeDistance(pos, enemyPos) <= 5 * distri[enemyPos] \
                                    and self.getMazeDistance(enemyPos, pos) < self.getMazeDistance(cur_pos, pos):
                                flag = False
                                break
                if flag:
                    markedExit.append(pos)
            for pos in markedExit:
                dist = self.getMazeDistance(pos, cur_pos)
                if dist < _min:
                    best = pos
            if best is not None:
                # print(best)
                return best
            else:
                return nearest_capsule

        def canSurvive():
            exitPath = self.waStarSearchFullPath(bestExit(), self.DetectOpponentGhostsHeuristic)
            # print(exitPath)
            _keyPos = cur_pos
            delta = 0
            for op_idx in self.opponents_index:
                _min = 999
                for epPos in exitPath:
                    for enemyP in [k for k in self.distributions[op_idx].keys() if self.distributions[op_idx][k] > 0]:
                        if _min > self.getMazeDistance(epPos, enemyP):
                            _min = self.getMazeDistance(epPos, enemyP)
                            _keyPos = epPos
                if self.getMazeDistance(_keyPos, cur_pos) > _min:  # calculate probability
                    delta = self.getMazeDistance(_keyPos, cur_pos) - _min
                    return False, len(exitPath), delta
            return True, len(exitPath), delta

        cur_pos = gameState.getAgentState(self.index).getPosition()
        # mid_distances = [self.getMazeDistance(cur_pos, mid_point) for mid_point in self.mid_points]
        # nearest_mid_point = self.GetNearestObject(self.mid_points, mid_distances)
        nearest_capsule = self.GetNearestCapsule(gameState)
        nearest_food = self.GetNearestFood(gameState)
        nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
        nearby_pacmans = self.GetNearbyOpponentPacmans(gameState)
        capsules = self.getCapsules(self.getPreviousObservation() if self.getPreviousObservation() else gameState)
        carry_points = self.getCurrentObservation().getAgentState(self.index).numCarrying

        cur_food = {}
        for f in [food for food in self.getFood(gameState).asList()]:
            cur_food[f] = self.getMazeDistance(f, cur_pos)

        self.update_lane_food()

        self.displayDistributionsOverPositions(self.updateDistribution())
        show_legalpos()

        _canSurvive, exitPathLen, delta_step = canSurvive()

        # print(self.isInvade(self.opponents_index[0]))
        def isLaneFood(f):
            if f in self.lane_food.keys() and self.lane_food[f] != 0:
                return True
            else:
                return False

        def cost_lane_food(f, curP):
            return self.getMazeDistance(f, curP) + self.lane_food[f]

        def evalution():
            nearby_ghosts = self.GetNearbyOpponentGhosts(gameState)
            if self.getCurrentObservation().data.timeleft <= 5 + self.getMazeDistance(cur_pos, bestExit()):
                return 'escape', bestExit()
            # if _canSurvive and carry_points / (exitPathLen + 1) > 1 and self.getMazeDistance(cur_pos,
            #                                                                                  certainFood()) >= 2:
            #     return 'escape', bestExit()
            # if not _canSurvive and nearby_ghosts == [] and delta_step <= 2:
            #     return 'escape', bestExit()
            if nearby_ghosts:
                isInv, leftTime = self.invincible_state
                dists = []
                chase = []
                for g in nearby_ghosts:
                    dis = self.getMazeDistance(cur_pos, g.getPosition())
                    dists.append(dis)
                    # if isInv and g.scaredTimer > dis and self.getMazeDistance(nearest_food, cur_pos) > 3:
                    #     chase.append((g.getPosition(), dis))

                # if chase != []:
                #     chase.sort(key=lambda x: x[1])
                #     return 'chase', chase[0][0]
                if isInv:
                    for f, _ in sorted(cur_food.items(), key=lambda x: x[1]):
                        can_eat = True
                        for g in nearby_ghosts:
                            if isLaneFood(f):
                                if g.scaredTimer <= cost_lane_food(f, cur_pos):
                                    can_eat = False
                        if can_eat:
                            return "eat_more", f

                    return 'escape', bestExit()
                else:
                    for d in dists:
                        if d <= 2:
                            return 'escape', bestExit()
                    for f, _ in sorted(cur_food.items(), key=lambda x: x[1]):
                        can_eat = True
                        for g in nearby_ghosts:
                            if isLaneFood(f):
                                for lane in self.lane.keys():
                                    if f in self.lane[lane]:
                                        entrance = self.lane_end_start[lane]
                                if self.getMazeDistance(g.getPosition(), entrance) < cost_lane_food(f, cur_pos)+1:
                                    can_eat = False
                        if can_eat:
                            return "eat_more", f
                    return 'escape', bestExit()

            # if _canSurvive and carry_points / (exitPathLen + 1) > 1 and self.getMazeDistance(cur_pos,
            #                                                                                  certainFood()) >= 2:
            #     return 'escape', bestExit()
            if not _canSurvive and nearby_ghosts == [] and delta_step <= 2:
                return 'escape', bestExit()
            # if min(eval_dist()) >= 8:
            #     return 'eat_more', nearest_food
            # if 5 == min(eval_dist()):
            #     return 'sneak', sneakPosition()

            return 'eat_more', certainFood()

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

        def sneakPosition():
            for op in [self.pre_op_idx, self.next_op_idx]:
                enemyPos = self.getCurrentObservation().getAgentState(op).getPosition()
                if enemyPos is not None:
                    for next_step, _ in self.GetSuccessors(cur_pos):
                        if util.manhattanDistance(next_step, enemyPos) > util.manhattanDistance(cur_pos, enemyPos):
                            return next_step
            return certainFood()

        def certainFood():
            x = lambda f: self.getMazeDistance(f, cur_pos)
            food = accessibleFood()
            dists = []
            if food == []:
                return cur_pos
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
                    for pos in self.distributions[op].keys():
                        if self.distributions[op][pos] != 0 and self.getMazeDistance(pos, f) < 5 * \
                                self.distributions[op][pos]:
                            isRisky = True
                if not isRisky:
                    accessible_food.append(f)
            return accessible_food

        def update_Invincible():
            isInvincible, leftTime = self.invincible_state
            if cur_pos in capsules:
                self.invincible_state = (True, SCARED_TIME)
            elif isInvincible:
                if leftTime - 1 > 0:
                    self.invincible_state = (True, leftTime - 1)
                else:
                    self.invincible_state = (False, 0)

        def pickAGate():
            gate = self.updateGate()
            candidate = {}
            self.weight_gate.incrementAll(self.weight_gate.keys(), 1)
            for pos in gate.keys():
                flag = True
                for op in self.opponents_index:
                    for p in [k for k in self.distributions[op].keys() if self.distributions[op][k] != 0]:
                        if self.getMazeDistance(pos, p) <= 5 * self.distributions[op][p]:
                            flag = False
                            self.weight_gate[pos] -= 5
                            break
                if flag:
                    candidate[pos] = gate[pos]
            # print(candidate)
            # print(self.weight_gate)
            # self.weight_gate.normalize()
            for p in candidate.keys():
                candidate[p] *= self.weight_gate[p]
            c_gate = sorted(candidate.items(), key=lambda x: x[1], reverse=True)[0][0]
            print(c_gate)
            return c_gate if c_gate is not None else cur_pos

        def pickAGate2():
            gate = {}
            self.weight_gate.incrementAll(self.weight_gate.keys(), 1)
            for pos in self.entrance:
                for op in self.opponents_index:
                    f = True
                    opp = self.getCurrentObservation().getAgentState(op).getPosition()
                    if opp is not None and self.getMazeDistance(pos, opp) <= 6:
                        f = False
                        self.weight_gate[pos] -= 5
                if f:
                    gate[pos] = self.getMazeDistance(pos, cur_pos)
            _list = sorted(gate.items(), key=lambda x: x[1])
            if _list is not None:
                print(_list[0][0])
                return _list[0][0]
            else:
                return cur_pos

        picked_gate = pickAGate()
        if self.mTerritory[cur_pos]:
            if cur_pos != picked_gate:
                return self.waStarSearch(picked_gate, self.noCrossingHeuristic)
            else:
                return self.waStarSearch(certainFood(), self.DetectOpponentGhostsHeuristic)
        strategy, goal = evalution()
        update_Invincible()
        # print(self.invincible_state)

        if strategy == 'eat_more':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'escape':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'sneak':
            return self.waStarSearch(goal, self.DetectOpponentGhostsHeuristic)
        if strategy == 'chase':
            return self.waStarSearch(goal, self.manhattanHeuristic)

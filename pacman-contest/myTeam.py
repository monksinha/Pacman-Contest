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

from heapq import heappop, heappush
from operator import itemgetter

from captureAgents import CaptureAgent
from distanceCalculator import manhattanDistance
from game import Actions, Directions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first='AStarAgent', second='AStarAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex, isRed, False), eval(second)(secondIndex, isRed, True)]

##########
# Agents #
##########

class AStarAgent(CaptureAgent, object):
    instances = [None, None]

    directions = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    nearby = {(2, 0), (0, 2), (-2, 0), (0, -2), (1, 1), (-1, -1), (1, -1), (-1, 1)}

    def __init__(self, index, red, defense, timeForComputing=.1):
        CaptureAgent.__init__(self, index, timeForComputing)
        object.__init__(self)

        self.red = red
        self.defense = defense
        self.bound = None
        self.actions = None
        self.escapes = None
        self.teammates = None
        self.prev_pos = None
        self.follow_path = None
        self.mask_food = None
        self.oppo_food = None
        self._walls = None
        self.carry_food = 0

        # record instance
        self.instances[index // 2] = self

    def registerInitialState(self, gameState):
        """
        Initialise the agent and compute an initial route
        """
        CaptureAgent.registerInitialState(self, gameState)

        data = gameState.data
        layout = data.layout
        height = layout.height
        width = layout.width
        half = width // 2
        red = self.red

        # record the bound in our side
        bound = half - 1 if red else half
        walls = layout.walls.data
        self.bound = set(
            (bound, y) for y in range(height) if not walls[bound][y]
        )

        # assume defensive agent will never reach the other side
        for x in (range(half, width) if self.red else range(half)):
            for y in range(height):
                walls[x][y] = True
        self._walls = walls

        # get an instance of the teammate
        self.teammates = self.instances[(self.index // 2 + 1) % 2]

    def chooseAction(self, gameState):
        """
        Choose an action based on the current status of the agent
        """
        return self.DefenseAction(gameState) if self.defense else self.AttackAction(gameState)

    def GetNextFood(self, gameState):
        self.escapes = None

        agentStates = gameState.data.agentStates
        red = self.red

        # determine the positions of the opponents
        poss = [agentStates[i] for i in (gameState.blueTeam if red else gameState.redTeam)]
        # only agent which is a ghost and not scared will mask the food
        poss = [s.configuration.pos for s in poss if not s.isPacman and s.scaredTimer == 0]

        distancer = self.distancer
        data = gameState.data
        layout = data.layout
        width = layout.width
        half = width // 2
        height = layout.height
        food = data.food.data

        # mark food that can be reached by an opponent in three steps
        mark_food = set()
        leftFood = set()
        for x in (range(half, width) if red else range(half)):
            for y in range(height):
                if food[x][y]:
                    pos = x, y
                    if any(distancer.getDistance(pos, p) < 6 for p in poss):
                        mark_food.add(pos)
                    else:
                        leftFood.add(pos)

        self.mask_food = mark_food
        self.oppo_food = leftFood

        index = self.index

        agent = agentStates[index]
        pos = tuple(map(int, agent.configuration.pos))

        # assign the closest food for trial
        if not leftFood:
            # start to escape asap if no foods left
            if not mark_food:
                return self.GetNextEscape(gameState)
            # TODO: coordinate with defensive agent
            mfs = min((
                (f, distancer.getDistance(pos, f))
                for f in mark_food
            ), key=itemgetter(1))[0]
            leftFood.add(mfs)
            mark_food.remove(mfs)

        # determine if the current path needs to be recomputed
        actions = self.actions
        recompute = not actions or mark_food != self.mask_food
        walls = data.layout.walls.data
        for i in (gameState.blueTeam if red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman and agentState.scaredTimer == 0:
                # pretend there are walls around the opponent agents if they are
                # not scared ghost
                x, y = pos = tuple(map(int, agentState.configuration.pos))
                if not actions or pos in actions:
                    recompute = True
                nx = x - 1
                if red and nx >= half or not red and nx < half:
                    walls[nx][y] = True
                nx = x + 1
                if red and nx >= half or not red and nx < half:
                    walls[nx][y] = True
                walls[x][y + 1] = walls[x][y - 1] = walls[x][y] = True

        x, y = pos = tuple(map(int, agent.configuration.pos))
        if not recompute:
            nx, ny = actions.pop()
            return Actions.vectorToDirection((nx - x, ny - y))

        self.actions = None

        # A*
        path = []
        h = min(distancer.getDistance(pos, f) for f in leftFood)
        q = [(h, h, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)
            if pos in leftFood:
                break
            visited.add(pos)
            x, y = pos
            for dx, dy in self.directions:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = min(distancer.getDistance(npos, f) for f in leftFood)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))
        if not path:
            return Directions.STOP

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()

        if path:
            self.actions = path
        return Actions.vectorToDirection((nx - x, ny - y))

    def observationFunction(self, gameState):
        return gameState

    def GetNextEscape(self, gameState):
        self.actions = None

        red = self.red
        index = self.index
        data = gameState.data
        half = data.layout.width // 2
        agentStates = data.agentStates
        bounds = self.bound
        distancer = self.distancer

        # determine if the escape path need to be recomputed
        escapes = self.escapes
        recompute = not escapes
        walls = data.layout.walls.data
        for i in (gameState.blueTeam if red else gameState.redTeam):
            agentState = agentStates[i]
            if not agentState.isPacman and agentState.scaredTimer == 0:
                # pretend there are walls around the opponent agents if they are
                # not scared ghost
                x, y = agentState.configuration.pos
                pos = x, y = int(x), int(y)
                if not escapes or pos in escapes:
                    recompute = True
                if red and x - 1 >= half or not red and x - 1 < half:
                    walls[x - 1][y] = True
                if red and x + 1 >= half or not red and x + 1 < half:
                    walls[x + 1][y] = True
                walls[x][y + 1] = walls[x][y - 1] = walls[x][y] = True

        agent = agentStates[index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        if not recompute:
            nx, ny = escapes.pop()
            return Actions.vectorToDirection((nx - x, ny - y))

        # reset path
        self.escapes = None

        # A* to escape
        path = []
        # escape to the nearest bound
        h = min(distancer.getDistance(pos, b) for b in bounds)
        q = [(h, h, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos in bounds:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self.directions:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = min(distancer.getDistance(npos, b) for b in bounds)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        if not path:
            # TODO: change behaviour
            return Directions.STOP

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()

        if path:
            self.escapes = path
        return Actions.vectorToDirection((nx - x, ny - y))

    def AttackAction(self, gameState):
        index = self.index
        red = self.red
        agentStates = gameState.data.agentStates
        agentState = agentStates[index]

        distancer = self.distancer
        pos = agentState.configuration.pos

        prev_pos = self.prev_pos
        if prev_pos is not None:
            if manhattanDistance(pos, prev_pos) > 1:
                self.escapes = None
                self.actions = None
                self.follow_path = None
                # TODO: notify the defensive agent
                # self.teammates._notifyReborn()
        self.prev_pos = pos

        # TODO: determine if has capsule beside and perform corresponding action

        # if escaping, finish escaping
        if self.escapes:
            return self.GetNextEscape(gameState)

        states = [
            agentStates[i]
            for i in (gameState.blueTeam if red else gameState.redTeam)
        ]
        # if determine to be in danger and currently carrying food, escape
        if any(
                not s.isPacman and s.scaredTimer == 0 and distancer.getDistance(
                    s.configuration.pos, pos
                ) < 4
                for s in states
        ):
            self.actions = None
            nc = self.carry_food = agentState.numCarrying
            if nc > 0:
                return self.GetNextEscape(gameState)

        # find the closest food to eat, not necessary to be a TSP, this is just
        # a greedy strategy to eat the current closest food
        return self.GetNextFood(gameState)

    def Chase(self, gameState, target):
        target = tuple(map(int, target))

        agent = gameState.data.agentStates[self.index]
        x, y = pos = tuple(map(int, agent.configuration.pos))
        distancer = self.distancer

        dist = distancer.getDistance(pos, target)
        cp = self.follow_path
        # determine if needs to recompute
        if cp is not None:
            movement = manhattanDistance(cp[0], target)
            # insert the target into the last
            if movement == 1:
                cp = [target] + cp

            # only follow the original route if the target didn't change
            if movement <= 1:
                if len(cp) <= dist:
                    nx, ny = cp.pop()
                    self.follow_path = cp if cp else None
                    return Actions.vectorToDirection((nx - x, ny - y))

        # reset path
        self.follow_path = None

        walls = self._walls
        # A* to chase
        path = []
        q = [(dist, dist, 0, pos, path)]
        visited = set()
        while q:
            _, _, g, pos, path = heappop(q)

            if pos == target:
                break

            visited.add(pos)

            x, y = pos
            for dx, dy in self.directions:
                npos = nx, ny = x + dx, y + dy
                if not walls[nx][ny] and npos not in visited:
                    h = distancer.getDistance(pos, target)
                    ng = g + 1
                    heappush(q, (ng + h, h, ng, npos, path + [npos]))

        if not path:
            # TODO: change behaviour
            return Directions.STOP

        path.reverse()
        x, y = agent.configuration.pos
        nx, ny = path.pop()

        self.follow_path = path if path else None
        return Actions.vectorToDirection((nx - x, ny - y))

    def DefenseAction(self, gameState):
        index = self.index
        red = self.red
        data = gameState.data
        agentStates = data.agentStates
        distancer = self.distancer
        bounds = self.bound
        agent = agentStates[index]
        pos = agent.configuration.pos
        scare = agent.scaredTimer > 0
        walls = data.layout.walls.data

        prev_pos = self.prev_pos
        if prev_pos is not None:
            if manhattanDistance(pos, prev_pos) > 1:
                self.escapes = None
                self.actions = None
                self.follow_path = None
                # TODO: notify the defensive agent
                # self.teammates._notifyReborn()
        self.prev_pos = pos

        # first select the target with the highest carrying food
        target = None
        rs = []
        pnc = 0
        for i in (gameState.blueTeam if red else gameState.redTeam):
            agentState = agentStates[i]
            nc = agentState.numCarrying
            npos = agentState.configuration.pos
            if nc > pnc:
                pnc = nc
                target = agentState.configuration.pos
            rs.append((min(((b, (distancer.getDistance(npos, b), distancer.getDistance(pos, b))) for b in bounds),
                           key=itemgetter(1)), npos, agentState.isPacman))
        layout = data.layout
        height, width = layout.height, layout.width
        if target is not None:
            if scare:
                tx, ty = target
                sur = [(int(tx + cx), int(ty + cy)) for cx, cy in self.nearby]
                sur = [(x, y) for x, y in sur if 0 <= x < width and 0 <= y < height and not walls[x][y]]
                sel = \
                    min(((s, min(distancer.getDistance(s, b) for b in bounds), distancer.getDistance(pos, s)) for s in
                         sur),
                        key=itemgetter(1, 2))[0]
                return self.Chase(gameState, sel)
            return self.Chase(gameState, target)

        # if no agent carries food, select the closest one which is currently a
        # Pacman
        mb = None
        mbd = (float("inf"), float("inf"))
        md = float("inf")
        for (b, bd), npos, pac in rs:
            dist = distancer.getDistance(npos, pos)
            if pac:
                if dist < md:
                    target = npos
                    md = dist
            else:
                if bd < mbd:
                    mb, mbd = b, bd

        if target is not None:
            if scare:
                tx, ty = target
                sur = [(int(tx + cx), int(ty + cy)) for cx, cy in self.nearby]
                sur = [(x, y) for x, y in sur if 0 <= x < width and 0 <= y < height and not walls[x][y]]
                sel = \
                    min(((s, min(distancer.getDistance(s, b) for b in bounds), distancer.getDistance(pos, s)) for s in
                         sur),
                        key=itemgetter(1, 2))[0]
                return self.Chase(gameState, sel)
            return self.Chase(gameState, target)

        # if both are still in their sides, just try to reach the closest bound
        # they could reach
        if scare:
            tx, ty = mb
            sur = [(int(tx + cx), int(ty + cy)) for cx, cy in self.nearby]
            sur = [(x, y) for x, y in sur if 0 <= x < width and 0 <= y < height and not walls[x][y]]
            sel = \
                min(((s, min(distancer.getDistance(s, b) for b in bounds), distancer.getDistance(pos, s)) for s in sur),
                    key=itemgetter(1, 2))[0]
            return self.Chase(gameState, sel)
        return self.Chase(gameState, mb)

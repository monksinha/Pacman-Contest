# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    fringe = util.Stack()
    explored = set()
    start_node = (problem.getStartState(), [], 0)
    fringe.push(start_node)
    while not fringe.isEmpty():
        curr_state, actions, acc_cost = fringe.pop()
        if problem.isGoalState(curr_state):
            return actions
        if curr_state not in explored:
            explored.add(curr_state)
            for successor in problem.getSuccessors(curr_state):
                next_state, action, cost = successor
                fringe.push((next_state, actions+[action], acc_cost+cost))
    return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    fringe = util.Queue()
    explored = set()
    explored.add(problem.getStartState())
    start_node = (problem.getStartState(), [], 0)
    fringe.push(start_node)
    while not fringe.isEmpty():
        curr_state, actions, acc_cost = fringe.pop()
        if problem.isGoalState(curr_state):
            return actions
        for successor in problem.getSuccessors(curr_state):
            next_state, action, cost = successor
            if next_state not in explored:
                explored.add(next_state)
                fringe.push((next_state, actions+[action], acc_cost+cost))
    return None


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE IF YOU WANT TO PRACTICE ***"
    fringe = util.PriorityQueue()
    explored = set()
    best_g = dict()
    initial_state = problem.getStartState()
    start_node = (initial_state, [], 0)
    priority = heuristic(initial_state, problem)
    fringe.push(start_node, priority)
    while not fringe.isEmpty():
        curr_state, actions, acc_cost = fringe.pop()
        if curr_state not in explored or acc_cost < best_g[curr_state]:
            explored.add(curr_state)
            best_g[curr_state] = acc_cost
            if problem.isGoalState(curr_state):
                return actions
            for successor in problem.getSuccessors(curr_state):
                next_state, action, cost = successor
                new_cost = acc_cost + cost
                new_actions = actions + [action]
                new_priority = problem.getCostOfActions(new_actions) + heuristic(next_state, problem)
                fringe.update((next_state, new_actions, new_cost), new_priority)
    return None


"""
Iterative Approach
"""
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE FOR TASK 1 ***"
    depth_limit = 0
    fringe = util.Stack()
    start_node = (problem.getStartState(), [], 0)
    while True:
        fringe.push(start_node)
        explored = set()
        path = depthLimitedSearch(problem, depth_limit, explored, fringe)
        if path is not None:
            return path
        print(len(fringe.list))
        depth_limit += 1


def depthLimitedSearch(problem, depth, explored, fringe):
    best_cost = dict()
    while not fringe.isEmpty():
        curr_state, actions, acc_cost = fringe.pop()
        if problem.isGoalState(curr_state):
            return actions
        if curr_state not in explored or acc_cost < best_cost[curr_state]:
            explored.add(curr_state)
            best_cost[curr_state] = acc_cost
            if len(actions) >= depth:
                continue
            for successor in problem.getSuccessors(curr_state):
                next_state, action, cost = successor
                fringe.push((next_state, actions+[action], acc_cost+cost))
    return None


"""
Recursive Approach
"""
# def iterativeDeepeningSearch(problem):
#     """Search the deepest node in an iterative manner."""
#     "*** YOUR CODE HERE FOR TASK 1 ***"
#     depth_limit = 0
#     while True:
#         (isFound, path) = depthLimitedSearch(problem, depth_limit, problem.getStartState())
#         if isFound:
#             return path
#         depth_limit += 1
#
#
# def depthLimitedSearch(problem, depth, state):
#     actions = []
#     explored = set()
#     explored.add(state)
#     def dls(problem, depth, state, actions, explored):
#         isFound = False
#         if problem.isGoalState(state):
#             isFound = True
#             return (isFound, actions)
#         elif depth == 0:
#             return (isFound, [])
#         else:
#             for successor in problem.getSuccessors(state):
#                 next_state, action, cost = successor
#                 if next_state not in explored:
#                     new_explored = explored.copy()
#                     new_explored.add(next_state)
#                     (result, path) = dls(problem, depth-1, next_state, actions + [action], new_explored)
#                     if result:
#                         isFound = True
#                         return (isFound, path)
#             return (isFound, [])
#     return dls(problem, depth, state, actions, explored)


def waStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has has the weighted (x 2) lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE FOR TASK 2 ***"
    fringe = util.PriorityQueue()
    explored = set()
    best_g = dict()
    initial_state = problem.getStartState()
    start_node = (initial_state, [], 0)
    priority = 2*heuristic(initial_state, problem)
    fringe.push(start_node, priority)
    while not fringe.isEmpty():
        curr_state, actions, acc_cost = fringe.pop()
        if curr_state not in explored or acc_cost < best_g[curr_state]:
            explored.add(curr_state)
            best_g[curr_state] = acc_cost
            if problem.isGoalState(curr_state):
                return actions
            for successor in problem.getSuccessors(curr_state):
                next_state, action, cost = successor
                # if heuristic(curr_state, problem)-heuristic(next_state, problem) > cost:
                #     print("not consistent")
                new_cost = acc_cost + cost
                new_actions = actions + [action]
                new_priority = problem.getCostOfActions(new_actions) + 2*heuristic(next_state, problem)
                fringe.update((next_state, new_actions, new_cost), new_priority)
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
wastar = waStarSearch

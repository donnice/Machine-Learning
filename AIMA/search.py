from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, Stack, FIFOQueue, PriorityQueue, name,
    distance
)

from collections import defaultdict
import math
import random
import sys
import bisect

class Problem(object):
    """Abstract class. Should be subclassed."""
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal
    
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

class Node:
    """
    A node in search tree
    """
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree node."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.path = parent.depth + 1
    
    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                    for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next = problem.result(self.state, action)
        return Node(next, self, action, 
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))
    def solution(self):
        """Return the sequence of action to go to root"""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes in the path"""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


def hill_climbing(problem):
    """From the initial node, keep choosing the neighbor with highest value"""
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, 
                                     key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state

def boggle_hill_climbing(board=None, ntimes=100, verbose=True):
    """Solve inverse Boggle by hill-climbing: find a high-scoring board by
    starting with a random one and changing it."""
    pass

def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda x: (k * math.exp(-lam * x) if x < limit else 0)

def selection_chances(fitness_fn, population):
    fitnesses = map(fitness_fn, population)
    return weighted_sampler(population, fitnesses)

def reproduce(x, y):
    n = len(x)
    # Return a randomly selected element from range(start, stop, step)
    c = random.randrange(1, n)
    return x[:c] + y[c:]

def mutate(x, gene_pool):
    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c+1:]

def simulated_annealing(problem, schedule=exp_schedule()):
    current = Node(problem.initial)
    # The largest positive integer supported by the platform’s Py_ssize_t type
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next = random.choice(neighbors)
        delta_e = problem.value(next.state) - problem.value(current.state)
        # return p > random.uniform(0.0, 1.0)
        if delta_e > 0 or probability(math.exp(delta_e / T)):
            current = next

def genetic_search(problem, fitness_fn, ngen=1000, pmut=0.1, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""

    s = problem.initial_state
    # s,a: state, action
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)
    return genetic_algorithm(states[:n], problem.value, ngen, pmut)

def genetic_algorithm(population, fitness_fn, ngen=1000, pmut=0.1, gene_pool=[0, 1], f_thres=None):
    """[Figure 4.8]"""
    for i in range(ngen):
        new_population = []
        random_selection = selection_chances(fitness_fn, population)
        for j in range(len(population)):
            x = random_selection()
            y = random_selection()
            child = reproduce(x, y)
            if random.uniform(0, 1) < pmut:
                child = mutate(child, gene_pool)
            new_population.append(child)
        population = new_population

        if t_thres:
            fittest_individual = argmax(population, key=fitness_fn)
            if fitness_fn(fittest_individual) >= f_thres:
                return fittest_individual
    # With a single iterable argument, return its largest item. 
    # With two or more arguments, return the largest argument.
    return argmax(population, key=fitness_fn)

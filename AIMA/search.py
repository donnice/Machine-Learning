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

class Node:
    """
    A node in search tree
    """
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree node"""
        self.state = state
        self.parent = parent


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

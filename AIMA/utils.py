"""Provides some utilities widely used by other modules"""

import bisect
import collections
import collections.abc
import operator
import os.path
import random
import math
import functools

def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)

def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)
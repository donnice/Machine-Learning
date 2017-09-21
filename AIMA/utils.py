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

def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    
    # bisect.bisect: Return the index where to insert item x in list a, assuming a is sorted.
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]
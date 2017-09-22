import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from sklearn.metrics import mutual_info_score

np.set_printoptions(precision=4)

class Mimic(object):
    """
    domain: List of tuples containing the min and max 
            value for each parameter to be optimized
    fitness_function: callable that will take a single 
            instance of your optimization parameters 
            and return a scalar fitness score
    samples: num of samples to generate from the 
            distribution each iteration
    percentile: Percentile of distribution keep after 
            each iteration
    """

    def __init__(self, domain, fitness_function, samples=1000, percentile=0.90):
        self.domain = domain
        self.samples = samples
        initial_samples = np.array(self._generate_initial_samples())
        self.sample_set = SampleSet(initial_samples, fitness_function)
        self.fitness_function = fitness_function
        self.percentile = percentile
    
    def fit(self):
        """
        Run this to perform one iteration of the Mimic algorithm
        return: A list containing the top percentile of data points
        """
        samples = self.sample_set.get_percentile(self.percentile)

    def _generate_initial_samples(self):
        return [self._generate_initial_sample() for i in xrange(self.samples)]

    def _generate_initial_sample(self):
        return [random.randint(self.domain[i][0], self.domain[i][1]) 
                for i in xrange(len(self.domain))]

class SampleSet(object):
    def __init__(self, samples, fitness_function, maximize=True):
        self.samples = samples
        self.fitness_function = fitness_function
        self.maximize = maximize
    
    def calculate_fitness(self):
        sorted_samples = sorted(
            self.samples,
            key=self.fitness_function,
            reverse=self.maximize
        )
        return np.array(sorted_samples)
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
        self.distribution = Distribution(samples)
        self.sample_set = SampleSet(
            self.distribution.generate_samples(self.samples),
            self.fitness_function
        )
        return self.sample_set.get_percentile(self.percentile)

    def _generate_initial_samples(self):
        # xrange is slightly faster than range
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

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index]

class Distribution(object):
    def __init__(self, samples):
        self.samples = samples
        self.complete_graph = self._generate_mutual_information_graph()
        self.spanning_graph = self._generate_spanning_graph()
        self._generate_bayes_net()
    
    def generate_samples(self, number_to_generate):
        root = 0
        sample_len = len(self.bayes_net.node)
        samples = np.zeros((number_to_generate, sample_len))
        values = self.bayes_net.node[root]["probabilities"].keys()
        probabilities = self.bayes_net.node[root]["probabilites"].values()
        # Random variates of given type.
        # name: The name of the instance.
        # values: (xk, pk) where xk are integers with non-zero probabilities 
        # pk with sum(pk) = 1
        dist = stats.rv_discrete(name="dist", values=(values, probabilities))
        # Random number generation
        samples[:, 0] = dist.rvs(size=number_to_generate)
        # Iterate over edges in a breadth-first-search starting at source
        for parent, current in nx.bfs_edges(self.bayes_net, root):
            for i in xrange(number_to_generate):
                parent_val = samples[i, parent]
                current_node = self.bayes_net.node[current]
                cond_dist = current_node["probabilities"][int(parent_val)]
                values = cond_dist.keys()
                probabilities = cond_dist.values()
                dist = stats.rv_discrete(
                    name="dist",
                    values=(values, probabilities)
                )
                samples[i, current] = dist.rvs()
        return samples


    def _generate_bayes_net(self):
        # 1. Start at any node (0)
        # 2. At each node figure out the conditional prob
        # 3. Add it to the new graph
        # 4. Find unprocessed adjacent nodes
        # 5. If any go to 2
        #      Else return the bayes net

        # Will it be possible that zero is not root? If
        # so, we need to pick one
        root = 0

        samples = np.asarry(self.samples)
        self.bayes_net = nx.bfs_tree(self.spanning_graph, root)
        
        for parent, child in self.bayes_net.edges():
            parent_array = samples[:, parent]

            # if node is not root, get probability of 
            # each gene appearing in parent
            # Return an iterator over predecessor nodes of n
            if not self.bayes_net.predecessors(parent):
                freqs = np.histogram(parent_array, len(np.unique(parent_array)))[0]
                # zip([a,b,c],[1,2,3]) -> [(a, 1), (...)]
                parent_probs = dict(zip(np.unique(sub_child), freqs/(sum(freqs)*1.0)))

                self.bayes_net.node[parent]["probabilities"] = {x:0 for x in range(len(self.samples))}
                self.bayes_net.node[parent]["probabilities"].update(parent_nodes)

            child_array = samples[:, child]

            unique_parent = np.unique(parent_array)
            for parent_val in unique_parents:
                parent_inds = np.argwhere(parent_array == parent_val)
                sub_child = child_array[parent_inds]

                # Compute the histogram of a set of data.
                freqs = np.histogram(sub_child, len(np.unique(sub_child)))[0]
                child_probs = dict(zip(np.unique(sub_child), freqs/(sum(freqs)*1.0)))

                self.bayes_net.node[child][parent_val] = {x:0 for x in range(len(self.samples))}
                self.bayes_net.node[child][parent_val].update(parent_nodes)

            self.bayes_net.node[child] = dict(probabilities=self.bayes_net.node[child])

        # PRIM_MST Compute a minimum spanning with Prims's algorithm.
        def _generate_spanning_graph(self):
            return nx.prim_mst(self.complete_graph)

        def _generate_mutual_information_graph(self):
            samples = np.asarray(self.samples)
            complete_graph = nx.complete_graph(samples.shape[1])

            for edge in complete_graph.edges():
                mutual_info = mutual_info_score(
                    samples[:, edge[0]],
                    samples[:, edge[1]]
                )

                complete_graph.edge[edge[0]][edge[1]]['weight'] = -mutual_info

            return complete_graph

if __name__ == "__main__":
    samples = [
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
    ]

    distribution = Distribution(samples)

    distribution._generate_bayes_net()

    for node_ind in distribution.bayes_net.nodes():
        print distribution.bayes_net.node[node_ind]
    
    pos = nx.spring_layout(distribution.spanning_graph)

    edge_labels = dict(
        [((u, v,), d['weight'])
        for u, v, d in distribution.spanning_graph.edges(data=True)]
    )

    nx.draw_networkx(distribution.spanning_graph, pos)
    nx.draw_networkx_edge_labels(
        distribution.spanning_graph,
        pos,
        edge_label=edge_labels
    )

    plt.show()
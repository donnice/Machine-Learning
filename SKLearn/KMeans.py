import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T # Stack arrays in sequence horizontally (column wise).
# X = np.vstack((cluster1, cluster2)).T # Stack arrays in sequence vertically (row wise).

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# plt.plot(K, meandistortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Average distortion')
# plt.title('Selecting k with Elbow Method')
# plt.show()

# More experiments
x, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)
plt.scatter(x[:, 0], x[:, 1], c='green', marker='o', s=50)
plt.grid()
plt.show()

# Algorithm
'''
1. Randomly pick k centroids from the sample points as initial cluster
2. Assign each sample to the nearest centroid
3. Move the centroids to the center of samples that were assigned to it
4. Repeat 2 and 3 until cluster assignment do not change or max iter is reached
'''
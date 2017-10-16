import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import mahotas as mh

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T # Stack arrays in sequence horizontally (column wise).
X = np.vstack((cluster1, cluster2)).T # Stack arrays in sequence vertically (row wise).

# More experiments
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)
plt.scatter(x[:, 0], x[:, 1], c='green', marker='o', s=50)
plt.grid()
plt.show()

# estimate the optimal number of clusters k for a given task
# if k increases, the distortion will decrease. 
# This is because the samples will be closer to the centroids they are assigned to.

K = range(1, 11)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot(K, meandistortions, 'o-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Selecting k with Elbow Method')
plt.show()


# K-means
'''
1. Randomly pick k centroids from the sample points as initial cluster
2. Assign each sample to the nearest centroid
3. Move the centroids to the center of samples that were assigned to it
4. Repeat 2 and 3 until cluster assignment do not change or max iter is reached
'''
# plt.subplot(3, 2, 1)

x1 = np.array([1, 2, 4, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
# >>> x = [1, 2, 3]
# >>> y = [4, 5, 6]
# >>> zipped = zip(x, y)
# >>> zipped
# [(1, 4), (2, 5), (3, 6)]
X = np.array(zip(x1, x2)).reshape(len(x1), 2)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title('Instances')
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'c']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
tests = [2, 3, 4, 5, 8]
subplot_counter = 1
for t in tests:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
       plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    # The Silhouette Coefficient is calculated using the mean 
    # intra-cluster distance (a) and the mean nearest-cluster distance 
    # (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b)
    # It measures the compactness and separation of the clusters
    # The higher, the "more correct" it's seperated
    plt.title('K = %s, silhouette coefficient  = %.03f '% ( \
                t, metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean')))
plt.show()

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=0)
# fit_predict: Compute cluster centers and predict cluster index for each sample
# fit: Compute k-means clustering.
y_km = km.fit_predict(X)

'''
K-means++
1. Initialize an empty set M to store the k centroids being selected
2. Randomly choose the first \mu^{(j)} from the input samples and assign it to M
   ## Here \mu^{(j)} is the centroid of the cluster
3. For each sample x^{(i)} that is not in M, find the minimum squared distance
   d(x^{(i)}, M)^{2} to any of the centroids in M
4. To randomly select the next centroid \mu^{(p)}, use a weightedp probability
   distribution equal to \frac{d(\mu^{(p)}, M)^{2}}{\sum d(x^{(i)}, M)^{2}}
5. Repeat 2 and 3 until k centroids are chosen
6. Proceed with classic k-means algorithm 
'''
plt.scatter(X[y_km==0, 0],
            X[y_km==0, 1],
            s=50, # size in points
            c='lightgreen',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km==1, 0],
            X[y_km==1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(X[y_km==2, 0],
            X[y_km==2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()
plt.show()
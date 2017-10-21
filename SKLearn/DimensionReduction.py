import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def implementPCA():
    data = load_iris()
    y = data.target
    X = data.data
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X)

    red_X, red_Y = [], []
    blue_X, blue_Y = [], []
    green_X, green_Y = [], []
    for i in range(len(reduced_X)):
        if y[i] == 0:
            red_X.append(reduced_X[i][0])
            red_Y.append(reduced_X[i][1])
        elif y[i] == 1:
            blue_X.append(reduced_X[i][0])
            blue_Y.append(reduced_X[i][1])
        else:
            green_X.append(reduced_X[i][0])
            green_Y.append(reduced_X[i][1])
    plt.scatter(red_X, red_Y, c='r', marker='x')
    plt.scatter(blue_X, blue_Y, c='b', marker='D')
    plt.scatter(green_X, green_Y, c='g', marker='.')
    plt.show()

implementPCA()
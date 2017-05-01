import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cluster.PSC import PSC

def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[:, 4].values
    y[y == "Iris-setosa"] = 0
    y[y == "Iris-versicolor"] = 1
    y[y == "Iris-virginica"] = 2
    x = df.iloc[:, [0, 2]].values

    x = np.array(x)
    y = np.array(y)

    x[:, 0] = (x[:, 0] - x[:, 0].min()) / (x[:, 0].max() - x[:, 0].min())
    x[:, 1] = (x[:, 1] - x[:, 1].min()) / (x[:, 1].max() - x[:, 1].min())

    clf = PSC(n_clusters=3, swarm_size=10, n_iter=50, w=0.72, lb_w=0.4, w_damp=None, c1=1.49, c2=1.49)
    clf.fit(x)

    plt.subplot(121)
    colors = ['g', 'm', 'y']
    for idx in range(len(np.unique(y))):
        plt.scatter(x[:, 0][y == idx], x[:, 1][y == idx], c=colors[idx], s=10, edgecolors='k')

    for k in clf.centroids:
        plt.scatter(clf.centroids[k][0], clf.centroids[k][1], marker='x', s=100, c='b')

    plt.title('PSC')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')

    plt.subplot(122)
    plt.plot([(i + 1) for i in range(len(clf.pso.best_cost))], clf.pso.best_cost, c='b')
    plt.legend(["Intra Cluster Cost"])
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')

    plt.show()

if __name__ == '__main__':
    main()

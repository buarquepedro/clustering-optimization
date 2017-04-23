import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from cluster import KMeans

def main():
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	y = df.iloc[:, 4].values
	y[y == "Iris-setosa"] = 0
	y[y == "Iris-versicolor"] = 1
	y[y == "Iris-virginica"] = 2
	x = df.iloc[:, [0,2]].values

	x = np.array(x)
	y = np.array(y)

	x[:, 0] = (x[:, 0] - x[:, 0].min())/(x[:, 0].max() - x[:, 0].min())
	x[:, 1] = (x[:, 1] - x[:, 1].min())/(x[:, 1].max() - x[:, 1].min())

	clf = KMeans.KMeans(n_clusters=3)
	clf.fit(x)
	
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()

	x1_range = np.arange(x1_min, x1_max, 0.02)
	x2_range = np.arange(x2_min, x2_max, 0.02)
	xx, yy = np.meshgrid(x1_range, x2_range)
	x_ = np.array([xx.ravel(), yy.ravel()]).T

	z = clf.predict(x_)
	z = z.reshape(xx.shape)
	plt.contourf(xx, yy, z)

	colors = ['g', 'm', 'y']
	for idx in range(len(np.unique(y))):
		plt.scatter(x[:, 0][y == idx], x[:, 1][y == idx], c=colors[idx], s=10, edgecolors='k')

	for k in clf.centroids:
		plt.scatter(clf.centroids[k][0], clf.centroids[k][1], marker='x', s=100, c='b')
		print clf.centroids[k]

	plt.title('K-Means')
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.legend(loc='upper left')
	plt.show()

if __name__ == '__main__':
	main()
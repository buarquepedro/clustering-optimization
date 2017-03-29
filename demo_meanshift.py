import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D

def main():
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	y = df.iloc[:, 4].values
	y[y == "Iris-setosa"] = 0
	y[y == "Iris-versicolor"] = 1
	y[y == "Iris-virginica"] = 2
	x = df.iloc[:, [0,2,3]].values

	x = np.array(x)
	y = np.array(y)

	x[:, 0] = (x[:, 0] - x[:, 0].min())/(x[:, 0].max() - x[:, 0].min())
	x[:, 1] = (x[:, 1] - x[:, 1].min())/(x[:, 1].max() - x[:, 1].min())

	clf = MeanShift.MeanShift()
	clf.fit(x)
	
	colors = ['g', 'm', 'y']
	fig = plt.figure()
	ax = Axes3D(fig)

	for idx in range(len(np.unique(y))):
		ax.scatter(x[:, 0][y == idx], x[:, 1][y == idx], x[:, 2][y == idx], c=colors[idx], s=10, edgecolors='k')

	for i in clf.centroids:
		ax.scatter(clf.centroids[i][0], clf.centroids[i][1], clf.centroids[i][2], marker='*', c='r', s=100, edgecolors='k')

	plt.title('MeanShift')
	plt.show()

if __name__ == '__main__':
	main()
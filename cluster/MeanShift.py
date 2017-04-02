import math
import numpy as np

class MeanShift(object):
	
	def __init__(self, radius=None, n_iter=500, shuffle=True, tolerance=0.001):
		self.radius = radius
		self.n_iter = n_iter
		self.tolerance = tolerance
		self.shuffle = shuffle
		self.run = False

 	def fit(self, x):
 		self.run = True
 		if len(x.shape) < 1:
 			raise Exception("DataException: Dataset must contain more examples" + 
 							"than the required number of clusters!")

 		self.points = np.copy(x)
 		shifted_points = np.zeros(self.points.shape)

 		for idx in range(self.n_iter):
 			max_dist = 0
 			for idx, xi in self.points:
 				shifted_points[int(idx)] = self.shift(xi, self.points)
 				if (np.linalg.norm(xi - shifted_points[int(idx)]) > max_dist):
 					max_dist = np.linalg.norm(xi - shifted_points[int(idx)])
 				
			self.points = np.copy(shifted_points)
			shifted_points = np.zeros(self.points.shape)

 			if max_dist < self.tolerance:
				break

		self.centroids = {}
		cluster_centers = self.filter(self.points, self.thresold)
 		cluster_centers = [tuple(l) for l in cluster_centers]
 		unique_cluster_centers = sorted(list(set(cluster_centers)))

 		for i in range(len(unique_cluster_centers)):
 			self.centroids[i] = np.array(unique_cluster_centers[i])

 		print self.centroids
 		return self

	def predict(self, x):
		if self.run:
			if len(x.shape) > 1:
				class_ = []
				for c in self.centroids:
					class_.append(np.sum((x - self.centroids[c]) ** 2, axis=1))
	 			return np.argmin(np.array(class_).T, axis=1)
	 		else:
	 			dist = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
 				class_ = dist.index(min(dist))
 				return class_
		else:
			raise Exception("NonTrainedModelException: You must fit data first!")

	def filter(self, x, cond):
	    out = []
	    for xi in x:
	        if all(cond(xi,xii) for xii in out):
	            out.append(xi)
	    return out

	def thresold(self, xs, ys):
		return sum((x-y)*(x-y) for x,y in zip(xs,ys)) > 2.5e-05

	def shift(self, xi, points):
		c1 = 0
 		c2 = 0
 		for x in points:
 			dist = np.linalg.norm(xi - x)
 			weight = self.gaussian_kernel(dist, 0.1)
			c1 += (weight * xi)
			c2 += weight
		return c1/c2

	def gaussian_kernel(self, dist, bandwidth=1.067):
		# bandwidth=1.067 is a default thumb-ruled value for gaussian kernel
		return (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dist / bandwidth))**2)
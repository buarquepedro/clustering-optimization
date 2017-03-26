import math
import numpy as np

class MeanShift(object):
	
	def __init__(self, radius=None, n_iter=500, tolerance=0.0001):
		self.radius = radius
		self.n_iter = n_iter
		self.tolerance = tolerance
		self.run = False

 	def fit(self, x):
 		self.run = True
 		self.centroids = {}

 		if self.radius is None:
 			self.radius = np.linalg.norm(np.average(x, axis=0))

 		for i in range(len(x)):
 			self.centroids[i] = x[i]

 		for idx in range(self.n_iter):
 			cluster_centers = []
 			for i in self.centroids:
 				neighbours = self.get_neighbours(self.centroids[i], x, self.radius)

 				c1 = 0
 				c2 = 0
 				for xi in neighbours:
 					dist = np.linalg.norm(xi - self.centroids[i])
 					weight = self.gaussian_kernel(dist)
					c1 += (weight * xi)
					c2 += weight

				new_centroid = c1/c2
 				cluster_centers.append(tuple(new_centroid))

 			is_done = True
 			for i in self.centroids:
 				if np.linalg.norm(self.centroids[i] - cluster_centers[i]) > self.tolerance:
 					is_done = False
 					break

 			if is_done:
 				break

 			# Othe Approach for remove 'duplicates':
 			# result = {}
			# for row in cluster_centers:
			#     key = tuple([round(v, 2) for v in row]) 
			#     if key not in result:
			#         result[key] = row

			# self.centroids = {}
			# cluster_centers = result.values()

 			self.centroids = {}
 			cluster_centers = self.filter(cluster_centers, self.thresold)
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
		return sum((x-y)*(x-y) for x,y in zip(xs,ys)) > self.tolerance

	def get_neighbours(self, centroid, x, radius):
		neighbors = []
		for xi in x:
			if np.linalg.norm(xi - centroid) <= radius:
				neighbors.append(xi)
		return neighbors		

	def gaussian_kernel(self, dist, bandwidth=1.067):
		# bandwidth=1.067 is a default thumb-ruled value for gaussian kernel
		return (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((dist / bandwidth))**2)
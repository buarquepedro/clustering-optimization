import numpy as np 
 
class FCMeans(object):
	
	def __init__(self, n_clusters=3, n_iter=300, fuzzy_c=2, tolerance=0.001):
		self.n_clusters = n_clusters
		self.n_iter = n_iter
		self.fuzzy_c = fuzzy_c
		self.tolerance = tolerance
		self.run = False

 	def fit(self, x):
 		self.run = True
 		self.centroids = {}

 		if len(x.shape) < 1:
 			raise Exception("DataException: Dataset must contain more examples" + 
 							"than the required number of clusters!")
 	
 		for k in range(self.n_clusters):
 			self.centroids[k] = np.random.random(x.shape[1])

 		self.degree_of_membership = np.zeros((x.shape[0], self.n_clusters))
 		for idx_ in self.centroids:
			for idx, xi in enumerate(x):
				updated_degree_of_membership = 0.0
				norm = np.linalg.norm(xi - self.centroids[idx_])
				all_norms = [norm/np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
				all_norms = np.power(all_norms, 2/(self.fuzzy_c-1))
				updated_degree_of_membership = 1/sum(all_norms)
				self.degree_of_membership[idx][idx_] = updated_degree_of_membership

		for iteration in range(self.n_iter):
 			powers = np.power(self.degree_of_membership, self.fuzzy_c)
 			for idx_ in self.centroids:
 				centroid = []
 				sum_membeship = 0
 				for idx, xi in enumerate(x):
 					centroid.append(powers[idx][idx_] * np.array(xi))
 					sum_membeship += powers[idx][idx_]
 				centroid = np.sum(centroid, axis=0)
 				centroid = centroid/sum_membeship
 				self.centroids[idx_] = centroid

 			max_episilon = 0.0
 			for idx_ in self.centroids:
 				for idx, xi in enumerate(x):
 					updated_degree_of_membership = 0.0
 					norm = np.linalg.norm(xi - self.centroids[idx_])
 					all_norms = [norm/np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
 					all_norms = np.power(all_norms, 2/(self.fuzzy_c-1))
 					updated_degree_of_membership = 1/sum(all_norms)
 					diff = updated_degree_of_membership - self.degree_of_membership[idx][idx_]
 					self.degree_of_membership[idx][idx_] = updated_degree_of_membership

 					if diff > max_episilon:
 						max_episilon = diff
 			if max_episilon <= self.tolerance:
 				break

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
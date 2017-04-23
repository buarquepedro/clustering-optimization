import numpy as np


class KMeans(object):
    def __init__(self, n_clusters=2, n_iter=300, shuffle=True, tolerance=0.001):
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.shuffle = shuffle
        self.run = False

    def fit(self, x):
        self.run = True
        self.centroids = {}

        if len(x.shape) < 1:
            raise Exception("DataException: Dataset must contain more examples" +
                            "than the required number of clusters!")
        if self.shuffle:
            r = np.random.permutation(x.shape[0])
            for k in range(len(r[:self.n_clusters])):
                self.centroids[k] = x[r[k]]
        else:
            for k in range(self.n_clusters):
                self.centroids[k] = x[k]

        for itr in range(self.n_iter):
            self.clusters = {}

            for k in range(self.n_clusters):
                self.clusters[k] = []

            for xi in x:
                dist = [np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                class_ = dist.index(min(dist))
                self.clusters[class_].append(xi)

            old_centroids = dict(self.centroids)
            for k in self.clusters:
                self.centroids[k] = np.average(self.clusters[k], axis=0)

            is_done = True
            for k in self.centroids:
                old_centroid = old_centroids[k]
                centroid = self.centroids[k]
                if (np.linalg.norm(old_centroid - centroid) > self.tolerance):
                    is_done = False

            if is_done:
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

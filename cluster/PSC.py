import numpy as np
from PSO import PSO

class PSC(object):
    def __init__(self, n_clusters=2, swarm_size=100, n_iter=500, w=0.72, lb_w=0.4, w_damp=None, c1=1.49, c2=1.49):
        self.n_clusters = n_clusters
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.w = w
        self.lb_w = lb_w
        self.c1 = c1
        self.c2 = c2
        self.run = False
        self.v_max = None

        if w_damp is None:
            self.w_damp = self.w - self.lb_w

    def fit(self, data):
        self.dim = data.shape[1]
        self.pso = PSO(dim=self.dim*self.n_clusters, minf=0, maxf=1, swarm_size=self.swarm_size, n_iter=self.n_iter, w=self.w,
                       lb_w=self.lb_w, c1=self.c1, c2=self.c2)
        self.pso.optimize(self.__objective_function, customizable=True, dim=self.dim, n_clusters=self.n_clusters,
                          data=data)

        self.centroids = {}
        raw_centroids = self.pso.global_optimum.pos.reshape((self.n_clusters, self.dim))

        for centroid in range(len(raw_centroids)):
            self.centroids[centroid] = raw_centroids[centroid]

    def predict(self, x):
        if self.run:
            if len(x.shape) > 1:
                class_ = []
                for c in self.centroids:
                    class_.append(np.sum((x - self.centroids[c].best_post) ** 2, axis=1))
                return np.argmin(np.array(class_).T, axis=1)
            else:
                dist = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
                class_ = dist.index(min(dist))
                return class_
        else:
            raise Exception("NonTrainedModelException: You must fit data first!")

    def __objective_function(self, particle, **kwargs):
        if ('dim' not in kwargs) or ('n_clusters' not in kwargs) or (('data' not in kwargs)):
            raise Exception('Illegal Arguments Exception: Expected arguments does not match!')

        dim = kwargs.get("dim")
        data = kwargs.get("data")
        n_clusters = kwargs.get("n_clusters")

        centroids = particle.reshape((n_clusters, dim))
        clusters = {}

        for k in range(n_clusters):
            clusters[k] = []

        for xi in data:
            dist = [np.linalg.norm(xi - centroids[c]) for c in range(len(centroids))]
            class_ = dist.index(min(dist))
            clusters[class_].append(xi)

        inter_cluster_sum = 0.0
        for c in range(len(centroids)):
            intra_sum = 0.0
            if len(clusters[c]) > 0:
                for point in clusters[c]:
                    intra_sum += np.linalg.norm(point - centroids[c])
                intra_sum = intra_sum / len(centroids[c])
            inter_cluster_sum += intra_sum
        inter_cluster_sum = inter_cluster_sum / len(centroids)
        return inter_cluster_sum
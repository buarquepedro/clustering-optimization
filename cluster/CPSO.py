import copy
import numpy as np

class Centroid(object):
    def __init__(self, data, idx):
        self.pos = data[idx]
        self.speed = [0.0 for _ in range(len(self.pos))]
        self.best_pos = self.pos
        self.cluster = []

class Particle(object):
    def __init__(self, n_clusters, data):
        self.centroids = {}
        self.cost = np.inf
        self.best_cost = self.cost

        for c in range(n_clusters):
            r = np.random.permutation(data.shape[0])[:n_clusters]
            self.centroids[c] = Centroid(data, r[c])

    def display_particle(self):
        for c in self.centroids:
            print "Centroid: " + str(self.centroids[c].best_pos)
        print "Cost: " + str(self.cost)

class CPSO(object):
    def __init__(self, n_clusters=2, swarm_size=100, n_iter=500, w=0.72, lb_w=0.4, w_damp=None, c1=1.49,
                 c2=1.49):
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
        if len(data.shape) < 1:
            raise Exception("DataException: Dataset must contain more examples" +
                            "than the required number of clusters!")

        self.run = True
        self.global_optimum = Particle(self.n_clusters, data)
        self.init_swarm(self.n_clusters, data)

        if self.v_max is None:
            self.min_f = data.ravel().min()
            self.max_f = data.ravel().max()
            self.v_max = 0.5 * np.absolute(self.max_f - self.min_f)

        for i in range(self.n_iter):
            for p in self.swarm:
                for point in data:
                    dist_ = []
                    for c in p.centroids:
                        dist = np.linalg.norm(point - p.centroids[c].pos)
                        dist_.append(dist)

                    idx = np.argmin(dist_)
                    p.centroids[idx].cluster.append(point)

                p.cost = self.objective_function(p)
                if p.cost < p.best_cost:
                    p.best_cost = p.cost
                    for c in p.centroids:
                        p.centroids[c].best_pos = p.centroids[c].pos

            for p in self.swarm:
                if p.best_cost < self.global_optimum.best_cost:
                    self.global_optimum = copy.deepcopy(p)

            for p in self.swarm:
                for c in p.centroids:
                    r1 = np.random.random(len(p.centroids[c].speed))
                    r2 = np.random.random(len(p.centroids[c].speed))
                    p.centroids[c].speed = self.w * np.array(p.centroids[c].speed) + self.c1 * r1 * (
                        p.centroids[c].best_pos - p.centroids[c].pos) + self.c1 * r2 * (
                    self.global_optimum.centroids[c].best_pos - p.centroids[c].pos)

                    if np.linalg.norm(p.centroids[c].speed) > self.v_max:
                        p.centroids[c].speed = self.v_max * p.centroids[c].speed / np.linalg.norm(p.centroids[c].speed)

                    p.centroids[c].pos = p.centroids[c].pos + p.centroids[c].speed
                    if (p.centroids[c].pos < self.min_f).any() or (p.centroids[c].pos > self.max_f).any():
                        p.centroids[c].pos[p.centroids[c].pos > self.max_f] = self.max_f
                        p.centroids[c].pos[p.centroids[c].pos < self.min_f] = self.min_f
                        p.centroids[c].speed = -1 * p.centroids[c].speed

            if (self.w > self.lb_w):
                self.w = (i / self.n_iter) * self.w_damp
        self.centroids = self.global_optimum.centroids

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

    def display_centroids(self):
        print "GLOBAL OPTIMUM:"
        self.global_optimum.display_particle()

    def init_swarm(self, n_clusters, data):
        self.swarm = []
        for i in range(self.swarm_size):
            p = Particle(n_clusters, data)
            self.swarm.append(p)

    def objective_function(self, particle):
        inter_cluster_sum = 0.0
        for c in particle.centroids:
            intra_sum = 0.0
            if len(particle.centroids[c].cluster) > 0:
                for point in particle.centroids[c].cluster:
                    intra_sum += np.linalg.norm(point - particle.centroids[c].pos)
                intra_sum = intra_sum / len(particle.centroids[c].cluster)
            inter_cluster_sum += intra_sum
        inter_cluster_sum = inter_cluster_sum / len(particle.centroids)
        return inter_cluster_sum
import numpy as np

class Centroid(object):
    def __init__(self, data):
        r = np.random.permutation(data.shape[0])
        self.pos = data[r[0]]
        self.speed = [0.0 for _ in range(dim)]
        self.cost = 0.0
        self.best_pos = self.pos
        self.best_cost = self.cost
        self.cluster = []

class Particle(object):
    def __init__(self, n_clusters, data):
        self.centroids = {}
        self.cost = np.inf
        for c in range(n_clusters):
            self.centroids[c] = Centroid(data)

class CPSO(object):
    def __init__(self, n_clusters=2, shuffle=True, swarm_size=100, n_iter=500, w=0.72, lb_w=0.4, w_damp=None, c1=1.49,
                 c2=1.49):
        self.n_clusters = n_clusters
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.w = w
        self.lb_w = lb_w
        self.c1 = c1
        self.c2 = c2
        self.global_optimum = np.inf
        self.shuffle = shuffle
        self.run = False
        self.v_max = None

        if w_damp is None:
            self.w_damp = self.w - self.lb_w

    def fit(self, data):
        if len(data.shape) < 1:
            raise Exception("DataException: Dataset must contain more examples" +
                            "than the required number of clusters!")

        self.run = True
        self.init_swarm(self.n_clusters, data)

        if self.v_max is None:
            min_f = data.ravel().min()
            max_f = data.ravel().max()
            self.v_max = 0.5 * np.absolute(max_f - min_f)

    def predict(self, x):
        pass

    def init_swarm(self, n_clusters, data):
        self.particles = []
        for i in range(self.swarm_size):
            p = Particle(n_clusters, data)
            p.cost = self.objective_function(p)
            self.particles.append(p)

    def objective_function(self, particle):
        inter_cluster_sum = 0.0
        for c in particle.centroids:
            intra_sum = 0.0
            for point in particle.centroids[c].cluster:
                intra_sum += np.linalg.norm(point - particle.centroids[c].pos)
            intra_sum = intra_sum/len(particle.centroids[c].cluster)
            inter_cluster_sum += intra_sum
        inter_cluster_sum = inter_cluster_sum/len(particle.centroids)
        return  inter_cluster_sum
import numpy as np

class Particle(object):
    def __init__(self, dim):
        self.pos = np.random.random(dim)
        self.speed = [0.0 for _ in range(dim)]
        self.best_pos = self.pos
        self.cost = 0.0
        self.best_cost = self.cost

class PSO(object):
    def __init__(self, dim, minf, maxf, min_init=None, max_init=None, swarm_size=100, n_iter=500, w=1, lb_w=0.2,
                 w_damp=None, c1=2, c2=2, v_max=None):
        self.dim = dim
        self.minf = minf
        self.maxf = maxf
        self.min_init = min_init
        self.max_init = max_init
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.w = w
        self.lb_w = lb_w
        self.c1 = c1
        self.c2 = c2
        self.global_optimum = np.inf
        self.candidates = []

        if w_damp is None:
            self.w_damp = self.w - self.lb_w

        if v_max is None:
            self.v_max = 0.5 * np.absolute(maxf - minf)

    def set_candidate(self, particle):
        self.candidates.append(particle)

    def __init_swarm(self, func_type, customizable=False, **kwargs):
        self.swarm = []
        self.global_optimum = Particle(self.dim)
        self.global_optimum.cost = np.inf

        for i in range(self.swarm_size):
            if len(self.candidates) > 0:
                particle = Particle(self.dim)
                particle.pos = self.candidates.pop()
                print 'Inserted Candidate:' + str(particle.pos)
            else:
                particle = Particle(self.dim)
                if (self.min_init is not None) and (self.max_init is not None):
                    particle.pos = np.random.uniform(self.min_init, self.max_init, self.dim)
                else:
                    particle.pos = np.random.uniform(self.minf, self.maxf, self.dim)
                print 'Random Candidate:' + str(particle.pos)

            if not customizable:
                particle.cost = self.__evaluate(particle.pos, func_type)
            else:
                particle.cost = self.__evaluate(particle.pos, func_type, customizable, **kwargs)
            particle.best_pos = particle.pos
            print 'Best Pos: ' + str(particle.best_pos)
            particle.best_cost = particle.cost
            print 'Cost: ' + str(particle.cost)
            print 'Best Pos: ' + str(particle.best_cost)
            print '--------------------------------------------------------------------------------------------'

            if particle.best_cost < self.global_optimum.cost:
                self.global_optimum.cost = particle.best_cost
            self.swarm.append(particle)
        print self.swarm_size
        print len(self.swarm)

    def __evaluate(self, x, func_type, customizable=False, **kwargs):
        if not customizable:
            return func_type(x)
        else:
            return  func_type(x, **kwargs)

    def optimize(self, func_type, customizable=False, **kwargs):

        if not customizable:
            self.__init_swarm(func_type)
        else:
            self.__init_swarm(func_type, customizable, **kwargs)
        self.best_cost = []

        for i in range(self.n_iter):
            self.best_cost.append(self.global_optimum.cost)

            for p in self.swarm:
                r1 = np.random.random(len(p.speed))
                r2 = np.random.random(len(p.speed))
                p.speed = self.w * np.array(p.speed) + self.c1 * r1 * (p.best_pos - p.pos) + self.c1 * r2 * (
                self.global_optimum.pos - p.pos)

                if np.linalg.norm(p.speed) > self.v_max:
                    p.speed = self.v_max * p.speed / np.linalg.norm(p.speed)

                p.pos = p.pos + p.speed
                if (p.pos < self.minf).any() or (p.pos > self.maxf).any():
                    p.pos[p.pos > self.maxf] = self.maxf
                    p.pos[p.pos < self.minf] = self.minf
                    p.speed = -1 * p.speed

                if not customizable:
                    p.cost = self.__evaluate(p.pos, func_type)
                else:
                    p.cost = self.__evaluate(p.pos, func_type, customizable, **kwargs)
                if p.cost < p.best_cost:
                    p.best_cost = p.cost
                    p.best_pos = p.pos

            for p in self.swarm:
                if p.best_cost < self.global_optimum.cost:
                    self.global_optimum.pos = p.best_pos
                    self.global_optimum.cost = p.best_cost

            if (self.w > self.lb_w):
                self.w = (i / self.n_iter) * self.w_damp
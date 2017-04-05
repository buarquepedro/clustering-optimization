import numpy as np 

class Particle(object):
        
    def __init__(self, dim):
        self.pos = np.random.random(dim)
        self.speed = [0.0 for _ in range(dim)]
        self.best_pos = self.pos
        self.cost = 0.0
        self.best_cost = self.cost
 
class PSO(object):
        
    def __init__(self, dim, minf, maxf, swarm_size=100, n_iter=500, w=1, lb_w=0.2, w_damp=0.99, c1=2, c2=2, v_max=None, hard_state=True):
        self.dim = dim
        self. minf = minf
        self.maxf = maxf
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.w = w
        self.lb_w = lb_w
        self.w_damp = w_damp
        self.c1 = c1
        self.c2 = c2
        self.global_optimum = np.inf
        self.hard_state = hard_state
        if v_max is None:
            self.v_max = 0.5*np.absolute(maxf - minf)

    def init_swarm(self, func_type):
        self.swarm = []
        self.global_optimum = Particle(self.dim)
        self.global_optimum.cost = np.inf

        for i in range(self.swarm_size):
            particle = Particle(self.dim)
            if self.hard_state:
                p = np.random.randint(self.dim)
                a = np.random.uniform(self.minf, 0.8*self.minf, p)
                b = np.random.uniform(0.8*self.maxf, self.maxf, self.dim-p)
                particle.pos = np.concatenate((a,b), axis=0)
                particle.pos = particle.pos[np.random.permutation(len(particle.pos))]
            else:
                particle.pos = np.random.uniform(self.minf, self.maxf, self.dim)
            particle.cost = self.evaluate(particle.pos, func_type)
            particle.best_pos = particle.pos
            particle.best_cost = particle.cost

            if particle.best_cost < self.global_optimum.cost:
                self.global_optimum.cost = particle.best_cost
            self.swarm.append(particle)

    def evaluate(self, x, func_type):
        return func_type(x)

    def optimize(self, func_type):
        self.init_swarm(func_type)
        self.best_cost = []

        for i in range(self.n_iter):
            self.best_cost.append(self.global_optimum.cost)

            for p in self.swarm:
                r1 = np.random.random(len(p.speed))
                r2 = np.random.random(len(p.speed))
                p.speed = self.w*np.array(p.speed) + self.c1*r1*(p.best_pos - p.pos) + self.c1*r2*(self.global_optimum.pos - p.pos)

                if np.linalg.norm(p.speed) > self.v_max:
                    p.speed = self.v_max * p.speed/np.linalg.norm(p.speed)

                p.pos = p.pos + p.speed
                p.pos[p.pos > self.maxf] = self.maxf
                p.pos[p.pos < self.minf] = self.minf
                p.cost = self.evaluate(p.pos, func_type)

                if p.cost < p.best_cost:
                    p.best_cost = p.cost
                    p.best_pos = p.pos

            for p in self.swarm:
                if p.best_cost < self.global_optimum.cost:
                    self.global_optimum.pos = p.best_pos
                    self.global_optimum.cost = p.best_cost
                    
            if (self.w > self.lb_w):
                self.w = self.w * self.w_damp
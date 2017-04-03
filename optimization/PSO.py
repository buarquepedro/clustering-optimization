import numpy as np 

class Particle(object):

	def __init__(self, dim):
		self.pos = np.random.random(dim)
		self.speed = [0.0 for _ in range(dim)]
		self.best_pos = np.random.random(dim)
		self.cost = 0.0
		self.best_cost = 0.0
 
class PSO(object):
	
	def __init__(self, dim, minf, maxf, swarm_size=100, n_iter=500, w=1, lb_w=0.2, w_damp=0.99, c1=2, c2=2):
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

 	def init_swarm(self, func_type):
 		self.swarm = []
 		self.global_optimum = Particle(self.dim)
 		self.global_optimum.cost = np.inf

 		for i in range(self.swarm_size):
 			particle = Particle(self.dim)
 			particle.pos = np.random.uniform(self.minf, self.maxf, self.dim)
 			particle.cost = self.evaluate(particle.pos, func_type)
 			particle.best_pos = particle.pos
 			particle.best_cost = particle.cost

 			if particle.best_cost < self.global_optimum.cost:
 				self.global_optimum.cost = particle.best_cost
 			self.swarm.append(particle)

 	def evaluate(self, x, func_type):
 		if func_type == 'sphere':
 			return np.sum(x ** 2)
 		elif func_type == 'rosenbrock':
 			sum_ = 0.0
 			for i in range(1, len(x)-1):
 				sum_ += 100 * (x[i+1] - x[i]**2 )** 2 + (x[i] - 1)**2
 			return sum_

	def optimize(self, func_type):
		self.init_swarm(func_type)
		self.best_cost = [0.0 for _ in range(self.n_iter)]

		for i in range(self.n_iter):
			for p in self.swarm:
				r1 = np.random.random(len(p.speed))
				r2 = np.random.random(len(p.speed))
				p.speed = self.w*np.array(p.speed) + self.c1*r1*(p.best_pos - p.pos) + self.c1*r1*(self.global_optimum.pos - p.pos)
				p.pos = p.pos + p.speed
				p.cost = self.evaluate(p.pos, func_type)

				if p.cost < p.best_cost:
					p.best_cost = p.cost
					p.best_pos = p.pos
					if p.best_cost < self.global_optimum.cost:
						self.global_optimum.pos = p.best_pos
 						self.global_optimum.cost = p.best_cost

 				if (self.w > self.lb_w):
 					self.w = self.w * self.w_damp
 			self.best_cost.append(self.global_optimum.cost)
	
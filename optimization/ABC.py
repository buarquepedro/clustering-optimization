import numpy as np 

class Bee(object):

 	def __init__(self, dim):
 	 	self.pos = np.random.random(dim)
 	 	self.cost = float(0.0)
        
class ABC(object):
	
 	def __init__(self, dim, minf, maxf, min_init=None, max_init=None, colony_size=100, n_iter=500, acc_coef=1, abandonment_limit_coef=None):
 	 	self.dim = dim
 	 	self.minf = minf
 	 	self.maxf = maxf
 	 	self.min_init = min_init
 	 	self.max_init = max_init
 	 	self.colony_size = colony_size
 	 	self.n_iter = n_iter
 	 	self.on_looker_bees = self.colony_size
 	 	self.acc_coef = acc_coef

 	 	if abandonment_limit_coef is None:
 	 	 	self.abandonment_limit_coef = int(0.6 * self.colony_size * self.dim)

 	def init_bee(self, func_type):
 	 	bee = Bee(self.dim)
 	 	if (self.min_init is not None) and (self.max_init is not None):
 	 	 	bee.pos = np.random.uniform(self.min_init, self.max_init, self.dim)
 	 	else:
 	 	 	bee.pos = np.random.uniform(self.minf, self.maxf, self.dim)
 	 	bee.cost = self.evaluate(bee.pos, func_type)
 	 	return bee

 	def init_bees(self, func_type):
 	 	self.colony = []
 	 	self.global_optimum = self.init_bee(func_type)
 	 	self.global_optimum.cost = np.inf

 	 	for i in range(self.colony_size):
 	 	 	bee = self.init_bee(func_type)
 	 	 	if bee.cost < self.global_optimum.cost:
 	 	 		self.global_optimum.pos = bee.pos
 	 	 	 	self.global_optimum.cost = bee.cost
 	 	 	self.colony.append(bee)

 	def evaluate(self, x, func_type):
 	 	return func_type(x)

 	def roulette_wheel_selection(self, prob):
 	 	total_prob = np.sum(prob)
 	 	p = float(np.random.uniform(0,1,1))
 	 	for i in range(len(prob)):
 	 	 	p = p - prob[i]
 	 	 	if (p <= 0):
 	 	 	 	return i
 	 	return len(prob)-1

	def optimize(self, func_type):
		self.init_bees(func_type)
		self.abandonment_counter = [0 for i in range(self.colony_size)]
		self.best_cost = []

		for i in range(self.n_iter):
			self.best_cost.append(self.global_optimum.cost)

			for b in range(self.colony_size):
				k = range(self.colony_size)
				k.remove(b)
				k = np.random.choice(np.array(k))
				phi = self.acc_coef * np.random.uniform(-1, 1, self.dim)

				new_bee = Bee(self.dim)
				new_bee.pos = self.colony[b].pos + phi*(self.colony[b].pos - self.colony[k].pos)
				new_bee.cost = func_type(new_bee.pos)

				if new_bee.cost <= self.colony[b].cost:
					self.colony[b] = new_bee
				else:
					self.abandonment_counter[b] += 1

			F = [0 for i in range(self.colony_size)]
			mean_cost = np.mean(np.array([b.cost for b in self.colony]))
			for b in range(self.colony_size):
				if (mean_cost == 0.0):
					mean_cost = 0.001
				F[b] = np.exp(-self.colony[b].cost/mean_cost)
			P = F/np.sum(F)

			for b in range(self.on_looker_bees):
				idx = self.roulette_wheel_selection(P)

				k = range(self.on_looker_bees)
				k.remove(idx)
				k = np.random.choice(np.array(k))
				phi = self.acc_coef * np.random.uniform(-1, 1, self.dim)

				new_bee = Bee(self.dim)
				new_bee.pos = self.colony[idx].pos + phi*(self.colony[idx].pos - self.colony[k].pos)
				new_bee.cost = func_type(new_bee.pos)

				if new_bee.cost <= self.colony[b].cost:
					self.colony[b] = new_bee
				else:
					self.abandonment_counter[b] += 1

			for b in range(self.colony_size):
				if self.abandonment_counter[b] >= self.abandonment_limit_coef:
					self.colony[b] = self.init_bee(func_type)
					self.abandonment_counter[b] = 0

			for bee in self.colony:
				if bee.cost <= self.global_optimum.cost:
					self.global_optimum.pos = bee.pos
					self.global_optimum.cost = bee.cost
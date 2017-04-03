import numpy as np 
import matplotlib.pyplot as plt
from optimization import PSO

def main():
	opt = PSO.PSO(dim=5, minf=-5.12, maxf=5.12, swarm_size=30, n_iter=10000, w=0.8, w_damp=0.99, c1=2.05, c2=2.05)
	opt.optimize('sphere')

	plt.plot([(i+1) for i in range(len(opt.best_cost))], opt.best_cost)
	plt.title('Particle Swarm Optimization')
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')
	plt.show()

if __name__ == '__main__':
	main()
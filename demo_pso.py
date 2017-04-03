import numpy as np 
import matplotlib.pyplot as plt
from optimization import PSO

def main():
	opt1 = PSO.PSO(dim=30, minf=-5.12, maxf=5.12, swarm_size=30, n_iter=10000, w=0.8, lb_w=0.4, w_damp=0.99, c1=2.05, c2=2.05)
	opt1.optimize('sphere')

	opt2 = PSO.PSO(dim=30, minf=-30, maxf=30, swarm_size=30, n_iter=10000, w=0.8, lb_w=0.4, w_damp=0.99, c1=2.05, c2=2.05)
	opt2.optimize('rosenbrock')

	plt.subplot(121)
	plt.plot([(i+1) for i in range(len(opt1.best_cost))], opt1.best_cost, c='b')
	plt.title('Particle Swarm Optimization')
	plt.legend(["Sphere"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(122)
	plt.plot([(i+1) for i in range(len(opt2.best_cost))], opt2.best_cost, c='r')
	plt.title('Particle Swarm Optimization')
	plt.legend(["Rosenbrock"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')
	plt.show()

if __name__ == '__main__':
	main()
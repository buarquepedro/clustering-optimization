import numpy as np 
import matplotlib.pyplot as plt
from optimization.PSO import PSO
from optimization.ObjectiveFunction import ObjectiveFunction

def main():
	opt1 = PSO(dim=30, minf=-100, maxf=100, swarm_size=30, n_iter=5000, w=0.8, lb_w=0.4, w_damp=0.99, c1=2.05, c2=2.05)
	opt1.optimize(ObjectiveFunction.sphere_function)

	opt2 = PSO(dim=30, minf=-30, maxf=30, swarm_size=30, n_iter=5000, w=0.8, lb_w=0.4, w_damp=0.99, c1=2.05, c2=2.05)
	opt2.optimize(ObjectiveFunction.rosenbrock_function)

	opt3 = PSO(dim=30, minf=-5.12, maxf=5.12, swarm_size=30, n_iter=5000, w=0.8, lb_w=0.4, w_damp=0.99, c1=2.05, c2=2.05)
	opt3.optimize(ObjectiveFunction.rastrigin_function)

	plt.subplot(131)
	plt.plot([(i+1) for i in range(len(opt1.best_cost))], opt1.best_cost, c='b')
	plt.title('PSO - Sphere Function')
	plt.legend(["Sphere"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(132)
	plt.plot([(i+1) for i in range(len(opt2.best_cost))], opt2.best_cost, c='r')
	plt.title('PSO - Rosenbrock Function')
	plt.legend(["Rosenbrock"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(133)
	plt.plot([(i+1) for i in range(len(opt3.best_cost))], opt3.best_cost, c='g')
	plt.title('PSO - Rastrigin Function')
	plt.legend(["Rastrigin"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')
	plt.show()

if __name__ == '__main__':
	main()
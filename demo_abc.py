import numpy as np 
import matplotlib.pyplot as plt
from optimization.ABC import ABC
from optimization.ObjectiveFunction import ObjectiveFunction

def main():
	opt1 = ABC(dim=30, minf=-100, maxf=100, min_init=50, max_init=100, colony_size=50, n_iter=1000)
	opt1.optimize(ObjectiveFunction.sphere_function)

	opt2 = ABC(dim=30, minf=-30, maxf=30, min_init=15, max_init=30, colony_size=50, n_iter=1000)
	opt2.optimize(ObjectiveFunction.rosenbrock_function)

	opt3 = ABC(dim=30, minf=-5.12, maxf=5.12, min_init=2.56, max_init=5.12, colony_size=50, n_iter=1000)
	opt3.optimize(ObjectiveFunction.rastrigin_function)

	opt4 = ABC(dim=30, minf=-100, maxf=100, min_init=50, max_init=100, colony_size=50, n_iter=1000)
	opt4.optimize(ObjectiveFunction.schwefel_function)

	opt5 = ABC(dim=30, minf=-600, maxf=600, min_init=300, max_init=600, colony_size=50, n_iter=1000)
	opt5.optimize(ObjectiveFunction.griewank_function)

	opt6 = ABC(dim=30, minf=-32, maxf=32, min_init=16, max_init=32, colony_size=50, n_iter=1000)
	opt6.optimize(ObjectiveFunction.ackley_function)

	plt.subplot(231)
	plt.plot([(i+1) for i in range(len(opt1.best_cost))], opt1.best_cost, c='b')
	plt.legend(["Sphere"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(232)
	plt.plot([(i+1) for i in range(len(opt2.best_cost))], opt2.best_cost, c='r')
	plt.legend(["Rosenbrock"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(233)
	plt.plot([(i+1) for i in range(len(opt3.best_cost))], opt3.best_cost, c='g')
	plt.legend(["Rastrigin"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(234)
	plt.plot([(i+1) for i in range(len(opt4.best_cost))], opt4.best_cost, c='c')
	plt.legend(["Schwefel"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(235)
	plt.plot([(i+1) for i in range(len(opt5.best_cost))], opt5.best_cost, c='m')
	plt.legend(["Griewank"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')

	plt.subplot(236)
	plt.plot([(i+1) for i in range(len(opt6.best_cost))], opt6.best_cost, c='y')
	plt.legend(["Ackley"])
	plt.xlabel('Iterations')
	plt.ylabel('Fitness')
	plt.show()

if __name__ == '__main__':
	main()
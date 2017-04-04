import math
import numpy as np 

class ObjectiveFunction(object):

    @staticmethod
    def sphere_function(x):
        return np.sum(x ** 2)

    @staticmethod
    def rosenbrock_function(x):
        sum_ = 0.0
        for i in range(1, len(x)-1):
            sum_ += 100 * (x[i+1] - x[i]**2 )** 2 + (x[i] - 1)**2
        return sum_

    @staticmethod
    def rastrigin_function(x):
        f_x = [xi**2 - 10*math.cos(2*math.pi*xi) + 10 for xi in x]
        return sum(f_x)

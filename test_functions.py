import numpy as np


class TestFunction:
    def __str__(self):
        return type(self).__name__

    def func(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented

    def hessian(self, x):
        raise NotImplemented


class Rosenbrock(TestFunction):
    def func(self, x):
        return sum(100 * (x[i+1] - x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))
    
    def derivative(self, x):
        res = []
        res.append(-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]))
        for i in range(1, len(x) - 1):
            res.append(200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i]))
        res.append(200 * (x[-1] - x[-2]**2))
        return np.array(res)
    

rosenbrock = Rosenbrock()

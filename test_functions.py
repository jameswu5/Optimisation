import numpy as np


class TestFunction:
    def __str__(self):
        return type(self).__name__

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def hessian(self, x):
        raise NotImplementedError


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

    def hessian(self, x):
        if len(x) != 2:
            raise NotImplementedError

        x1, x2 = x
        return np.array([
            [1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1],
            [-400 * x1, 200]
        ])


rosenbrock = Rosenbrock()

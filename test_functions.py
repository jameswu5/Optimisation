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


class Ackley(TestFunction):
    def func(self, x):
        assert len(x) == 2
        x1, x2 = x
        return -20 * np.exp(-0.2 * np.sqrt(0.5*(x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20
    
    # I think this is wrong
    def derivative(self, x):
        assert len(x) == 2
        x1, x2 = x
        r = np.sqrt(x1**2 + x2**2)
        return np.array([
            (2**1.5 * x1 * np.exp(-0.2 * np.sqrt(0.5) * r)) / r + np.pi * np.sin(2 * np.pi * x1) * np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))),
            (2**1.5 * x2 * np.exp(-0.2 * np.sqrt(0.5) * r)) / r + np.pi * np.sin(2 * np.pi * x2) * np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
        ])
    # Shun modified the derivative


class Himmelblau(TestFunction):
    def func(self, x):
        assert len(x) == 2
        x1, x2 = x
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    
    def derivative(self, x):
        assert len(x) == 2
        x1, x2 = x
        return np.array([
            4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7),
            4 * x2 * (x1 + x2**2 - 7) + 2 * (x1**2 + x2 - 11)
        ])
    
    def hessian(self, x):
        assert len(x) == 2
        x1, x2 = x
        return np.array([
            [12 * x1**2 + 4*x2 - 42, 4*x1 + 4*x2],
            [4*x1 + 4*x2, 4*x1 + 12*x2**2 - 26]
        ])


class Rastrigin(TestFunction):
    def func(self, x):
        n = len(x)
        return 10 * n + sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(n))
    
    def derivative(self, x):
        return np.array([2 * x[i] + 20 * np.pi * np.sin(2 * np.pi * x[i]) for i in range(len(x))])

    def hessian(self, x):
        return np.diag([2 + 40 * np.pi**2 * np.cos(2 * np.pi * x[i]) for i in range(len(x))])


rosenbrock = Rosenbrock()
ackley = Ackley()
himmelblau = Himmelblau()
rastrigin = Rastrigin()

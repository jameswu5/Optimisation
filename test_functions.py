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
        res = np.zeros((len(x), len(x)))
        res[0][0] = 1200 * x[0]**2 - 400 * x[1] + 2
        res[0][1] = -400 * x[0]
        for i in range(1, len(x) - 1):
            res[i][i-1] = -400 * x[i-1]
            res[i][i] = 202 + 1200 * x[i]**2 - 400 * x[i+1]
            res[i][i+1] = -400 * x[i]
        res[-1][-2] = -400 * x[-2]
        res[-1][-1] = 200
        return res


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


class Polynomial(TestFunction):
    def __init__(self, coefficients):
        # The ith index is the coefficient of the x^i term
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def func(self, x):
        assert len(x) == 1
        res = 0
        cur = 1
        for i in range(self.degree + 1):
            res += cur * self.coefficients[i]
            cur *= x[0]
        return res
    
    def derivative(self, x):
        assert len(x) == 1
        res = 0
        cur = 1
        for i in range(1, self.degree + 1):
            res += i * cur * self.coefficients[i]
            cur *= x[0]
        return np.array([res])
    
    def hessian(self, x):
        assert len(x) == 1
        res = 0
        cur = 1
        for i in range(2, self.degree + 1):
            res += i * (i-1) * cur * self.coefficients[i]
            cur *= x[0]
        return np.array([[res]])


rosenbrock = Rosenbrock()
ackley = Ackley()
himmelblau = Himmelblau()
rastrigin = Rastrigin()

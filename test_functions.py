import numpy as np


class TestFunction:

    def __str__(self):
        return type(self).__name__

    def func(self, x):
        pass

    def derivative(self, x):
        pass


class Rosenbrock(TestFunction):
    def func(self, x):
        x1, x2 = x
        return 100 * (x2 - x1**2)**2 + (1-x1)**2
    
    def derivative(self, x):
        x1, x2 = x
        return np.array([
            -400*x1*(x2-x1**2) - 2*(1-x1),
            200*(x2-x1**2)
        ])
    
rosenbrock = Rosenbrock()

import numpy as np
from trm import trm_cauchy, trm_dogleg, trm_subspace, cauchy, dogleg, subspace
from trust_region_modification import trm_subproblem, SR1_trm, subproblem_solve
from test_functions import rosenbrock, himmelblau, rastrigin, ackley, Polynomial

def test_trm(methods, functions, delta0, delta_max, eta, iter_time, tolerance):
    """
    methods, functions must be an iterable (i.e. list)

    """
    results = []
    for mtd in methods:
        for func in functions:
            x0 = np.random.uniform(low=-10.0, high=10.0, size=(2,))
            x_star = mtd(func.func, func.derivative, func.hessian, x0, delta0, delta_max, eta, iter_time, tolerance)
            f_star = func.func(x_star)
            results.append({
                'x0': x0,
                'method': mtd.__name__,
                'function': str(func),
                'x_opt': x_star,
                'f_opt': f_star
            })
    return results

def test_trm_local(methods, functions, delta0, delta_max, eta, iter_time, tolerance):
    """
    methods must be an iterable (i.e. list)
    functions must be an iterable
    This one tries points in a disc, r=0.5, centre = (0,0)

    """
    results = []
    for mtd in methods:
        for func in functions:
            # uniformly distributed in a disc
            sqrt_r = np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2*np.pi)
            x0 = 0.5*np.array([sqrt_r*np.cos(theta), sqrt_r*np.sin(theta)])

            x_star = mtd(func.func, func.derivative, func.hessian, x0, delta0, delta_max, eta, iter_time, tolerance)
            f_star = func.func(x_star)
            results.append({
                'x0': x0,
                'method': mtd.__name__,
                'function': str(func),
                'x_opt': x_star,
                'f_opt': f_star
            })
    return results


def test_trm_poly(methods, functions, delta0, delta_max, eta, iter_time, tolerance):
    """
    For polynomial

    """
    results = []
    for mtd in methods:
        for func in functions:
            x0 = [np.random.uniform(low=-5.0, high=5.0)]
            x_star = mtd(func.func, func.derivative, func.hessian, x0, delta0, delta_max, eta, iter_time, tolerance)
            f_star = func.func(x_star)
            results.append({
                'x0': x0,
                'method': mtd.__name__,
                'function': str(func),
                'x_opt': x_star,
                'f_opt': f_star
            })
    return results



def test_SR1_trm(submethod, func, delta0, eta, iter_time, tolerance):
    """
    For Ackley
    This one tries points in a disc, r=0.5, centre = (0,0)

    """
    results = []
    # sqrt_r = np.sqrt(np.random.uniform(0, 1))
    # theta = np.random.uniform(0, 2*np.pi)
    # x0 = 0.5*np.array([sqrt_r*np.cos(theta), sqrt_r*np.sin(theta)])
    x0 = np.random.uniform(0, 0.5, size=2)
    print(x0)

    x_star = SR1_trm(submethod, func.func, func.derivative, x0, delta0, eta, iter_time, tolerance)
    f_star = func.func(x_star)
    results.append({
        'x0': x0,
        'method': submethod.__name__,
        'function': str(func),
        'x_opt': x_star,
        'f_opt': f_star
    })
    return results


# result = test_trm([trm_cauchy, trm_dogleg, trm_subspace, trm_subproblem],
#                   [rastrigin, himmelblau, rosenbrock],
#                   0.2, 0.5, 0.05, 50, 1e-8)


#print(test_SR1_trm(subproblem_solve, rastrigin, 0.1, 1e-4, 20, 1e-8))

print(SR1_trm(subproblem_solve, rastrigin.func, rastrigin.derivative, [0.5,0.01], 0.04, 0.5*1e-3, 40, 1e-8 ))

coefficients = np.array([3, -5, 4, 1])
poly = Polynomial(coefficients)

result = test_trm_poly([trm_cauchy, trm_dogleg, trm_subspace, trm_subproblem], [poly],
                        0.2, 0.5, 0.05, 50, 1e-8)

for res in result:
    print(res)
    print()

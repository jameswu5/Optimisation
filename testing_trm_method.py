import numpy as np


def test_trm(methods, functions, delta0, delta_max, eta, iter_time, tolerance):
    results = []
    for mtd in methods:
        for func in functions:
            x0 = np.random.uniform(low=-10.0, high=10.0, size=(2,))
            x_star = mtd(func.func, x0, delta0, delta_max, eta, iter_time, tolerance)
            f_star = func.func(x_star)
            results.append({
                'method': mtd.__name__,
                'function': str(func),
                'x_opt': x_star,
                'f_opt': f_star
            })
    return results

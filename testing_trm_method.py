import numpy as np
import matplotlib.pyplot as plt
from trm import trm_cauchy, trm_dogleg, trm_subspace, cauchy, dogleg, subspace
from trust_region_modification import trm_subproblem, SR1_trm, SR1_algo, subproblem_solve, ConvergenceError
from test_functions import rosenbrock, himmelblau, rastrigin, ackley, Polynomial, sphere


"""
Need convergence plot, iterations, color plot
Need to test subproblem ones again, now algorithm modified
"""

def test_trm_global(methods, functions, delta0, delta_max, eta, iter_time, tolerance, num_restarts=100):
    """
    methods, functions must be an iterable (i.e. list)

    """
    results = []
    for mtd in methods:
        for func in functions:
            best_f_star = np.inf
            best_x_star = None
            best_x0 = None
            for _ in range(num_restarts):
                x0 = np.random.uniform(low=-5.0, high=5.0, size=(2,))
                x_star = mtd(func.func, func.derivative, func.hessian, x0, delta0, delta_max, eta, iter_time, tolerance)
                f_star = func.func(x_star)
                if f_star < best_f_star:
                    best_f_star = f_star
                    best_x_star = x_star
                    best_x0 = x0
            results.append({
                'x0': best_x0,
                'method': mtd.__name__,
                'function': str(func),
                'x_opt': best_x_star,
                'f_opt': best_f_star
            })
    return results

# print(test_trm_global([trm_cauchy], [sphere],0.1, 0.5, 1e-4, 10, 1e-7, 100))

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

# test code
# for r in np.linspace(0,1,100):
#     print(SR1_trm(subproblem_solve, rosenbrock.func, rosenbrock.derivative, [-2, 3],0.01, 1e-4, 20, r))


def iterates(submethod, func, x0, delta0, delta_max, eta, iter_time, tolerance):
    """Return the iterates of the algorithm"""
    x = x0
    f = func.func
    df = func.derivative
    B = func.hessian
    delta = delta0
    res = [np.copy(x)]
    
    for i in range(iter_time):
        p = submethod(df(x), B(x), delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(df(x), p) - 0.5 * np.dot(p.T, np.dot(B(x), p))
        # Avoid division by zero or invalid value in rho
        if pr == 0:
            rho = 0
        else:
            rho = ar / pr

        # Adjust the trust region radius
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_max)
        else:
            delta = delta

        # Update the iteration point
        if rho > eta:
            x = x + p
        else:
            x = x
        res.append(np.copy(x))

        # Check if the current solution closed enough to the optimal solution
        if np.linalg.norm(df(x)) < tolerance:
            break

    return res


# copied, adapted from testing.py

def log_error_plot(xlis):
    # treat final point as equilibrium
    eq = xlis[-1]
    errs = []
    for i in range(len(xlis) - 1):  # avoid final point, zero error for log
        try:
            err = np.linalg.norm(eq - xlis[i])
        except Exception:
            err = np.inf
        errs.append(err)

    plt.plot(errs)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Log Error")
    plt.show()

def convergence_plot(xlis, func):
    f_val = [func.func(x) for x in xlis]

    plt.plot(range(len(f_val)), f_val)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.yscale('log')
    plt.show()

############################################### messy code testing

# log_error_plot(iterates(subspace, rosenbrock, [1.5,1.5], 0.1, 1, 0.1, 100, 1e-6))

# def convergence_plot_a(xlis):
#     # treat final point as equilibrium
#     eq = xlis[-1]
#     # eq = [0,0]
#     errs = [np.linalg.norm(eq-xlis[i]) for i in range(len(xlis))] # avoid final point, zero error for log

#     plt.plot(errs)
#     plt.yscale('log')
#     plt.xlabel('Iteration')
#     plt.ylabel('Log Error')
#     plt.savefig("ackley_convergence_subproblem_2_2_")
#     plt.show()

# arr = np.array(SR1_algo(subproblem_solve, rosenbrock.func, rosenbrock.derivative, [1.1, 1.1], 0.01, 1e-4, 100000))
# plt.plot(arr)
# plt.show()
#convergence_plot(iterates(subproblem_solve, rosenbrock, [1.1, 1.1], 0.01, 0.5, 1e-4, 300, 1e-8))

# RuntimeWarning: invalid value encountered in sqrt
# tau = np.sqrt(delta**2 - np.linalg.norm(components)**2) inside 1e-

# arr = np.array(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [2, 2], 1e-6, 1e-4, 200))
# print(arr[-1])
# print(arr)
# print(len(arr))
# plt.plot([ackley.func(x) for x in arr])
# plt.xlabel('Iteration')
# plt.ylabel('f(x)')
# plt.savefig("ackley_objective_2_2")
# plt.show()
# convergence_plot_a(arr)

# print(cauchy(np.array([-2.07199,0]), np.array([[30.67876,0], [0,51.53711]]), 0.01))
# runtime error
# np.array(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [0.4,0.1], 1e-6, 1e-4, 100))

# print(subproblem_solve(np.array([-2.71853715,0.78074053]), np.array([[1.78584619e+16,6.21831331e+16], [6.21831331e+16,-4.47141260e+16]]), 1e-6))
# print(len(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [-0.05,-0.15], 1e-6, 1e-4, 200)))
# rho turns to 0, difference so small, Sso no movement


# arr = np.array(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [2,2], 1e-6, 1e-4, 100))
# plt.plot([ackley.func(x) for x in arr])
# print(arr)
# print(arr[-1], ackley.derivative(arr[-1]))
# plt.xlabel("Iterations")
# plt.ylabel("Objective function value")
# # plt.legend(['x-coord', 'y-coord'])
# # plt.savefig("coordinates")
# plt.show()
# print(len(arr))
# convergence_plot_a(SR1_algo(cauchy, ackley.func, ackley.derivative, [0,0], 1e-6, 1e-4, 100))

# print(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [0.001,0.001], 1e-6, 1e-4, 200))
# arr = np.array(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [0.9,0.0], 1e-2, 1e-4, 200))
# plt.plot(arr[2:])
# plt.legend(['x-coord', 'y-coord'])
# plt.show()
# convergence_plot(SR1_algo(subproblem_solve, ackley.func, ackley.derivative, [0.001,0.001], 1e-2, 1e-4, 200))
# seems like get negative rho= ared/pred, remain while trust region radius gets smaller

# for r in range(10):
#     print(subproblem_solve([-2.07199,0],[[30.67876,0],[0,51.53711]],10**(-r)))
#     print()

################################################## testing

def get_sol_numbers(known_eqs, Z, tolerance=1e-6):
    """
    Return index number of equilibrium point

    known_eqs (list of numpy arrays): list storing the known equilibrium points
    Z (2D list): stores a grid of points that are returned by descend
    """
    def get_sol_num(z):
        for i in range(len(known_eqs)):
            if np.linalg.norm(known_eqs[i] - z) < tolerance:
                return i
        return -1 # no equilibrium
    
    return [[get_sol_num(Z[i][j]) for i in range(len(Z))] for j in range(len(Z[0]))]


def himmelblau_plot(mtd, func, delta0, delta_max, eta, iter_time, tolerance):
    """
    Displays which equilibrium point each point converges to in a colour plot
    Takes - seconds to run for step=0.1

    function must be 2d
    """
    
    known_eqs = [
        np.array([3.0, 2.0]),
        np.array([-2.805118, 3.131312]),
        np.array([-3.779310, -3.283186]),
        np.array([3.584428, -1.848126])
    ]

    step = 0.1
    minimum = -5.0
    maximum = 5.0

    xs = np.arange(minimum, maximum + step, step)
    ys = np.arange(minimum, maximum + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]

    for i in range(len(X)):
        for j in range(len(Y)):
            x,y = xs[i], ys[j]
            try:
                Z[i][j] = mtd(func.func, func.derivative, func.hessian, np.array([x,y]), delta0, delta_max, eta, iter_time, tolerance)
                print(i, j, Z[i][j])
            except:
                Z[i][j] = -1
    
    sols = get_sol_numbers(known_eqs, Z)

    plt.pcolormesh(X, Y, sols)
    plt.colorbar()
    plt.show()

def himmelblau_plot_SR1(mtd, func, delta0, delta_max, eta, iter_time, tolerance):
    """
    Displays which equilibrium point each point converges to in a colour plot
    Takes - seconds to run for step=0.1

    function must be 2d
    """
    
    known_eqs = [
        np.array([3.0, 2.0]),
        np.array([-2.805118, 3.131312]),
        np.array([-3.779310, -3.283186]),
        np.array([3.584428, -1.848126])
    ]

    step = 0.1
    minimum = -5.0
    maximum = 5.0

    xs = np.arange(minimum, maximum + step, step)
    ys = np.arange(minimum, maximum + step, step)
    X, Y = np.meshgrid(xs, ys)
    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]

    for i in range(len(X)):
        for j in range(len(Y)):
            x,y = xs[i], ys[j]
            try:
                Z[i][j] = SR1_trm(mtd, func.func, func.derivative, np.array([x,y]), delta0, eta, iter_time, tolerance)
                print(i, j, Z[i][j])
            except ConvergenceError:
                Z[i][j] = -1
    
    sols = get_sol_numbers(known_eqs, Z)

    plt.pcolormesh(X, Y, sols)
    plt.colorbar()
    plt.show()

# test code
# himmelblau_plot_SR1(dogleg, himmelblau, 0.2, 0.5, 0.05, 50, 1e-8)
# already errors, some having NoneType, repeat
# himmelblau_plot_SR1(cauchy, himmelblau, 0.2, 0.5, 0.05, 50, 1e-8)
# convergence errors (50 iterations)
# himmelblau_plot_SR1(subspace, himmelblau, 0.2, 0.5, 0.05, 50, 1e-8)
# a lot of convergence errors, bigger scar glitches there, no NoneType (safeguarded in code)
# himmelblau_plot_SR1(subproblem_solve, himmelblau, 0.2, 0.5, 0.05, 50, 1e-8)
# Safeguard needs upgrade if want fewer iterations, but time constraints

def number_of_iterations_plot(submethod, func, width, density=100):
    x = np.linspace(-width, width, density)
    y = np.linspace(-width, width, density)
    X, Y = np.meshgrid(x, y)
    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]

    for i in range(len(X)):
        print(X[0][i])
        for j in range(len(Y)):
            try:
                Z[i][j] = len(iterates(submethod, func, np.array([X[i][j],Y[i][j]]), 0.1, 0.5, 0.05, 200, 1e-8))
                #print(i, j, Z[i][j])
            except:
                Z[i][j] = np.inf
    
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.savefig("rosenbrock_trm_iter_subproblem.png")
    plt.show()


# test code
# number_of_iterations_plot(subspace, rosenbrock, 5, 100)
# Larger trust region for sphere only
# Cauchy - most points converge late


def number_of_iterations_plot_SR1(submethod, func, width, density=100):
    x = np.linspace(-width, width, density)
    y = np.linspace(-width, width, density)
    X, Y = np.meshgrid(x, y)
    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]

    for i in range(len(X)):
        print(X[0][i])
        for j in range(len(Y)):
            try:
                Z[i][j] = len(SR1_algo(submethod, func.func, func.derivative, np.array([X[i][j],Y[i][j]]),1e-2, 1e-4, iter_time=500))
                #print(i, j, Z[i][j])
            except ConvergenceError:
                Z[i][j] = np.inf
    
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.savefig("ackley_iter500_subproblem3_1e-6_SR1.png")
    plt.show()


###################### more messy testing
# number_of_iterations_plot_SR1(subproblem_solve, ackley, 5, 51)
# This can take few minutes to finish, esp with more points, max iterations


# result = test_trm([trm_cauchy, trm_dogleg, trm_subspace, trm_subproblem],
#                   [rastrigin, himmelblau, rosenbrock],
#                   0.2, 0.5, 0.05, 50, 1e-8)


# coefficients = np.array([3, -5, 4, 1])
# poly = Polynomial(coefficients)

# result = test_trm_poly([trm_cauchy, trm_dogleg, trm_subspace, trm_subproblem], [poly],
#                         0.2, 0.5, 0.05, 50, 1e-8)

# for res in result:
#     print(res)
#     print()
# array([ 9.70540728, -0.87380706])

# x0 = np.random.uniform(low=-10.0, high=10.0, size=(2,))

# step = 0.1
# minimum = -5.0
# maximum = 5.0

# xs = np.arange(minimum, maximum + step, step)
# ys = np.arange(minimum, maximum + step, step)
# X, Y = np.meshgrid(xs, ys)
# Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]
# x,y = xs[49], ys[28]
# trm_subproblem(himmelblau.func, himmelblau.derivative, himmelblau.hessian, np.array([x,y]), 0.2, 0.5, 0.05, 50, 1e-8)

# SR1
# convergence_plot(SR1_algo(subproblem_solve, rastrigin.func, rastrigin.derivative, [0.5,0.01], 0.04, 0.5*1e-3, 40, 1e-8))
################# end

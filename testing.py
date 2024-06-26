from step_selection import backtracking, wolfe
from test_functions import rosenbrock, himmelblau, ackley, rastrigin, sphere, Polynomial
from descent import Descent
import numpy as np
import matplotlib.pyplot as plt

from modification_methods import is_positive_definite

sp = Descent(sphere)
ro = Descent(rosenbrock)
hi = Descent(himmelblau)
ac = Descent(ackley)
ra = Descent(rastrigin)

def sphere_testing():
    x0 = np.array([1.2, 3.2, 6.2, -2.5, 0.2, 0.39])
    eq = sp.descend(x0, sp.steepest, wolfe)
    eq2 = sp.descend(x0, sp.newton, wolfe)
    print(eq)
    print(eq2)

    eq3 = sp.BFGS(x0)
    print(eq3)


def rosenbrock_testing():
    x0 = [0.5, 0.5]
    # x0 = np.array([1.2, 1.2])

    # x0 = [0.8, 0.7, 0.6, 1.1, 0.51]
    # x0 = [0.01, 0.05]

    # x1 = [-6, 7, 8]

    # x0 = [0, 1]
    x0 = [-13, 169]
    print(rosenbrock.derivative(x0))
    # hessian = rosenbrock.hessian(x0)
    # print(hessian)

    # sol = ro.descend(x0, ro.steepest, backtracking)
    # sol = ro.descend(x0, ro.newton, backtracking, display=True)
    # sol = ro.descend(x0, ro.steepest, wolfe)
    # sol = ro.descend(x0, ro.newton_diagonal_modification, wolfe, display=True)
    # sol = ro.BFGS(x0)
    # print(sol)


    # number_of_iterations_plot(ro, ro.newton, wolfe, width=2, density=100) # This is quite interesting

    # x2 = [1.1, 1.1, 1.1]

    # here we plot with rho=0.4, c=0.3

    # function_convergence_plot(ro, x2, ro.steepest, backtracking, xlog=False, ylog=True) # plotted with rho=0.4, c=0.3
    # function_convergence_plot(ro, x2, ro.newton, backtracking, xlog=False, ylog=True) # plotted with rho=0.4, c=0.3

    # function_evaluation_plot(ro, x2, ro.steepest, backtracking, xlog=False, ylog=True, save="images/rosenbrock_steepest_evaluation.png")
    # function_evaluation_plot(ro, x2, ro.newton, backtracking, xlog=False, ylog=True, save="images/rosenbrock_newton_evaluation.png")



def himmelblau_testing():
    """
    Newton method doesn't work, I suspect because hessian is not positive definite
    """
    h = Descent(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])

    eq1 = h.descend(p1, h.steepest, backtracking)  # converges to 1st equilibrium
    eq2 = h.descend(p2, h.steepest, backtracking)  # converges to 2rd equilibrium
    eq3 = h.descend(p3, h.steepest, backtracking)  # converges to 3nd equilibrium
    eq4 = h.descend(p4, h.steepest, backtracking)  # converges to 4th equilibrium
    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)

    # eq1 = h.descend(np.array([0.5, 0.5]), h.newton, backtracking, display=True)
    # hess =  himmelblau.hessian(np.array([0.5, 0.5]))
    # print(hess)
    # print(is_positive_definite(hess))

    # Strangely enough, on (-5, -5) wolfe and backtracking converges to different equilibria
    # sol = h.BFGS(p3, step_selection_mode=wolfe, display=True)
    
    # print(sol)


def modified_himmelblau_testing():
    """
    Introducing modified hessian to himmelblau
    """
    h = Descent(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])
    eq1 = h.descend(p1, h.newton_diagonal_modification, backtracking)
    eq2 = h.descend(p2, h.newton_diagonal_modification, backtracking)
    eq3 = h.descend(p3, h.newton_diagonal_modification, backtracking)
    eq4 = h.descend(p4, h.newton_diagonal_modification, backtracking) 
    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)


def rastrigin_testing():
    r = Descent(rastrigin)

    x0 = np.array([-0.1, 0.5])
    eq = r.descend(x0, r.newton, wolfe)
    eq2 = r.BFGS(x0)

    print(eq)
    print(eq2)

# We test this one on BFGS
def ackley_testing():
    a = Descent(ackley)

    # x0 = np.array([1.9, 3.5]) # converges to [2.97313942 1.98213995]
    x0 = np.array([0.2, -0.7]) # converges to [ 1.44368052e-09 -9.52166544e-01]
    # eq = a.BFGS(x0, display=True)
    # print(eq)

    # function_convergence_plot_bfgs(a, x0, save="images/ackley_convergence_bfgs.png")
    function_convergence_plot_bfgs(a, x0)

    # print(ackley.derivative(np.array([ 1.44368052e-09, -9.52166544e-01])))


def polynomial_testing():
    coefficients = np.array([3, -5, 4, 1])
    poly = Polynomial(coefficients)
    p = Descent(poly)
    x0 = np.array([-1.])
    print(p.descend(x0, p.newton, backtracking, display=True))


def himmelblau_plot():
    """
    Displays which equilibrium point each point converges to in a colour plot
    Takes 10 seconds to run for step=0.1
    """

    h = Descent(himmelblau)

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
            x, y = xs[i], ys[j]
            try:
                Z[i][j] = h.descend2D(x, y, h.steepest, wolfe).x
                print(i, j, Z[i][j])
            except:
                Z[i][j] = np.inf


    sols = get_solution_numbers(known_eqs, Z)

    plt.pcolormesh(X, Y, sols)
    plt.colorbar()
    plt.savefig("images/himmelblau_convergence_wolfe.png")
    # plt.show()


def get_solution_numbers(known_eqs, Z, tolerance=1e-6):
    """
    Returns the index number of each equilibrium point.

    known_eqs (list of numpy arrays): list storing the known equilibrium points
    Z (2D list): stores a grid of points that are returned by descend
    """
    def get_solution_number(z):
        for i in range(len(known_eqs)):
            if np.linalg.norm(known_eqs[i] - z) < tolerance:
                return i
        return np.inf  # No equilibrium found
    
    return [[get_solution_number(Z[i][j]) for i in range(len(Z))] for j in range(len(Z[0]))]


def rosenbrock_convergence_plot():
    x0 = [0.5, 0.5]
    xs = ro.descend(x0, ro.steepest, wolfe).xs
    convergence_plot(xs, xlog=False, ylog=True)


def himmelblau_convergence_plot():
    h = Descent(himmelblau)
    x0 = np.array([0.5, 3.])
    xs = h.descend(x0, h.steepest, backtracking).xs
    convergence_plot(xs, xlog=False, ylog=True)

def evaluation_plot(f:Descent, xs, xlog=False, ylog=True, save=None):
    """
    Plots the function evaluations at each iteration

    xs (list): a list of all the points x takes in each iteration of the descent algorithm
    """

    vals = [f.f(x) for x in xs]
    plt.plot(vals)
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    if xlog:
        plt.xscale('log')
        plt.xlabel("Log Iteration")
    if ylog:
        plt.yscale('log')
        plt.ylabel("Log f(x)")
    if save:
        plt.savefig(save)
    else:
        plt.show()

def function_evaluation_plot(f: Descent, x0, descent_mode, step_selection_mode, xlog=False, ylog=False, save=None):
    xs = f.descend(x0, descent_mode, step_selection_mode).xs
    evaluation_plot(f, xs, xlog, ylog, save=save)

def function_evaluation_plot_bfgs(f: Descent, x0, xlog=False, ylog=True, save=None):
    xs = f.BFGS(x0).xs
    evaluation_plot(f, xs, xlog, ylog, save=save)

def convergence_plot(xs, xlog=False, ylog=False, save=None):
    """
    Plots a log plot of how far each iteration is from the equilibrium

    xs (list): a list of all the points x takes in each iteration of the descent algorithm
    """

    # Treats the final point as the equlibrium
    eq = xs[-1]
    errs = [np.linalg.norm(eq - xs[i]) for i in range(len(xs) - 1)]  # avoid final point to avoid zero error

    plt.plot(errs)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    if xlog:
        plt.xscale('log')
        plt.xlabel("Log Iteration")
    if ylog:
        plt.yscale('log')
        plt.ylabel("Log Error")

    if save:
        plt.savefig(save)
    else:
        plt.show()


def function_convergence_plot(f: Descent, x0, descent_mode, step_selection_mode, xlog=False, ylog=False, save=None):
    xs = f.descend(x0, descent_mode, step_selection_mode).xs
    convergence_plot(xs, xlog=xlog, ylog=ylog, save=save)

def function_convergence_plot_bfgs(f: Descent, x0, xlog=False, ylog=True, save=None):
    xs = f.BFGS(x0).xs
    convergence_plot(xs, xlog=xlog, ylog=ylog, save=save)


def number_of_iterations_plot(f: Descent, descent_mode, step_selection_mode, width=5, density=100):
    """
    width (int): the boundaries of the plot is [-width, width]
    density (int): the number of sample points from [-width, width]
    """
    # x = np.linspace(-width, width, density)
    # y = np.linspace(-width, width, density)

    x = np.linspace(-2, 2, density)
    y = np.linspace(-1, 3, density)
    X, Y = np.meshgrid(x, y)

    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]
    for i in range(len(X)):
        for j in range(len(Y)):
            try:
                Z[i][j] = f.descend2D(X[i][j], Y[i][j], descent_mode, step_selection_mode).iterations
            except:
                Z[i][j] = np.inf
            print(X[i][j], Y[i][j], Z[i][j])


    # Plot y=x^2
    x_line = np.linspace(-1.72, 1.72, 100)
    y_line = x_line**2
    plt.plot(x_line, y_line, color='red')

    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    # plt.savefig("images/rosenbrock_iterations.png")


def polynomial_descent_plot():
    coefficients = np.array([3, -5, 4, 1])
    coefficients = np.array([2, -3, -1, 1])
    poly = Polynomial(coefficients)
    p = Descent(poly)
    x0 = np.array([-0.5])
    # x0 = np.array([-1.])
    xs = p.descend(x0, p.steepest, backtracking).xs
    fxs = [p.f(x) for x in xs]

    # Plot the polynomial
    x = np.linspace(-2, 3, 100)
    y = [p.f(np.array([i])) for i in x]
    plt.plot(x, y)

    # Plot the descent
    plt.scatter(xs, fxs, color='orange')
    plt.plot(xs, fxs)

    # for i, txt in enumerate(xs):
        # if i <= 2:
            # plt.text(txt-0.03, fxs[i]+0.4, i, fontsize=8)
        # if i == 3:
            # plt.text(txt-0.05, fxs[i]+0.4, i, fontsize=8)


    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.savefig("images/polynomial_descent.png")
    plt.show()


# himmelblau_plot()
rosenbrock_testing()
# sphere_testing()
# ackley_testing()
# number_of_iterations_plot(ra, ra.newton, wolfe, width=1)
# number_of_iterations_plot(sp, sp.newton, backtracking)

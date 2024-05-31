from step_selection import backtracking, wolfe
from test_functions import rosenbrock, himmelblau, ackley
from descent import Descent
import numpy as np


def main():
    x0 = np.array([0.5, 0.5]).T
    # x0 = np.array([1.2, 1.2]).T

    r = Descent(rosenbrock)

    # Neither the Himmelblau nor Ackley seem to work
    h = Descent(himmelblau)
    a = Descent(ackley)

    sol = r.descend(x0, r.steepest, backtracking)
    # sol = r.descend(x0, r.newton, backtracking)
    # sol = r.descend(x0, r.steepest, wolfe)
    
    print(sol)


main()

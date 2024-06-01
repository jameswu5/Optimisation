cend(p1, h.steepest, backtracking)  # converges to 1st equilibrium
    eq2 = h.descend(p2, h.steepest, backtracking)  # converges to 2rd equilibrium
    eq3 = h.descend(p3, h.steepest, backtracking)  # converges to 3nd equilibrium
    eq4 = h.descend(p4, h.steepest, backtracking)  # converges to 4th equilibrium

    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)
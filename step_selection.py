def line_search(phi, dphi, c1=1e-4, c2=0.9, max_iterations=1000):
    
    alpha_max = 2 # this is an arbitrary choices
    prev_alpha = 0
    alpha = alpha_max / 2 # this is an arbitrary choice

    # Precompute reusable values
    phi_0 = phi(0)
    dphi_0 = dphi(0)

    def zoom(alpha_low, alpha_high):
        for _ in range(max_iterations):
            # assume it's halfway for now. it's incorrect and I will change later.
            alpha_j = (alpha_low + alpha_high) / 2

            phi_j = phi(alpha_j)
            if phi_j > phi_0 + c1 * alpha_j * dphi_0 or phi_j >= phi(alpha_low):
                alpha_high = alpha_j
                continue

            dphi_j = dphi(alpha_j)
            if abs(dphi_j) <= -c2 * dphi_0:
                return alpha_j

            if dphi_j * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_j

        raise Exception()
    
    for i in range(max_iterations):
        if phi(alpha) > phi_0 + c1 * alpha * dphi_0 or (phi(alpha) >= phi(prev_alpha) and i > 0):
            return zoom(prev_alpha, alpha)
        
        if abs(dphi(alpha)) <= -c2 * phi_0:
            return alpha
        
        if dphi(alpha) >= 0:
            return zoom(alpha, prev_alpha)

        prev_alpha, alpha = alpha, (alpha + alpha_max) / 2 # I choose it to be halfway through

    raise Exception()
def newton_raphson(equation_str, initial_guess, tol=1e-6, max_iter=100):
    x = symbols('x')
    f = lambdify(x, equation_str)
    f_prime = lambdify(x, f.diff(x))  # Derivada de la función

    x_n = initial_guess
    for i in range(max_iter):
        x_n1 = x_n - f(x_n) / f_prime(x_n)
        if abs(x_n1 - x_n) < tol:
            return {'root': x_n1, 'convergence': True}
        x_n = x_n1

    return {'root': x_n, 'convergence': False}

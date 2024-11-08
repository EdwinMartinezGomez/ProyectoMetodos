from sympy import symbols, lambdify

def fixed_point_method(equation_str, x0=0, tol=1e-6, max_iter=100):
    """
    Implementación del método de punto fijo
    """
    x = symbols('x')
    g = lambdify(x, equation_str)  # Convierte la ecuación en una función
    
    for i in range(max_iter):
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return {'root': x1, 'convergence': True}
        x0 = x1
    
    return {'root': x1, 'convergence': False}

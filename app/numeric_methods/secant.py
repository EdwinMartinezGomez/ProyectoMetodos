from sympy import symbols, lambdify

def secant_method(equation_str, x0=-1, x1=1, tol=1e-6, max_iter=100):
    """
    Implementación del método de la secante
    """
    x = symbols('x')
    f = lambdify(x, equation_str)  # Convierte la ecuación en una función
    
    for i in range(max_iter):
        f_x0, f_x1 = f(x0), f(x1)
        if abs(f_x1) < tol:
            return {'root': x1, 'convergence': True}
        
        # Actualizar el siguiente valor de x usando la fórmula de la secante
        x_temp = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x0, x1 = x1, x_temp
    
    return {'root': x1, 'convergence': False}

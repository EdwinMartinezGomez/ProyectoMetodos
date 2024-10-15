import numpy as np
from scipy.optimize import broyden1

def broyden_method(system_function, initial_guess, tol=1e-6, maxiter=5000):
    try:
        # Intentar resolver el sistema usando Broyden
        solution = broyden1(system_function, initial_guess, tol=tol, maxiter=maxiter)
        error = verificar_solucion(solution, system_function)
        if error < tol:
            return f"La solución con el método de Broyden es: {solution} con un error de: {error}"
        else:
            return "El método de Broyden no convergió a una solución aceptable."
    except Exception as e:
        return f"Error durante la ejecución del método de Broyden: {e}"

def verificar_solucion(solution, system_function):
    if isinstance(solution, np.ndarray):
        residual = system_function(solution)
        return np.linalg.norm(residual)
    else:
        return np.inf

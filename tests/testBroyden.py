import sympy as sp
import numpy as np

# Define las ecuaciones
eq1 = sp.sympify('x**2 + y**2 - 4')
eq2 = sp.sympify('x**2 - y - 1')

# Define las variables
variables = [sp.Symbol('x'), sp.Symbol('y')]

# Define la función F
def F(x):
    subs = {var: val for var, val in zip(variables, x)}
    return np.array([eq1.evalf(subs=subs), eq2.evalf(subs=subs)], dtype=float)

# Calcula la matriz Jacobiana
def compute_jacobian(x):
    J = []
    for eq in [eq1, eq2]:
        row = []
        for var in variables:
            derivative = sp.diff(eq, var)
            derivative_func = sp.lambdify(variables, derivative, modules=['numpy'])
            row.append(derivative_func(*x))
        J.append(row)
    return np.array(J, dtype=float)

# Implementación simplificada del método de Broyden
def broyden_method(F, J_initial, x0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    J = J_initial.copy()
    for i in range(max_iter):
        try:
            delta = np.linalg.solve(J, -F(x))
        except np.linalg.LinAlgError:
            print("Jacobian singular")
            return None, False

        x_new = x + delta
        F_new = F(x_new)
        error = np.linalg.norm(delta, ord=np.inf)
        print(f"Iter {i+1}: x = {x_new}, F(x) = {F_new}, error = {error}")

        if error < tol:
            return x_new, True

        y = F_new - F(x)
        delta = delta.reshape(-1,1)
        y = y.reshape(-1,1)
        denom = np.dot(delta.T, delta)[0,0]
        if denom == 0:
            print("Denominator zero")
            return None, False
        J += ((y - J @ delta) @ delta.T) / denom
        x = x_new

    return x, False

# Estimación inicial
initial_guess = [1.0, 0.0]

# Calcula J_initial
J_initial = compute_jacobian(initial_guess)

# Ejecuta el método
solution, converged = broyden_method(F, J_initial, initial_guess)

if converged:
    print(f"Converged to: {solution}")
else:
    print("Did not converge")

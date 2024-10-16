import sympy as sp

import sympy as sp

class Raphson:
    def __init__(self, f_expr, x0, tol=1e-9, max_iter=100):
        self.f_expr = f_expr  # Función en forma simbólica
        self.x0 = x0  # Valor inicial
        self.tol = tol  # Tolerancia
        self.max_iter = max_iter  # Máximo de iteraciones

    def calcular_raiz(self):
        x = sp.Symbol('x')  # Definir el símbolo
        df_expr = sp.diff(self.f_expr, x)  # Derivar la función

        f = sp.lambdify(x, self.f_expr)  # Convertir la expresión a función evaluable
        df = sp.lambdify(x, df_expr)  # Convertir la derivada a función evaluable

        error = float('inf')
        iter_count = 0
        x0 = self.x0  # Inicializar x0

        while error > self.tol and iter_count < self.max_iter:
            x1 = x0 - f(x0) / df(x0)
            error = abs((x1 - x0) / x1) * 100
            x0 = x1
            iter_count += 1

            print(f"Iteración {iter_count}: x = {x1}, Error = {error:.6f}%")

        if iter_count == self.max_iter:
            print("El método no convergió en el número máximo de iteraciones.")
        else:
            print(f"Raíz aproximada encontrada en x = {x1} con un error de {error:.6f}% después de {iter_count} iteraciones.")
        return x1

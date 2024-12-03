import sympy as sp
import numpy as np

# Definir la variable
x = sp.Symbol('x')

# Definir la ecuación
eq_str = "\sqrt{x^3}-\frac{1}{2}\cdot x+\frac{2\cdot x^2}{7}-4"
#\sqrt{x**3} - \frac{1}{2}x + \frac{2x^2}{7} - 4

derivada = "\left(4-\frac{1}{2\cdot x}-\frac{2\cdot x^2}{7}\right)^{\frac{2}{3}}"
# Mapear funciones y constantes
allowed_funcs = {
    'E': sp.E,
    'e': sp.E,
    'ℯ': sp.E,
    'exp': sp.exp,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'log': sp.log,
    'sqrt': sp.sqrt,
    'abs': sp.Abs,
    'pi': sp.pi
}

# Parsear la ecuación
expr = sp.sympify(eq_str, locals={'x': x, **allowed_funcs})
print("Expresión:", expr)

# Calcular la derivada
derivative = sp.diff(expr, x)
print("Derivada:", derivative)

# Convertir a función evaluable
fprime = sp.lambdify(x, derivative, modules=['numpy'])

# Evaluar la derivada en puntos de prueba
print("f'(-1.0) =", fprime(-1.0))
print("f'(1.0) =", fprime(1.0))

import sympy as sp

def newton_raphson(f_expr, x0, tol=0.000000001, max_iter=100):
    x = sp.Symbol('x')  # Definir el símbolo
    df_expr = sp.diff(f_expr, x)  # Derivar la función
    
    f = sp.lambdify(x, f_expr)  # Convertir la expresión a función evaluable
    df = sp.lambdify(x, df_expr)  # Convertir la derivada a función evaluable
    
    error = float('inf')  
    iter_count = 0  

    while error > tol and iter_count < max_iter:
        x1 = x0 - f(x0) / df(x0)
        error = abs((x1 - x0) / x1) * 100  
        x0 = x1  
        iter_count += 1  

        print(f"Iteración {iter_count}: x = {x1}, Error = {error:.6f}%")
    
    if iter_count == max_iter:
        print("El método no convergió en el número máximo de iteraciones.")
    else:
        print(f"Raíz aproximada encontrada en x = {x1} con un error de {error:.6f}% después de {iter_count} iteraciones.")
    return x1

# Solicitar los datos por consola
func_input = input("Ingrese la función f(x): ")

# Convertir la entrada en una expresión de sympy
x = sp.Symbol('x')
f_expr = sp.sympify(func_input)

# Solicitar los otros datos por consola
x0 = float(input("Ingrese el valor inicial (x0): "))
tol = float(input("Ingrese la tolerancia (tol): "))
max_iter = int(input("Ingrese el número máximo de iteraciones (max_iter): "))

# Ejecutar el método de Newton-Raphson
raiz = newton_raphson(f_expr, x0, tol, max_iter)


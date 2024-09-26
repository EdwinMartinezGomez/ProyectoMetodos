import math

# Definir la función f(x)
def f(x):
    return (1/3)*x**3 + (2*x)/3 - 4*x**2 + 3

# Definir la derivada f'(x)
def df(x):  
    return (x**2)+(2)/(3)-8*(x)

# Método de Newton-Raphson
def newton_raphson(x0, tol=0.000000001, max_iter=100):
    error = float('inf')  # Inicializamos el error en infinito
    iter_count = 0  # Contador de iteraciones

    while error > tol and iter_count < max_iter:
        x1 = x0 - f(x0) / df(x0)  # Aplicamos la fórmula de Newton-Raphson
        error = abs((x1 - x0) / x1) * 100  # Calculamos el error porcentual
        x0 = x1  # Actualizamos x0 para la siguiente iteración
        iter_count += 1  # Incrementamos el contador

        print(f"Iteración {iter_count}: x = {x1}, Error = {error:.6f}%")
    
    if iter_count == max_iter:
        print("El método no convergió en el número máximo de iteraciones.")
    else:
        print(f"Raíz aproximada encontrada en x = {x1} con un error de {error:.6f}% después de {iter_count} iteraciones.")
    return x1

# Valor inicial
x0 = 8


# Llamamos a la función de Newton-Raphson
raiz = newton_raphson(x0)
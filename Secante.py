import math

# Definir la función f(x)
def f(x):
    return math.sin(x) + (x**2)/4 - 2 

# Método de la secante
def secante(x0, x1, tol=1e-9, max_iter=100):
    error = float('inf')  # Inicializamos el error en infinito
    iter_count = 0  # Contador de iteraciones
    
    while error > tol and iter_count < max_iter:
        # Aplicamos la fórmula de la secante
        try:
            f_x0 = f(x0)
            f_x1 = f(x1)
            x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        except ZeroDivisionError:
            print("División por cero al calcular la secante.")
            return None
        
        error = abs((x2 - x1) / x2) * 100  # Calculamos el error porcentual
        x0, x1 = x1, x2  # Actualizamos los valores de x0 y x1
        iter_count += 1  # Incrementamos el contador de iteraciones

        print(f"Iteración {iter_count}: x = {x2}, Error = {error:.6f}%")
    
    if iter_count == max_iter:
        print("El método no convergió en el número máximo de iteraciones.")
    else:
        print(f"Raíz aproximada encontrada en x = {x2} con un error de {error:.6f}% después de {iter_count} iteraciones.")
    
    return x2

# Valores iniciales
x0 = -10 # Primer punto
x1 = -5  # Segundo punto

# Llamamos a la función de la secante
raiz = secante(x0, x1)

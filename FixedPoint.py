import math


def punto_fijo(g, x0, tolerancia, iteraciones_max, ):
    """
    Encuentra la raíz de la ecuación f(x) = 0 utilizando el método del punto fijo.
    
    Args:
        g: Función de iteración g(x).
        x0: Aproximación inicial.
        tolerancia: Tolerancia para la convergencia.
        iteraciones_max: Número máximo de iteraciones permitidas.
        
    Returns:
        Aproximación de la raíz de la ecuación.
    """
    iteracion = 0
    while iteracion < iteraciones_max:
        x1 = g(x0)
        if abs(x1 - x0) < tolerancia:
            return x1, iteracion
        x0 = x1
        iteracion += 1
    
    ##print("El método no convergió después de", iteraciones_max, "iteraciones.")
    return None, iteracion

# Ejemplo de función de iteración g(x) = sqrt(10 / x)
def g(x):
    return math.sqrt(100 / x)

# Parámetros
x0 = 2.0  # Aproximación inicial
tolerancia = 1e-6  # Tolerancia para la convergencia
iteraciones_max = 10000  # Número máximo de iteraciones

# Llamada al método del punto fijo
raiz, iteracion = punto_fijo(g, x0, tolerancia, iteraciones_max)
if raiz is  None:
    raiz="error"
    def result ():
        data = {'Root' : raiz, 'Iterations' : iteracion }
        return data
    print("La raíz aproximada es:", raiz)
else:
    def result ():
        data = {'Root' : raiz, 'Iterations' : iteracion }
        return data



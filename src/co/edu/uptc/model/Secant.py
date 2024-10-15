import src.co.edu.uptc.model.Secant as Secant
import re
import numpy as np
import math

# Mapeo de funciones matemáticas
function_mapping = {
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'asin': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'sinh': 'np.sinh',
    'cosh': 'np.cosh',
    'tanh': 'np.tanh',
    'asinh': 'np.arcsinh',
    'acosh': 'np.arccosh',
    'atanh': 'np.arctanh',
    'log': 'np.log',
    'log10': 'np.log10',
    'exp': 'np.exp',
    'sqrt': 'np.sqrt'
}

# Transformar funciones de entrada a sintaxis compatible
def transform_function(user_input):
    for key, value in function_mapping.items():
        user_input = re.sub(r'\b' + key + r'\b', value, user_input)
    return user_input

# Evaluación de la función matemática
def f(x, functionMath):
    try:
        return eval(functionMath, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error al evaluar functionMath: {e}")
        return None

# Implementación del método Secante
def secantMethod(functionMath, p0, p1, tol, n):
    result = ""
    i = 1
    while i <= n:
        f_p0 = f(p0, functionMath)
        f_p1 = f(p1, functionMath)
        if f_p1 - f_p0 == 0:
            result += "División por cero. El método falla.\n"
            return result
        
        p = p1 - (f_p1 * (p1 - p0)) / (f_p1 - f_p0)
        
        result += f"Iteración {i}: p0 = {p0}, p1 = {p1}, p = {p}\n"
        
        if abs(p - p1) < tol:
            result += f"La raíz aproximada es {p} después de {i} iteraciones.\n"
            return result
        
        p0 = p1
        p1 = p
        i += 1
    
    result += "El método no convergió en el número máximo de iteraciones.\n"
    return result

# Menú principal para elegir el método numérico
def menu():
    while True:
        print("Seleccione el método numérico:")
        print("1. Punto Fijo")
        print("2. Bisección")
        print("3. Newton-Raphson")
        print("4. Secante")
        print("5. Salir")
        
        choice = input("Ingrese su elección (1-5): ")
        
        if choice == '1':
            # Implementación del Punto Fijo
            pass
        
        elif choice == '2':
            # Implementación de la Bisección
            pass
        
        elif choice == '3':
            # Implementación de Newton-Raphson
            pass
        
        elif choice == '4':
            # Implementación del método Secante
            functionMath = input("Ingrese la función matemática (por ejemplo, 'sin(x)'): ")
            functionMath = transform_function(functionMath)
            p0 = float(input("Ingrese el valor inicial p0: "))
            p1 = float(input("Ingrese el valor inicial p1: "))
            tol = float(input("Ingrese la tolerancia: "))
            n = int(input("Ingrese el número máximo de iteraciones: "))
            print(secantMethod(functionMath, p0, p1, tol, n))
        
        elif choice == '5':
            print("Saliendo del programa.")
            break
        
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

# Llamada al menú
menu()

import re
import numpy as np
import sympy as sp
import src.co.edu.uptc.model.FixedPoint as FixedPoint
import src.co.edu.uptc.model.Bisection as Bisection
import src.co.edu.uptc.model.Secant as Secant
import src.co.edu.uptc.model.Broyden as Broyden
from src.co.edu.uptc.model.Raphson import Raphson

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

def transform_function(user_input):
    for key, value in function_mapping.items():
        user_input = re.sub(r'\b' + key + r'\b', value, user_input)
    return user_input

def menu():
    while True:
        print("Seleccione el método numérico:")
        print("1. Punto Fijo")
        print("2. Bisección")
        print("3. Newton-Raphson")
        print("4. Secante")
        print("5. Broyden")
        print("6. Salir")

        choice = input("Ingrese su elección (1-6): ")

        try:
            if choice == '1':
                functionMath = input("Ingrese la función matemática (por ejemplo, 'sin(x)'): ")
                functionTransformed = input("Ingrese la función transformada (por ejemplo, '1-x**3'): ")
                functionMath = transform_function(functionMath)
                functionTransformed = transform_function(functionTransformed)
                p0 = float(input("Ingrese el valor inicial p0: "))
                print(FixedPoint.fixedPoint(functionMath, functionTransformed, p0, 0.0001, 1000, 1))

            elif choice == '2':
                functionMath = input("Ingrese la función matemática (por ejemplo, 'sin(x)'): ")
                functionMath = transform_function(functionMath)
                a = float(input("Ingrese el valor inicial a: "))
                b = float(input("Ingrese el valor inicial b: "))
                tol = float(input("Ingrese la tolerancia: "))
                n = int(input("Ingrese el número máximo de iteraciones: "))
                print(Bisection.bisection(functionMath, a, b, tol, n))

            elif choice == '3':
                function_str = input("Ingrese la función matemática en términos de 'x' (por ejemplo, 'exp(x)/2 + 1/3*x**2 - 1/x**3 - 2'): ")
                x = sp.Symbol('x')
                functionMath = sp.sympify(function_str)

                p0 = float(input("Ingrese el valor inicial p0: "))
                tol = float(input("Ingrese la tolerancia: "))
                n = int(input("Ingrese el número máximo de iteraciones: "))

                raphson_instance = Raphson(functionMath, p0, tol, n)
                raiz = raphson_instance.calcular_raiz()
                print(f"La raíz aproximada es: {raiz}")

            elif choice == '4':
                functionMath = input("Ingrese la función matemática (por ejemplo, 'x**3-2*x-5'): ")
                functionMath = transform_function(functionMath)
                p0 = float(input("Ingrese el valor inicial p0: "))
                p1 = float(input("Ingrese el valor inicial p1: "))
                tol = float(input("Ingrese la tolerancia: "))
                n = int(input("Ingrese el número máximo de iteraciones: "))
                print(Secant.secantMethod(functionMath, p0, p1, tol, n))

            elif choice == '5':
                print("Ingrese las ecuaciones del sistema (ejemplo para el sistema eléctrico):")
                initial_guess = np.array([float(x) for x in input("Ingrese las condiciones iniciales separadas por comas (ejemplo: 1.0, 1.0, 1.0): ").split(",")])
                print(Broyden.broyden_method(Broyden.sistema_electrico, initial_guess))

            elif choice == '6':
                print("Saliendo del programa.")
                break

            else:
                print("Opción no válida. Por favor, intente de nuevo.")

        except ValueError as ve:
            print(f"Error de valor: {ve}. Asegúrate de ingresar números válidos.")
        except ZeroDivisionError as zde:
            print(f"Error: División por cero detectada: {zde}. Intenta con un valor inicial diferente.")
        except sp.SympifyError as se:
            print(f"Error en la función matemática: {se}. Verifica la función ingresada.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}. Por favor, revisa la entrada.")

if __name__ == "__main__":
    menu()

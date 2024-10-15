import sympy as sp
from Raphson import Raphson  # Asegúrate de tener la clase Raphson en un archivo separado

class View:
    def solicitar_datos(self):
        # Solicitar la función al usuario
        func_input = input("Ingrese la función f(x): ")

        # Convertir la entrada en una expresión de sympy
        x = sp.Symbol('x')
        f_expr = sp.sympify(func_input)

        # Solicitar los otros datos por consola
        x0 = float(input("Ingrese el valor inicial (x0): "))
        tol = float(input("Ingrese la tolerancia (tol): "))
        max_iter = int(input("Ingrese el número máximo de iteraciones (max_iter): "))

        # Crear una instancia de la clase Raphson con los datos ingresados
        raphson = Raphson(f_expr, x0, tol, max_iter)

        # Ejecutar el método de Newton-Raphson
        raphson.calcular_raiz()

# Uso del programa
if __name__ == "__main__":
    view = View()
    view.solicitar_datos()

import numpy as np
import math

def f(x, functionMath):
    try:
        return eval(functionMath, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error evaluating functionMath: {e}")
        return None

def bisection(functionMath, a, b, tol, n):
    try:
        a = float(a)
        b = float(b)
        tol = float(tol)
        n = int(n)
    except ValueError as ve:
        return f"Error: Se ingresaron valores no numéricos: {ve}"

    if tol <= 0:
        return "Error: La tolerancia debe ser un número positivo."

    if n <= 0:
        return "Error: El número de iteraciones debe ser mayor a 0."

    try:
        fa = f(a, functionMath)
        fb = f(b, functionMath)
        if fa is None or fb is None:
            return "Error en la evaluación de la función matemática."
        if fa * fb >= 0:
            return "El método de bisección no es aplicable, ya que no hay cambio de signo en el intervalo [a, b]."
    except Exception as e:
        return f"Error al evaluar la función: {e}"

    i = 1
    result = ""
    try:
        while i <= n:
            c = (a + b) / 2
            fc = f(c, functionMath)
            if fc is None:
                return "Error en la evaluación de la función matemática en el punto medio."
            if fc == 0 or (b - a) / 2 < tol:
                result += f"La raíz es {c} después de {i} iteraciones\n"
                return result
            i += 1
            if fc * fa < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            result += f"Iteración {i - 1}: {c}\n"
    except Exception as e:
        return f"Error durante la ejecución del método: {e}"

    result += f"El método no converge después de {n} iteraciones. Último intervalo: [{a}, {b}]\n"
    return result

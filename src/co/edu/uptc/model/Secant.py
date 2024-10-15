import numpy as np
import math

def f(x, functionMath):
    try:
        return eval(functionMath, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error al evaluar functionMath: {e}")
        return None

def secantMethod(functionMath, p0, p1, tol, n):
    result = ""
    for i in range(1, n + 1):
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
        
        p0, p1 = p1, p
    
    result += "El método no convergió en el número máximo de iteraciones.\n"
    return result

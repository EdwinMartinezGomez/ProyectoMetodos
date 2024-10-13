import numpy as np
import math

def f(x, functionMath):
    try:
        return eval(functionMath, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error evaluating functionMath: {e}")
        return None

def g(x, functionTransformed):
    try:
        return eval(functionTransformed, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error evaluating functionTransformed: {e}")
        return None

def fixedPoint(functionMath, functionTransformed, p0, tol, n,i):
    result=""
    while i <= n:
        p = g(p0, functionTransformed)
        if abs(p0 - p) < tol:
            result=result+ f"La raiz es {p} después de {i} iteraciones\n"
            #print("La raiz es", p, "después de", i, "iteraciones")
            return result
        i =i+ 1
        p0 = p
        result = result + f"Iteración {i - 1}: {p0}\n"
    if i > n:
        result=result+"El método diverge\n"
        #print("El método diverge")
        return result

def inputFixedPoint():
    global functionMath, funtionTransformed 
    functionMath = input("Ingrese la función f(x): ")
    funtionTransformed = input("Ingrese la función g(x): ")
    p0 = float(input("Ingrese el valor del punto inicial: "))
    tol = float(0.0001)
    n = int(1000)

import numpy as np

functionMath = ""
funtionTransformed = ""

def f(x):
    return eval(functionMath, {"x": x, "np": np})

def g(x):
    return eval(funtionTransformed, {"x": x, "np": np})

def fixedPoint(functionMath, funtionTransformed, p0, tol, n, i):
    while i <= n:
        p = g(p0)
        if abs(p - p0) < tol:
            print("El punto fijo es", p, "después de", i, "iteraciones")
            break
        i = i + 1
        p0 = p
        print("Iteración", i - 1, ": ", p0)
    if i >= n:
        print("El método diverge")

def inputFixedPoint():
    global functionMath, funtionTransformed 
    functionMath = input("Ingrese la función f(x): ")
    funtionTransformed = input("Ingrese la función g(x): ")
    p0 = float(input("Ingrese el valor del punto inicial: "))
    tol = float(0.0001)
    n = int(1000)
    i = 1
    fixedPoint(functionMath, funtionTransformed, p0, tol, n, i)

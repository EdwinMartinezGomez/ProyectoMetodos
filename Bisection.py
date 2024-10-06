import math

functionMath = "x**3 - x - 2"  # Esta es la ecuación  para que no se meta por teclado

def f(x):
    return eval(functionMath, {"x": x, "math": math})

def bisection(a, b, tol, n):
    if f(a) * f(b) >= 0:
        print("El método de bisección no es aplicable.")
        return None

    i = 1
    while i <= n:
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            print("La raíz es ", c, " después de ", i, " iteraciones")
            return c
        i += 1
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        print("Iteración ", i - 1, ": ", c)

    print("El método no converge después de ", n, " iteraciones.")
    return None

def inputBisection():
    print(f"La ecuación utilizada es: {functionMath}")
    a = float(input("Ingrese el valor de a: "))
    b = float(input("Ingrese el valor de b: "))
    tol = float(0.0001)
    n = int(1000)
    bisection(a, b, tol, n)
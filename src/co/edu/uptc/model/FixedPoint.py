import math

functionMath=""
funtionTransformed=""
def f(x):
    return eval(functionMath, {"x": x, "math": math})

def g(x):
    return eval(funtionTransformed, {"x": x, "math": math})

def fixedPoint(functionMath, funtionTransformed, p0, tol, n, i):
    while i<=n:
        p=g(p0)
        if abs(p-p0)<tol:
            print("El punto fijo es ",p," despues de ",i," iteraciones")
            break
        i=i+1
        p0=p
        print("iteracion ",i-1,": ",p0)
    if i>=n:
        print("diverge")

def inputFixedPoint():
    functionMath=input("Ingrese la funcion f(x) ")
    funtionTransformed=input("Ingrese la funcion g(x) ")
    p0=float(input("Ingrese el valor del punto inicial "))
    tol=float(0.0001)
    n=int(1000)
    i=1
    fixedPoint(functionMath, funtionTransformed, p0, tol, n, i)
from app.routes import parse_equation

# Prueba con la ecuación LaTeX
equation = "\sqrt{x^3}+\frac{1}{2x}+\frac{x^2}{\frac{7}{2}}-4"
# (4-((1)/(2 x))-((2 x^(2))/(7)))^(((2)/(3)))
print("\n=== Prueba del Parser de Ecuaciones ===")
print(f"Probando ecuación: {equation}")

try:
    f = parse_equation(equation)
    print("\n✓ Éxito!")
    print(f"Evaluación en x=0: {f(0)}")
except ValueError as e:
    print(f"\n✗ Error: {e}")
except Exception as e:
    print(f"\n✗ Error inesperado: {e}")

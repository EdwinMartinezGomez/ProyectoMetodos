from app.routes import parse_equation

# Prueba con la ecuación LaTeX
equation = "\frac{e^{4x}}{\frac{2}{3}}+x^2-\frac{3}{4}x-2"

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

from app.routes import preprocess_equation, parse_equation

def are_parentheses_balanced(expr):
    stack = []
    for char in expr:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return not stack

if __name__ == "__main__":
    # Utilizar una cadena raw para evitar problemas con secuencias de escape
    test_equation = r"\frac{e^{4x}}{\frac{2}{3}}+x^2-\frac{3}{4}x-2"
    try:
        processed = preprocess_equation(test_equation)
        print(f"Procesada: {processed}")
        balanced = are_parentheses_balanced(processed)
        print("Paréntesis balanceados:", balanced)
        
        # Parsear la ecuación con SymPy
        f = parse_equation(test_equation)
        print("La función se ha parseado correctamente.")
        
        # Evaluar la función en un punto específico, por ejemplo x = 1.0
        x_val = 1.0
        y_val = f(x_val)
        print(f"f({x_val}) = {y_val}")
        
    except Exception as e:
        print(f"Error al preprocesar o parsear la ecuación: {e}")

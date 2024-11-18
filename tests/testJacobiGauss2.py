# test2.py
from app.routes import parse_system

def main():
    equations = [
        '6*x + 3*z = 1/2',
        '(1/4)*x - 4*y - (4/3)*z = 1/2',
        '6*z + (1/3)*w = 1/2',
        '2*x + 6*w = 1/2'
    ]
    variables = ['x', 'y', 'z', 'w']
    try:
        A, b = parse_system(equations, variables)
        print("Matriz A:", A)
        print("Vector b:", b)
    except ValueError as ve:
        print("Error:", ve)

if __name__ == "__main__":
    main()

# test2.py
from app.routes import parse_system

def main():
    equations = [
        '3*x - (3/2)*z + w = 1',
        '-2/3*x + 3*y - 3*z = 1',
        '3*z - 3/2*w = 1',
        '-1/2*x + w = 1'
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

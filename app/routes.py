from flask import Blueprint, request, jsonify, render_template
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from scipy import optimize
import sympy as sp
import re
import logging

main = Blueprint('main', __name__)

# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@main.route('/')
def index():
    return render_template('calculator.html')

def preprocess_equation(equation):
    """Preprocess equation to handle LaTeX-like notation and common input formats."""
    eq = equation.strip().replace(' ', '')
    
    # Reemplazar notación de fracción LaTeX por división estándar
    eq = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'(\1)/(\2)', eq)
    
    # Manejar la notación e^{...} y convertirla a exp(...)
    eq = re.sub(r'e\^\{([^}]+)\}', r'exp(\1)', eq)
    
    # Reemplazar ^ por ** para exponentes
    eq = re.sub(r'\^', '**', eq)
    
    # Añadir * para multiplicaciones implícitas donde sea necesario
    eq = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', eq)
    eq = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', eq)
    eq = re.sub(r'\)([a-zA-Z(])', r')*\1', eq)
    
    # Reemplazar 'e' por 'E' cuando 'e' se usa como constante
    eq = re.sub(r'\be\b', 'E', eq)
    
    # Eliminar comandos de LaTeX como \left y \right si aún quedan
    eq = re.sub(r'\\left|\\right', '', eq)
    
    return eq

def parse_equations(equations, variables):
    """
    Convierte una lista de ecuaciones en formato de texto a una matriz de coeficientes y un vector b.
    """
    A = []
    b = []
    for eq in equations:
        # Suponiendo que las ecuaciones están en la forma "ax + by + cz = d"
        lhs, rhs = eq.split('=')
        rhs = float(rhs.strip())
        b.append(rhs)
        
        coeffs = []
        for var in variables:
            # Buscar el coeficiente de cada variable usando expresiones regulares
            pattern = r'([+-]?\s*\d*\.?\d*)\s*' + re.escape(var)
            match = re.search(pattern, lhs)
            if match:
                coeff_str = match.group(1).replace(' ', '')
                if coeff_str in ['', '+']:
                    coeff = 1.0
                elif coeff_str == '-':
                    coeff = -1.0
                else:
                    coeff = float(coeff_str)
            else:
                coeff = 0.0
            coeffs.append(coeff)
        A.append(coeffs)
    return A, b

def parse_equation(equation_str):
    """
    Analiza y valida la ecuación proporcionada por el usuario.
    """
    try:
        if not equation_str:
            raise ValueError("La ecuación no puede estar vacía.")

        # Eliminar 'Math.'
        equation_str = equation_str.replace('Math.', '')

        processed_eq = preprocess_equation(equation_str)
        x = sp.Symbol('x')

        # Funciones y constantes permitidas usando SymPy
        allowed_funcs = {
            'E': sp.E,             # Constante matemática e
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi
        }

        expr = sp.sympify(processed_eq, locals={'x': x, **allowed_funcs})
        f = sp.lambdify(x, expr, modules=['numpy'])

        return f
    except sp.SympifyError as e:
        raise ValueError(f"La ecuación ingresada no es válida: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error al procesar la ecuación: {str(e)}")

def parse_derivative_equation(equation_str):
    """
    Analiza y valida la derivada de la ecuación proporcionada por el usuario.
    """
    try:
        if not equation_str:
            raise ValueError("La ecuación no puede estar vacía.")

        processed_eq = preprocess_equation(equation_str)
        x = sp.Symbol('x')

        # Funciones y constantes permitidas usando SymPy
        allowed_funcs = {
            'E': sp.E,             # Constante matemática e
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi
        }

        expr = sp.sympify(processed_eq, locals={'x': x, **allowed_funcs})
        derivative_expr = sp.diff(expr, x)
        fprime = sp.lambdify(x, derivative_expr, modules=['numpy'])

        # Prueba de evaluación de la derivada (excluyendo x=0.0)
        test_points = [-1.0, 1.0]  # Excluye x=0.0
        for test_x in test_points:
            try:
                result = fprime(test_x)
                if not np.isfinite(result):
                    raise ValueError(f"La derivada de la función no es finita en x={test_x}.")
            except Exception as e:
                raise ValueError(f"La derivada de la función no es válida en x={test_x}: {str(e)}")

        return fprime
    except sp.SympifyError as e:
        raise ValueError(f"La derivada de la ecuación ingresada no es válida: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error al procesar la derivada de la ecuación: {str(e)}")
def parse_g_function(g_str):
    try:
        if not g_str:
            raise ValueError("La función g(x) no puede estar vacía.")
        
        processed_g = preprocess_equation(g_str)
        x = sp.Symbol('x')
        
        allowed_funcs = {
            'E': sp.E,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi
        }
        
        expr = sp.sympify(processed_g, locals={'x': x, **allowed_funcs})
        g = sp.lambdify(x, expr, modules=['numpy'])
        
        # Prueba de evaluación
        for test_x in [-1.0, 0.5, 1.0, 1.5, 2.0]:  # Añadido más puntos de prueba
            try:
                result = g(test_x)
                if not np.isfinite(result):
                    raise ValueError(f"La función g(x) no es finita en x={test_x}.")
            except Exception as e:
                raise ValueError(f"La función g(x) no es válida en x={test_x}: {str(e)}")
        
        return g
    except sp.SympifyError as e:
        raise ValueError(f"La función g(x) ingresada no es válida: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error al procesar la función g(x): {str(e)}")

def find_valid_interval(f, start=-10, end=10, num_points=1000):
    """
    Encuentra un intervalo válido [a, b] donde f(a) y f(b) tengan signos opuestos.
    """
    x_vals = np.linspace(start, end, num_points)
    f_vals = np.array([f(x) for x in x_vals])

    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    if sign_changes.size > 0:
        index = sign_changes[0]
        return x_vals[index], x_vals[index + 1]
    else:
        raise ValueError("No se encontró un intervalo válido donde la función cambie de signo.")

def bisection_method(f, a, b, max_iter, iteration_history, tol=1e-6):
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")

    converged = False
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a) / 2.0  # Calcula el error como la mitad del intervalo
        iteration_history.append({
            'iteration': i,  # Añadido
            'x': round(float(c), 6),  # Cambiado a 3
            'fx': round(float(fc), 6),  # Cambiado a 3
            'error': round(float(error), 6)  # Cambiado a 3
        })
        if abs(fc) < tol or error < tol:
            converged = True
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, converged, i
def newton_raphson(f, fprime, x0, max_iter=100, tol=1e-6):
    iteration_history = []
    x_prev = x0
    for i in range(1, max_iter + 1):
        try:
            fx = f(x_prev)
            fpx = fprime(x_prev)
            if fpx == 0:
                raise ZeroDivisionError(f"La derivada es cero en x = {x_prev}.")
            x_next = x_prev - fx / fpx
            error = abs(x_next - x_prev)
            iteration_history.append({
                'iteration': i,  # Añadido
                'x': round(float(x_next), 6),  # Cambiado a 3
                'fx': round(float(fx), 6),  # Cambiado a 3
                'error': round(float(error), 6)  # Cambiado a 3
            })
            if error < tol:
                return x_next, True, i, iteration_history
            x_prev = x_next
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método Newton-Raphson: {str(e)}")
            return None, False, i, iteration_history
    return x_prev, False, max_iter, iteration_history

def fixed_point_method(g, x0, max_iter, iteration_history, tol=1e-6):
    x_prev = x0
    converged = False
    for i in range(1, max_iter + 1):
        try:
            x_next = g(x_prev)
            fx = x_next - x_prev  # diferencia entre iteraciones
            error = abs(x_next - x_prev)
            if not np.isfinite(x_next):
                raise ValueError(f"El método de Punto Fijo produjo un valor no finito en la iteración {i}.")
            iteration_history.append({
                'iteration': i,  # Añadido
                'x': round(float(x_next), 6),  # Cambiado a 3
                'fx': round(float(fx), 6),  # Cambiado a 3
                'error': round(float(error), 6)  # Cambiado a 3
            })
            logger.debug(f"Iteración {i}: x = {x_next}, f(x) = {fx}, error = {error}")
            if error < tol:
                converged = True
                break
            x_prev = x_next
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método de Punto Fijo: {str(e)}")
            return None, False, i, iteration_history

    return x_next, converged, i, iteration_history

def jacobi_method(A, b, x0, max_iter, tol=1e-6, iteration_history=None):
    n = len(A)
    x = np.array(x0, dtype=float)
    x_new = np.zeros_like(x)
    converged = False

    for i in range(1, max_iter + 1):
        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            if A[j][j] == 0:
                raise ZeroDivisionError(f"División por cero detectada en la fila {j}.")
            x_new[j] = (b[j] - s) / A[j][j]
        
        # Calcular el error como la norma infinita
        error = np.linalg.norm(x_new - x, ord=np.inf)
        
        # Almacenar el historial
        if iteration_history is not None:
            iteration_history.append({
                'iteration': i,
                'x': [round(float(val), 6) for val in x_new],
                'error': round(float(error), 6)
            })
        
        if error < tol:
            converged = True
            break
        x = x_new.copy()
    
    return x.tolist(), converged, i

def gauss_seidel_method(A, b, x0, max_iter, tol=1e-6, iteration_history=None):
    """
    Implementación del método de Gauss-Seidel para resolver sistemas de ecuaciones lineales A x = b.
    
    Parámetros:
    - A: Matriz de coeficientes.
    - b: Vector de términos independientes.
    - x0: Vector de estimación inicial.
    - max_iter: Número máximo de iteraciones.
    - tol: Tolerancia para la convergencia.
    - iteration_history: Lista para almacenar el historial de iteraciones.
    
    Retorna:
    - x: Solución encontrada.
    - converged: Booleano indicando si la convergencia fue exitosa.
    - iterations: Número de iteraciones realizadas.
    """
    n = len(A)
    x = np.array(x0, dtype=float)
    converged = False

    for i in range(1, max_iter + 1):
        x_old = x.copy()
        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            if A[j][j] == 0:
                raise ZeroDivisionError(f"División por cero detectada en la fila {j}.")
            x[j] = (b[j] - s) / A[j][j]
        
        # Calcular el error como la norma infinita
        error = np.linalg.norm(x - x_old, ord=np.inf)
        
        # Almacenar el historial
        if iteration_history is not None:
            iteration_history.append({
                'iteration': i,
                'x': [round(float(val), 6) for val in x],
                'error': round(float(error), 6)
            })
        
        if error < tol:
            converged = True
            break
    
    return x.tolist(), converged, i
def secant_method(f, x0, x1, max_iter, tol=1e-6):
    iteration_history = []
    for i in range(1, max_iter + 1):
        try:
            fx0 = f(x0)
            fx1 = f(x1)
            if fx1 - fx0 == 0:
                raise ZeroDivisionError("División por cero en el método Secante.")
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            error = abs(x2 - x1)
            iteration_history.append({
                'iteration': i,  # Añadido
                'x': round(float(x2), 6), 
                'fx': round(float(f(x2)), 6),
                'error': round(float(error), 6)  
            })
            if error < tol:
                return x2, True, i, iteration_history
            x0, x1 = x1, x2
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método Secante: {str(e)}")
            return None, False, i, iteration_history
    return x1, False, max_iter, iteration_history

def parse_system(equations, variables):
    """
    Analiza y valida un sistema de ecuaciones lineales.
    
    Parámetros:
    - equations: Lista de cadenas representando las ecuaciones.
    - variables: Lista de variables en el sistema.
    
    Retorna:
    - A: Matriz de coeficientes.
    - b: Vector de términos independientes.
    """
    try:
        num_eq = len(equations)
        num_var = len(variables)
        
        if num_eq != num_var:
            raise ValueError("El número de ecuaciones debe ser igual al número de variables.")
        
        A = np.zeros((num_eq, num_var), dtype=float)
        b = np.zeros(num_eq, dtype=float)
        
        for i, eq in enumerate(equations):
            if '=' not in eq:
                raise ValueError(f"La ecuación {i+1} no contiene un signo de igual.")
            
            lhs, rhs = eq.split('=')
            
            # Convertir el lado derecho en un número usando SymPy para manejar fracciones
            rhs_expr = sp.sympify(rhs.strip())
            b[i] = float(rhs_expr.evalf())
            
            # Parsear el lado izquierdo
            expr = sp.sympify(preprocess_equation(lhs))
            for j, var in enumerate(variables):
                coeff = expr.coeff(sp.Symbol(var))
                A[i][j] = float(coeff)
        
        # Logs para verificar las matrices
        logger.info(f"Matriz A: {A}")
        logger.info(f"Vector b: {b}")
        
        return A.tolist(), b.tolist()
    except Exception as e:
        raise ValueError(f"Error al parsear el sistema de ecuaciones: {str(e)}")
@main.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        if not data:
            logger.error("No se recibió ningún dato.")
            return jsonify({'error': 'No se recibió ningún dato.'}), 400

        # Validación básica de campos comunes
        common_required_keys = ['method', 'iterations']
        for key in common_required_keys:
            if key not in data:
                logger.error(f"Falta el campo: {key}")
                return jsonify({'error': f'Falta el campo: {key}'}), 400

        method = data['method']
        try:
            max_iter = int(data['iterations'])
        except ValueError:
            logger.error('El número de iteraciones debe ser un entero.')
            return jsonify({'error': 'El número de iteraciones debe ser un entero.'}), 400

        # Validaciones adicionales
        if not (1 <= max_iter <= 1000):
            logger.error('El número de iteraciones debe ser entre 1 y 1000.')
            return jsonify({'error': 'El número de iteraciones debe ser entre 1 y 1000.'}), 400

        if method not in ['bisection', 'newton', 'secant', 'fixed_point', 'jacobi', 'gauss_seidel']:
            logger.error('Método no válido.')
            return jsonify({'error': 'Método no válido.'}), 400

        # Determinar si se trata de una sola ecuación o un sistema
        is_system = method in ['jacobi', 'gauss_seidel']

        if is_system:
            # Validar campos para sistemas
            if 'equations' not in data or 'variables' not in data:
                logger.error('Faltan los campos: equations y/o variables')
                return jsonify({'error': 'Faltan los campos: equations y/o variables'}), 400

            equations = data['equations']  # Lista de ecuaciones
            variables = data['variables']  # Lista de variables

            if not isinstance(equations, list) or not isinstance(variables, list):
                logger.error('Las ecuaciones y variables deben ser listas.')
                return jsonify({'error': 'Las ecuaciones y variables deben ser listas.'}), 400

            if len(equations) == 0 or len(variables) == 0:
                logger.error('Las ecuaciones y variables no pueden estar vacías.')
                return jsonify({'error': 'Las ecuaciones y variables no pueden estar vacías.'}), 400

            try:
                A, b = parse_system(equations, variables)
            except ValueError as ve:
                logger.error(str(ve))
                return jsonify({'error': str(ve)}), 400

            # Inicializar la estimación inicial
            if 'initial_guess' not in data:
                logger.error('Falta el campo: initial_guess')
                return jsonify({'error': 'Falta el campo: initial_guess'}), 400

            x0 = data['initial_guess']
            if not isinstance(x0, list) or len(x0) != len(variables):
                logger.error('initial_guess debe ser una lista con el mismo número de elementos que variables.')
                return jsonify({'error': 'initial_guess debe ser una lista con el mismo número de elementos que variables.'}), 400

            try:
                x0 = [float(val) for val in x0]
            except ValueError:
                logger.error('Todos los elementos de initial_guess deben ser números.')
                return jsonify({'error': 'Todos los elementos de initial_guess deben ser números.'}), 400

        else:
            # Métodos para una sola ecuación
            if 'equation' not in data:
                logger.error('Falta el campo: equation')
                return jsonify({'error': 'Falta el campo: equation'}), 400

            equation = data['equation']
            try:
                f = parse_equation(equation)
            except ValueError as ve:
                logger.error(str(ve))
                return jsonify({'error': str(ve)}), 400

            iteration_history = []

            if method == 'bisection':
                # Validación para Bisección
                if 'a' not in data or 'b' not in data:
                    logger.error(f'Faltan los campos: a y/o b para el método {method}')
                    return jsonify({'error': f'Faltan los campos: a y/o b para el método {method}'}), 400
                try:
                    a = float(data['a'])
                    b = float(data['b'])
                except ValueError:
                    logger.error('Los límites del intervalo deben ser números válidos.')
                    return jsonify({'error': 'Los límites del intervalo deben ser números válidos.'}), 400

                if a >= b:
                    logger.error('El límite inferior (a) debe ser menor que el superior (b).')
                    return jsonify({'error': 'El límite inferior (a) debe ser menor que el superior (b).'}), 400

                # Validar si f(a) y f(b) cambian de signo
                fa = f(a)
                fb = f(b)
                if fa * fb >= 0:
                    # Intentar encontrar un intervalo válido automáticamente
                    try:
                        a, b = find_valid_interval(f)
                        logger.info(f"Intervalo ajustado automáticamente a: a={a}, b={b}")
                    except Exception as e:
                        logger.error(f"Error al validar el intervalo: {str(e)}")
                        return jsonify({'error': str(e)}), 400

            elif method == 'secant':
                # Validación para Secante
                if 'x0' not in data or 'x1' not in data:
                    logger.error('Faltan los campos: x0 y/o x1')
                    return jsonify({'error': 'Faltan los campos: x0 y/o x1'}), 400
                try:
                    x0_sec = float(data['x0'])
                    x1_sec = float(data['x1'])
                except ValueError:
                    logger.error('Las estimaciones iniciales x0 y x1 deben ser números válidos.')
                    return jsonify({'error': 'Las estimaciones iniciales x0 y x1 deben ser números válidos.'}), 400

                if x0_sec == x1_sec:
                    logger.error('Las estimaciones iniciales x0 y x1 deben ser distintas.')
                    return jsonify({'error': 'Las estimaciones iniciales x0 y x1 deben ser distintas.'}), 400

            elif method in ['newton', 'fixed_point']:
                # Validación para Newton y Punto Fijo
                if 'initial_guess' not in data:
                    logger.error('Falta el campo: initial_guess')
                    return jsonify({'error': 'Falta el campo: initial_guess'}), 400
                try:
                    initial_guess = float(data['initial_guess'])
                except ValueError:
                    logger.error('El punto inicial debe ser un número válido.')
                    return jsonify({'error': 'El punto inicial debe ser un número válido.'}), 400

                if method == 'fixed_point':
                    # Validar g(x) para Punto Fijo
                    if 'gFunction' not in data:
                        logger.error('Falta la función g(x).')
                        return jsonify({'error': 'Falta la función g(x).'}), 400
                    gFunction = data['gFunction']
                    try:
                        g = parse_g_function(gFunction)
                    except ValueError as ve:
                        logger.error(str(ve))
                        return jsonify({'error': str(ve)}), 400

        # Implementación de métodos
        iteration_history = []
        root = None
        converged = False
        iterations = 0

        if is_system:
            A_matrix = np.array(A, dtype=float)
            b_vector = np.array(b, dtype=float)
            if method == 'jacobi':
                try:
                    root, converged, iterations = jacobi_method(
                        A_matrix, b_vector, x0, max_iter, tol=1e-6, iteration_history=iteration_history
                    )
                except Exception as e:
                    logger.error(f"Error en el método Jacobi: {str(e)}")
                    return jsonify({'error': f"Error en el método Jacobi: {str(e)}"}), 400

            elif method == 'gauss_seidel':
                try:
                    root, converged, iterations = gauss_seidel_method(
                        A_matrix, b_vector, x0, max_iter, tol=1e-6, iteration_history=iteration_history
                    )
                except Exception as e:
                    logger.error(f"Error en el método Gauss-Seidel: {str(e)}")
                    return jsonify({'error': f"Error en el método Gauss-Seidel: {str(e)}"}), 400

            # Generar gráfica para sistemas de ecuaciones
            try:
                # Extraer nombres de variables y sus valores en cada iteración
                variables = data['variables']
                iterations_numbers = [entry['iteration'] for entry in iteration_history]
                plot_traces = []

                for var_idx, var in enumerate(variables):
                    var_values = [entry['x'][var_idx] for entry in iteration_history]
                    trace = go.Scatter(
                        x=iterations_numbers,
                        y=var_values,
                        mode='lines+markers',
                        name=var
                    )
                    plot_traces.append(trace)

                layout = go.Layout(
                    title='Convergencia de Variables por Iteración',
                    xaxis=dict(title='Iteración'),
                    yaxis=dict(title='Valor de la Variable'),
                    plot_bgcolor='#f0f0f0',
                    paper_bgcolor='#ffffff',
                    hovermode='closest'
                )

                fig = go.Figure(data=plot_traces, layout=layout)
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                response = {
                    'solution': {var: round(float(root[i]), 6) for i, var in enumerate(variables)},
                    'converged': converged,
                    'iterations': iterations,
                    'iteration_history': iteration_history,
                    'plot_json': graphJSON
                }

            except Exception as e:
                logger.error(f"Error al generar la gráfica para el sistema: {str(e)}")
                return jsonify({'error': f"Error al generar la gráfica para el sistema: {str(e)}"}), 400

        else:
            # Métodos para una sola ecuación
            if method == 'bisection':
                try:
                    root, converged, iterations = bisection_method(f, a, b, max_iter, iteration_history)
                except Exception as e:
                    logger.error(f"Error en el método Bisección: {str(e)}")
                    return jsonify({'error': f"Error en el método Bisección: {str(e)}"}), 400
            elif method == 'newton':
                try:
                    # Método de Newton-Raphson
                    fprime = parse_derivative_equation(equation)
                    root, converged, iterations, iteration_history = newton_raphson(f, fprime, initial_guess, max_iter, tol=1e-6)
                    if not converged:
                        logger.error('El método Newton-Raphson no convergió.')
                        return jsonify({'error': 'El método Newton-Raphson no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Newton-Raphson: {str(e)}")
                    return jsonify({'error': f"Error en el método Newton-Raphson: {str(e)}"}), 400
            elif method == 'secant':
                try:
                    # Método de la Secante
                    root, converged, iterations, iteration_history = secant_method(f, x0_sec, x1_sec, max_iter, tol=1e-6)
                    if not converged:
                        logger.error('El método Secante no convergió.')
                        return jsonify({'error': 'El método Secante no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Secante: {str(e)}")
                    return jsonify({'error': f"Error en el método Secante: {str(e)}"}), 400
            elif method == 'fixed_point':
                try:
                    # Método de Punto Fijo
                    root, converged, iterations, iteration_history = fixed_point_method(g, initial_guess, max_iter, iteration_history, tol=1e-6)
                    if not converged:
                        logger.error('El método Punto Fijo no convergió.')
                        return jsonify({'error': 'El método Punto Fijo no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Punto Fijo: {str(e)}")
                    return jsonify({'error': f"Error en el método Punto Fijo: {str(e)}"}), 400

            # Preparar la gráfica para una sola ecuación
            try:
                # Definir el rango de la gráfica
                plot_a, plot_b = -10, 10  # Valores por defecto, pueden ser ajustados
                if method == 'bisection':
                    plot_a, plot_b = a, b
                elif method in ['newton', 'fixed_point']:
                    plot_a, plot_b = initial_guess - 10, initial_guess + 10
                elif method == 'secant':
                    plot_a, plot_b = min(x0_sec, x1_sec) - 10, max(x0_sec, x1_sec) + 10

                # Evaluar los valores de y para la gráfica
                x_vals = np.linspace(plot_a, plot_b, 1000)
                try:
                    y_vals = [f(xi) for xi in x_vals]
                except Exception as e:
                    logger.error(f"Error al evaluar la función para la gráfica: {str(e)}")
                    y_vals = [float('nan') for _ in x_vals]

                # Trace de la función principal
                function_trace = go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name='f(x)',
                    line=dict(color='blue'),
                    hoverinfo='none'
                )

                # Trace de rastro inicial (vacío)
                trace_rastro_initial = go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='Rastro',
                    line=dict(color='orange', dash='dash'),
                    hoverinfo='none'
                )

                # Inicializar data_traces con las trazas base
                data_traces = [function_trace, trace_rastro_initial]

                # Inicializar los frames y las listas de iteración
                frames = []
                iteration_x = []
                iteration_y = []

                # Agregar las iteraciones a los traces para la animación
                if iteration_history:
                    for idx, iter_data in enumerate(iteration_history):
                        if method == 'fixed_point':
                            current_x = iter_data['x']
                            current_y = g(current_x)
                        else:
                            current_x = iter_data['x']
                            current_y = f(current_x)

                        iteration_x.append(current_x)
                        iteration_y.append(current_y)

                        # Trace de la aproximación actual
                        approx_trace = go.Scatter(
                            x=[current_x],
                            y=[current_y],
                            mode='markers+text',
                            name=f'Iteración {idx + 1}',
                            marker=dict(color='red', size=10),
                            text=[f"{idx + 1}"],
                            textposition='top center',
                            hoverinfo='x+y'
                        )

                        # Trace de rastro acumulado
                        trace_rastro = go.Scatter(
                            x=iteration_x.copy(),
                            y=iteration_y.copy(),
                            mode='lines',
                            name='Rastro',
                            line=dict(color='orange', dash='dash'),
                            hoverinfo='none'
                        )

                        # Crear frame con ambos traces
                        frames.append(go.Frame(
                            data=[approx_trace, trace_rastro],
                            name=str(idx)
                        ))

                # Agregar trace de la raíz final
                if root is not None:
                    try:
                        y_root = f(root)
                    except Exception as e:
                        logger.error(f"Error al evaluar f(root): {str(e)}")
                        y_root = float('nan')

                    root_trace = go.Scatter(
                        x=[root],
                        y=[y_root],
                        mode='markers',
                        name='Raíz',
                        marker=dict(color='green', size=12, symbol='star'),
                        hoverinfo='x+y'
                    )
                    data_traces.append(root_trace)

                # Manejo específico para el método Secante (agregar trazas de x0 y x1)
                if not is_system and method == 'secant':
                    try:
                        y0_sec = f(x0_sec)
                        y1_sec = f(x1_sec)
                    except Exception as e:
                        logger.error(f"Error al evaluar f(x0_sec) o f(x1_sec): {str(e)}")
                        y0_sec = float('nan')
                        y1_sec = float('nan')

                    x0_trace = go.Scatter(
                        x=[x0_sec],
                        y=[y0_sec],
                        mode='markers',
                        name='x₀',
                        marker=dict(color='purple', size=10, symbol='circle'),
                        hoverinfo='x+y'
                    )

                    x1_trace = go.Scatter(
                        x=[x1_sec],
                        y=[y1_sec],
                        mode='markers',
                        name='x₁',
                        marker=dict(color='brown', size=10, symbol='circle'),
                        hoverinfo='x+y'
                    )

                    data_traces.extend([x0_trace, x1_trace])

                # Generar frames para la animación
                if iteration_history:
                    for idx, iter_data in enumerate(iteration_history):
                        if method == 'fixed_point':
                            current_x = iter_data['x']
                            current_y = g(current_x)
                        else:
                            current_x = iter_data['x']
                            current_y = f(current_x)

                        iteration_x.append(current_x)
                        iteration_y.append(current_y)

                        # Trace de la aproximación actual
                        approx_trace = go.Scatter(
                            x=[current_x],
                            y=[current_y],
                            mode='markers+text',
                            name=f'Iteración {idx + 1}',
                            marker=dict(color='red', size=10),
                            text=[f"{idx + 1}"],
                            textposition='top center',
                            hoverinfo='x+y'
                        )

                        # Trace de rastro acumulado
                        trace_rastro = go.Scatter(
                            x=iteration_x.copy(),
                            y=iteration_y.copy(),
                            mode='lines',
                            name='Rastro',
                            line=dict(color='orange', dash='dash'),
                            hoverinfo='none'
                        )

                        # Crear frame con ambos traces
                        frames.append(go.Frame(
                            data=[approx_trace, trace_rastro],
                            name=str(idx)
                        ))

                # Definir el layout de la gráfica con animación
                layout = go.Layout(
                    title=dict(
                        text='Gráfica de la Ecuación con Proceso de Iteración',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=24)
                    ),
                    xaxis=dict(title='x', range=[plot_a, plot_b]),
                    yaxis=dict(title='f(x)', range=[min(y_vals) - 10, max(y_vals) + 10]),
                    plot_bgcolor='#f0f0f0',
                    paper_bgcolor='#ffffff',
                    hovermode='closest',
                    updatemenus=[
                        {
                            "buttons": [
                                {
                                    "args": [None, {"frame": {"duration": 700, "redraw": True},
                                                    "fromcurrent": True}],
                                    "label": "Play",
                                    "method": "animate"
                                },
                                {
                                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                      "mode": "immediate",
                                                      "transition": {"duration": 0}}],
                                    "label": "Pause",
                                    "method": "animate"
                                }
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 87},
                            "showactive": False,
                            "type": "buttons",
                            "x": 0.1,
                            "xanchor": "right",
                            "y": 0,
                            "yanchor": "top"
                        }
                    ],
                    sliders=[
                        {
                            "steps": [
                                {
                                    "args": [
                                        [str(k)],
                                        {"frame": {"duration": 700, "redraw": True},
                                         "mode": "immediate"}
                                    ],
                                    "label": f"Iteración {k + 1}",
                                    "method": "animate"
                                } for k in range(len(frames))
                            ],
                            "transition": {"duration": 0},
                            "x": 0.1,
                            "y": 0,
                            "currentvalue": {"font": {"size": 16}, "prefix": "Iteración: ", "visible": True, "xanchor": "right"},
                            "len": 0.9
                        }
                    ]
                )

                fig = go.Figure(data=data_traces, layout=layout, frames=frames)
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                # Preparar la respuesta
                response = {
                    'root': round(float(root), 6) if root is not None else 'N/A',
                    'converged': converged,
                    'iterations': iterations,
                    'iteration_history': iteration_history,
                    'plot_json': graphJSON
                }

            except Exception as e:
                logger.error(f"Error al generar la gráfica para la ecuación: {str(e)}")
                return jsonify({'error': f"Error al generar la gráfica para la ecuación: {str(e)}"}), 400

        # Retornar la respuesta
        return jsonify(response)

    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error inesperado durante el cálculo.")
        return jsonify({'error': 'Ocurrió un error inesperado durante el cálculo.'}), 500


@main.route('/find_valid_interval', methods=['POST'])
def find_valid_interval_route():
    try:
        data = request.get_json()
        if not data:
            logger.error("No se recibió ningún dato.")
            return jsonify({'error': 'No se recibió ningún dato.'}), 400

        if 'equation' not in data:
            logger.error('Falta el campo: equation')
            return jsonify({'error': 'Falta el campo: equation'}), 400

        equation = data['equation']
        f = parse_equation(equation)

        # Buscar intervalo válido
        a, b = find_valid_interval(f)

        return jsonify({'a': a, 'b': b})
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error inesperado al buscar el intervalo.")
        return jsonify({'error': 'Ocurrió un error inesperado al buscar el intervalo.'}), 500

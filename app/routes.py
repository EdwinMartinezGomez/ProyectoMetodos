from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from app.numeric_methods import Simpson as simpson
from app.numeric_methods import Trapecio as trapecio
from app.numeric_methods import GaussSeidel as gauss
from app.numeric_methods import Jacobi as jacobi
from app.numeric_methods import bisection
from app.numeric_methods import Broyden as broyden
from app.numeric_methods import fixed_point
from app.numeric_methods import newton_raphson
from app.numeric_methods import secant
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import sympy as sp
import re
import logging

# Definir las transformaciones incluyendo 'convert_xor'
transformations = (
    standard_transformations +
    (implicit_multiplication_application,) +
    (convert_xor,)
)

main = Blueprint('main', __name__)

# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@main.route('/')
def index():
    return render_template('calculator.html')

def replace_integrals(eq):
    """
    Reemplaza expresiones de integrales definidas en LaTeX por la sintaxis de SymPy.
    Por ejemplo, \int_{a}^{b} f(x) dx -> Integral(f(x), (x, a, b))
    """
    # Patrón para detectar \int_{a}^{b} f(x) dx
    integral_pattern = r'\\int_\{([^}]+)\}\^{([^}]+)\}\s*([^\\]+?)\s*dx'
    
    # Función de reemplazo
    def integral_replacer(match):
        lower_limit = match.group(1).strip()
        upper_limit = match.group(2).strip()
        integrand = match.group(3).strip()
        return f'Integral({integrand}, (x, {lower_limit}, {upper_limit}))'
    
    # Reemplazar todas las integrales encontradas
    eq = re.sub(integral_pattern, integral_replacer, eq)
    
    return eq

def replace_fractions(eq):
    """
    Reemplaza todas las instancias de \frac{a}{b} por (a)/(b).
    Maneja múltiples y fracciones anidadas.
    """
    while '\\frac' in eq:
        frac_start = eq.find('\\frac')
        first_brace = eq.find('{', frac_start)
        if first_brace == -1:
            logger.error("Fracción malformada: No se encontró '{' después de '\\frac'.")
            raise ValueError("Fracción malformada: No se encontró '{' después de '\\frac'.")

        # Función para extraer el contenido dentro de las llaves
        def extract_brace_content(s, start):
            if s[start] != '{':
                logger.error(f"Se esperaba '{{' en la posición {start}.")
                return None, start
            stack = 1
            content = []
            for i in range(start + 1, len(s)):
                if s[i] == '{':
                    stack += 1
                    content.append(s[i])
                elif s[i] == '}':
                    stack -= 1
                    if stack == 0:
                        return ''.join(content), i
                    else:
                        content.append(s[i])
                else:
                    content.append(s[i])
            logger.error("Fracción malformada: No se encontró '}' correspondiente.")
            return None, start  # No matching closing brace

        # Extraer el numerador
        numerator, num_end = extract_brace_content(eq, first_brace)
        if numerator is None:
            raise ValueError("Fracción malformada: No se pudo extraer el numerador.")

        # Encontrar la primera '{' después del numerador
        denominator_start = eq.find('{', num_end)
        if denominator_start == -1:
            raise ValueError("Fracción malformada: Faltante '{' para el denominador.")

        # Extraer el denominador
        denominator, den_end = extract_brace_content(eq, denominator_start)
        if denominator is None:
            raise ValueError("Fracción malformada: No se pudo extraer el denominador.")

        # Reemplazar \frac{numerador}{denominador} con (numerador)/(denominador)
        frac_full = eq[frac_start:den_end + 1]
        frac_replacement = f'({numerator})/({denominator})'
        eq = eq.replace(frac_full, frac_replacement, 1)
        logger.info(f"Reemplazado '{frac_full}' por '{frac_replacement}'.")

    return eq

def preprocess_equation(equation):
    """
    Preprocesa la ecuación para manejar notación LaTeX, inserción de '*', reemplazo de '^' por '**', y reemplazo de \frac y \int.
    """
    eq = equation.strip()
    logger.info(f"Ecuación original: {eq}")

    # 1. Reemplazar \frac{a}{b} por (a)/(b)
    eq = replace_fractions(eq)
    logger.info(f"Ecuación después de reemplazar fracciones: {eq}")

    # 2. Reemplazar \sqrt{...} por sqrt(...)
    eq = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', eq)
    logger.info(f"Ecuación después de reemplazar '\\sqrt{{...}}': {eq}")

    # 3. Reemplazar otros comandos de LaTeX
    eq = re.sub(r'\\left|\\right', '', eq)
    eq = re.sub(r'\\cdot|\\times', '*', eq)
    eq = re.sub(r'\\div', '/', eq)
    eq = re.sub(r'\\pi', 'pi', eq)
    eq = re.sub(r'\\ln', 'log', eq)
    eq = re.sub(r'\\log', 'log10', eq)
    eq = re.sub(r'\\exp\{([^{}]+)\}', r'exp(\1)', eq)
    eq = re.sub(r'\\sin', 'sin', eq)
    eq = re.sub(r'\\cos', 'cos', eq)
    eq = re.sub(r'\\tan', 'tan', eq)
    logger.info(f"Ecuación después de reemplazar otros comandos de LaTeX: {eq}")

    # 4. Reemplazar integrales
    eq = replace_integrals(eq)
    logger.info(f"Ecuación después de reemplazar integrales: {eq}")

    # 5. Reemplazar '{' y '}' por '(' y ')'
    eq = eq.replace('{', '(').replace('}', ')')
    logger.info(f"Ecuación después de reemplazar '{{}}' por '()': {eq}")

    # 6. Insertar explícitamente '*' entre dígitos y letras o '(' con manejo de espacios
    eq = re.sub(r'(\d)\s*([a-zA-Z(])', r'\1*\2', eq)
    eq = re.sub(r'(\))\s*([a-zA-Z(])', r'\1*\2', eq)
    logger.info(f"Ecuación después de insertar '*': {eq}")

    # 7. Reemplazar '^' por '**' para exponentiación
    eq = eq.replace('^', '**')
    logger.info(f"Ecuación después de reemplazar '^' por '**': {eq}")

    # Validar paréntesis balanceados
    if eq.count('(') != eq.count(')'):
        raise ValueError("Paréntesis desbalanceados en la ecuación.")

    logger.info(f"Ecuación preprocesada: {eq}")
    return eq

def parse_equation(equation_str):
    try:
        if not equation_str:
            raise ValueError("La ecuación no puede estar vacía.")

        equation_str = equation_str.replace('Math.', '')
        processed_eq = preprocess_equation(equation_str)

        x = sp.Symbol('x')

        allowed_funcs = {
            'E': sp.E,
            'e': sp.E,
            'ℯ': sp.E,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'log10': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi,
            'Integral': sp.Integral
        }

        expr = parse_expr(processed_eq, local_dict={'x': x, **allowed_funcs}, transformations=transformations)
        f_original = sp.lambdify(x, expr, modules=['numpy'])

        # Función segura para manejar evaluaciones
        def safe_f(val):
            try:
                result = f_original(val)
                if np.isfinite(result):
                    return result
                return np.inf
            except OverflowError:
                return np.inf
            except Exception:
                return np.inf

        # Vectorizar la función segura sin causar recursión
        f = np.vectorize(safe_f)

        return expr, f

    except Exception as e:
        logger.error(f"Error al procesar la ecuación: {str(e)}")
        raise ValueError(f"Error al procesar la ecuación: {str(e)}")

def parse_derivative_equation(equation_str):
    """
    Analiza y valida la derivada de la ecuación proporcionada por el usuario.
    """
    try:
        if not equation_str:
            raise ValueError("La ecuación no puede estar vacía.")

        processed_eq = preprocess_equation(equation_str)
        logger.info(f"Ecuación preprocesada para derivada: {processed_eq}")

        x = sp.Symbol('x')

        allowed_funcs = {
            'E': sp.E,
            'e': sp.E,
            'ℯ': sp.E,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'log10': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi,
            'Integral': sp.Integral
        }

        expr = parse_expr(processed_eq, local_dict={'x': x, **allowed_funcs}, transformations=transformations)
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
    except Exception as e:
        logger.error(f"Error al procesar la derivada de la ecuación: {str(e)}")
        raise ValueError(f"Error al procesar la derivada de la ecuación: {str(e)}")

def parse_g_function(g_func_str):
    """
    Analiza y valida la función g(x) proporcionada por el usuario.
    """
    try:
        if not g_func_str:
            raise ValueError("La función g(x) no puede estar vacía.")

        processed_g_func = preprocess_equation(g_func_str)
        logger.info(f"Función g(x) preprocesada: {processed_g_func}")

        x = sp.Symbol('x')

        allowed_funcs = {
            'E': sp.E,
            'e': sp.E,
            'ℯ': sp.E,
            'exp': sp.exp,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'log10': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi,
            'Integral': sp.Integral
        }

        expr = parse_expr(processed_g_func, local_dict={'x': x, **allowed_funcs}, transformations=transformations)
        g = sp.lambdify(x, expr, modules=['numpy'])

        return g
    except Exception as e:
        logger.error(f"Error al procesar la función g(x): {str(e)}")
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

def render_integration_plot(f, a, b, n, method, extra_shapes):
    x_vals = np.linspace(a, b, 1000)
    y_vals = f(x_vals)

    function_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(color='blue')
    )
    
    data_traces = [function_trace]

    # Agregar las áreas bajo la curva (trapezoides o parábolas)
    for shape in extra_shapes:
        trace = go.Scatter(
            x=shape['x'],
            y=shape['y'],
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.2)',
            mode='lines',
            line=dict(color='rgba(0, 100, 255, 0.5)'),
            showlegend=False
        )
        data_traces.append(trace)

    layout = go.Layout(
        title=f"Integración usando el Método {method.capitalize()}",
        xaxis=dict(title='x'),
        yaxis=dict(title='f(x)'),
        plot_bgcolor='#f0f0f0'
    )

    fig = go.Figure(data=data_traces, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def evaluate_system(equations, variables):
    def F(x):
        subs = {var: val for var, val in zip(variables, x)}
        return np.array([float(eq.evalf(subs=subs)) for eq in equations], dtype=float)
    return F

def compute_initial_jacobian(equations, variables, x0):
    """
    Calcula la matriz Jacobiana inicial en el punto x0.

    Args:
        equations: Lista de expresiones SymPy que representan las ecuaciones.
        variables: Lista de símbolos SymPy que representan las variables.
        x0: Estimación inicial (lista o array).

    Returns:
        J_initial: Matriz Jacobiana evaluada en x0.
    """
    J = []
    for eq in equations:
        row = []
        for var in variables:
            derivative = sp.diff(eq, var)
            derivative_func = sp.lambdify(variables, derivative, modules=['numpy'])
            row.append(derivative_func(*x0))
        J.append(row)
    return np.array(J, dtype=float)


def parse_system(equations, variables):
    """
    Convierte una lista de ecuaciones en formato de texto a una matriz de coeficientes y un vector b.
    """
    try:
        num_eq = len(equations)
        num_var = len(variables)
        
        if num_eq != num_var:
            raise ValueError("El número de ecuaciones debe ser igual al número de variables.")
        
        A = np.zeros((num_eq, num_var), dtype=float)
        b_vector = np.zeros(num_eq, dtype=float)
        
        for i, eq in enumerate(equations):
            if '=' not in eq:
                raise ValueError(f"La ecuación '{eq}' no contiene un signo de igual '='.")
            
            lhs, rhs = eq.split('=')
            
            # Convertir el lado derecho en un número usando SymPy para manejar fracciones
            rhs_expr = sp.sympify(rhs.strip())
            b_vector[i] = float(rhs_expr.evalf())
            
            # Parsear el lado izquierdo
            expr = sp.sympify(preprocess_equation(lhs))
            for j, var in enumerate(variables):
                coeff = expr.coeff(sp.Symbol(var))
                A[i][j] = float(coeff)
        
        logger.info(f"Matriz A: {A}")
        logger.info(f"Vector b: {b_vector}")
        
        return A.tolist(), b_vector.tolist()
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

        if method not in ['bisection', 'newton', 'secant', 'fixed_point', 'jacobi', 'gauss_seidel', 'broyden', 'trapezoidal', 'simpson']:
            logger.error('Método no válido.')
            return jsonify({'error': 'Método no válido.'}), 400

        # Determinar si se trata de una sola ecuación o un sistema
        is_system = method in ['jacobi', 'gauss_seidel', 'broyden']  # Añadido 'broyden'

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

            # Inicializar el historial de iteraciones
            iteration_history = []

            if method in ['jacobi', 'gauss_seidel']:
                try:
                    A, b = parse_system(equations, variables)
                except ValueError as ve:
                    logger.error(str(ve))
                    return jsonify({'error': str(ve)}), 400

                A_matrix = np.array(A, dtype=float)
                b_vector = np.array(b, dtype=float)

                if method == 'jacobi':
                    try:
                        root, converged, iterations = jacobi.jacobi_method(
                            A_matrix, b_vector, x0, max_iter, tol=1e-6, iteration_history=iteration_history
                        )
                    except Exception as e:
                        logger.error(f"Error en el método Jacobi: {str(e)}")
                        return jsonify({'error': f"Error en el método Jacobi: {str(e)}"}), 400

                elif method == 'gauss_seidel':
                    try:
                        root, converged, iterations = gauss.gauss_seidel_method(
                            A_matrix, b_vector, x0, max_iter, tol=1e-6, iteration_history=iteration_history
                        )
                    except Exception as e:
                        logger.error(f"Error en el método Gauss-Seidel: {str(e)}")
                        return jsonify({'error': f"Error en el método Gauss-Seidel: {str(e)}"}), 400

                # Generar gráfica para sistemas de ecuaciones
                try:
                    plot_json = broyden.render_broyden_plot(iteration_history, variables)
                    response = {
                        'solution': {var: round(float(root[i]), 6) for i, var in enumerate(variables)},
                        'converged': converged,
                        'iterations': iterations,
                        'iteration_history': iteration_history,
                        'plot_json': plot_json
                    }
                except Exception as e:
                    logger.error(f"Error al generar la gráfica para el sistema: {str(e)}")
                    return jsonify({'error': f"Error al generar la gráfica para el sistema: {str(e)}"}), 400

            elif is_system and method == 'broyden':
                try:
                    # Parsear las ecuaciones como expresiones SymPy
                    exprs = []
                    for eq in equations:
                        if '=' not in eq:
                            raise ValueError(f"La ecuación '{eq}' no contiene un signo de igual '='.")
                        lhs, rhs = eq.split('=')
                        expr = sp.sympify(preprocess_equation(lhs)) - sp.sympify(preprocess_equation(rhs))
                        exprs.append(expr)

                    variables_symbols = [sp.Symbol(var) for var in variables]

                    # Definir F(x)
                    F = evaluate_system(exprs, variables_symbols)

                    # Calcular la matriz Jacobiana inicial en x0
                    J_initial = compute_initial_jacobian(exprs, variables_symbols, x0)

                    # Ejecutar el Método de Broyden
                    root, converged, iterations =broyden.broyden_method(F, J_initial, x0, max_iter, tol=1e-6, iteration_history=iteration_history)

                    if not converged:
                        logger.error('El método de Broyden no convergió.')
                        return jsonify({'error': 'El método de Broyden no convergió.'}), 400

                    # Generar gráfica para sistemas de ecuaciones
                    try:
                        plot_json = broyden.render_broyden_plot(iteration_history, variables)
                        response = {
                            'solution': {var: round(float(root[i]), 6) for i, var in enumerate(variables)},
                            'converged': converged,
                            'iterations': iterations,
                            'iteration_history': iteration_history,
                            'plot_json': plot_json
                        }
                    except Exception as e:
                        logger.error(f"Error al generar la gráfica para el sistema: {str(e)}")
                        return jsonify({'error': f"Error al generar la gráfica para el sistema: {str(e)}"}), 400

                except np.linalg.LinAlgError as e:
                    logger.error(f"Error al resolver el sistema lineal: {str(e)}")
                    return jsonify({'error': 'Error al resolver el sistema lineal durante el método de Broyden.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método de Broyden: {str(e)}")
                    return jsonify({'error': f"Error en el método de Broyden: {str(e)}"}), 400

        elif method in ['trapezoidal', 'simpson']:
            # Validar los parámetros para integración
            if 'equation' not in data or 'a' not in data or 'b' not in data or 'n' not in data:
                logger.error('Faltan los campos: equation, a, b y/o n para el método de integración.')
                return jsonify({'error': 'Faltan los campos: equation, a, b y/o n para el método de integración.'}), 400

            equation = data['equation']
            a = float(data['a'])
            b = float(data['b'])
            n = int(data['n'])

            # Validar la ecuación y convertirla a función y expresión simbólica
            try:
                expr, f = parse_equation(equation)
            except ValueError as ve:
                logger.error(str(ve))
                return jsonify({'error': str(ve)}), 400

            # Calcular el área y preparar la gráfica
            try:
                if method == 'trapezoidal':
                    area, trapezoids = trapecio.trapezoidal_method(f, a, b, n)
                    estimatedError = trapecio.calculate_trapezoidal_error(expr, a, b, n)  # Calcular el error
                    plot_json = render_integration_plot(f, a, b, n, 'trapezoidal', trapezoids)
                elif method == 'simpson':
                    area, parabolas = simpson.simpson_method(f, a, b, n)
                    estimatedError = simpson.calculate_simpson_error(expr, a, b, n)  # Calcular el error

                    plot_json = render_integration_plot(f, a, b, n, 'simpson', parabolas)

                response = {
                    'method': method,
                    'area': round(area, 6),
                    'estimatedError': round(estimatedError, 6),  # Incluir el error en la respuesta
                    'plot_json': plot_json
                }
                return jsonify(response)
            except Exception as e:
                return jsonify({'error': f"Error en el cálculo: {str(e)}"}), 400

        else:
            # Métodos para una sola ecuación
            if 'equation' not in data:
                logger.error('Falta el campo: equation')
                return jsonify({'error': 'Falta el campo: equation'}), 400

            equation = data['equation']
            try:
                expr, f = parse_equation(equation)
            except ValueError as ve:
                logger.error(str(ve))
                return jsonify({'error': str(ve)}), 400

            iteration_history = []

            if method == 'bisection':
                try:
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

                except Exception as e:
                    logger.error(f"Error en la preparación para Bisección: {str(e)}")
                    return jsonify({'error': f"Error en la preparación para Bisección: {str(e)}"}), 400

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
            if method in ['jacobi', 'gauss_seidel']:
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
                    # Función mejorada para crear layouts
                    def create_enhanced_layout(title, x_label='Iteración', y_label='Valor de la Variable'):
                        return go.Layout(
                            title=dict(
                                text=title,
                                x=0.5,
                                xanchor='center',
                                font=dict(
                                    size=24,
                                    family='Arial, sans-serif',
                                    color='#2c3e50'
                                )
                            ),
                            xaxis=dict(
                                title=dict(
                                    text=x_label,
                                    font=dict(size=16, family='Arial, sans-serif')
                                ),
                                gridcolor='#e0e0e0',
                                zerolinecolor='#2c3e50',
                                zerolinewidth=2
                            ),
                            yaxis=dict(
                                title=dict(
                                    text=y_label,
                                    font=dict(size=16, family='Arial, sans-serif')
                                ),
                                gridcolor='#e0e0e0',
                                zerolinecolor='#2c3e50',
                                zerolinewidth=2
                            ),
                            plot_bgcolor='#ffffff',
                            paper_bgcolor='#ffffff',
                            hovermode='closest',
                            showlegend=True,
                            legend=dict(
                                x=1.05,
                                y=1,
                                bgcolor='rgba(255, 255, 255, 0.9)',
                                bordercolor='#2c3e50'
                            ),
                            margin=dict(l=80, r=80, t=100, b=80)
                        )

                    variables = data['variables']
                    iterations_numbers = list(range(1, len(iteration_history) + 1))
                    plot_traces = []

                    # Paleta de colores mejorada
                    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c']

                    for var_idx, var in enumerate(variables):
                        var_values = [entry['x'][var_idx] for entry in iteration_history]
                        trace = go.Scatter(
                            x=iterations_numbers,
                            y=var_values,
                            mode='lines+markers',
                            name=var,
                            line=dict(
                                color=colors[var_idx % len(colors)],
                                width=3,
                                dash='solid'
                            ),
                            marker=dict(
                                size=8,
                                symbol='circle',
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate=f"{var}: %{{y:.6f}}<br>Iteración: %{{x}}<extra></extra>"
                        )
                        plot_traces.append(trace)

                        # Trace de tendencia
                        trace_trend = go.Scatter(
                            x=iterations_numbers,
                            y=var_values,
                            mode='lines',
                            name=f'{var} (tendencia)',
                            line=dict(
                                color=colors[var_idx % len(colors)],
                                width=1,
                                dash='dot'
                            ),
                            opacity=0.3,
                            showlegend=False
                        )
                        plot_traces.append(trace_trend)

                    layout = create_enhanced_layout(
                        'Convergencia de Variables por Iteración',
                        'Iteración',
                        'Valor de la Variable'
                    )

                    # Agregar animación de aparición gradual
                    frames = []
                    for i in range(1, len(iterations_numbers) + 1):
                        frame_data = []
                        for var_idx in range(len(variables)):
                            var_values = [entry['x'][var_idx] for entry in iteration_history[:i]]
                            frame_data.append(
                                go.Scatter(
                                    x=iterations_numbers[:i],
                                    y=var_values,
                                    mode='lines+markers',
                                    name=variables[var_idx],
                                    line=dict(
                                        color=colors[var_idx % len(colors)],
                                        width=3,
                                        dash='solid'
                                    ),
                                    marker=dict(
                                        size=8,
                                        symbol='circle',
                                        line=dict(width=2, color='white')
                                    ),
                                    hovertemplate=f"{variables[var_idx]}: %{{y:.6f}}<br>Iteración: %{{x}}<extra></extra>"
                                )
                            )
                            # Agregar trace de tendencia en cada frame
                            frame_data.append(
                                go.Scatter(
                                    x=iterations_numbers[:i],
                                    y=var_values,
                                    mode='lines',
                                    name=f'{variables[var_idx]} (tendencia)',
                                    line=dict(
                                        color=colors[var_idx % len(colors)],
                                        width=1,
                                        dash='dot'
                                    ),
                                    opacity=0.3,
                                    showlegend=False
                                )
                            )
                        frames.append(go.Frame(
                            data=frame_data,
                            name=f'frame{i}'
                        ))

                    # Agregar controles de animación
                    layout.update(
                        updatemenus=[{
                            "buttons": [
                                {
                                    "args": [None, {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True
                                    }],
                                    "label": "▶ Play",
                                    "method": "animate"
                                },
                                {
                                    "args": [[None], {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}
                                    }],
                                    "label": "⏸ Pause",
                                    "method": "animate"
                                }
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 10},
                            "showactive": False,
                            "type": "buttons",
                            "x": 0.1,
                            "y": 1.1,
                            "xanchor": "right",
                            "yanchor": "top"
                        }],
                        sliders=[{
                            "active": 0,
                            "yanchor": "top",
                            "xanchor": "left",
                            "currentvalue": {
                                "font": {"size": 16},
                                "prefix": "Iteración: ",
                                "visible": True,
                                "xanchor": "right"
                            },
                            "transition": {"duration": 300, "easing": "cubic-in-out"},
                            "pad": {"b": 10, "t": 50},
                            "len": 0.9,
                            "x": 0.1,
                            "y": 0,
                            "steps": [
                                {
                                    "args": [
                                        [str(k)],
                                        {"frame": {"duration": 700, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 300}}
                                    ],
                                    "label": f"Iteración {k + 1}",
                                    "method": "animate"
                                } for k in range(len(frames))
                            ]
                        }]
                    )

                    fig = go.Figure(data=plot_traces, layout=layout, frames=frames)

                    response = {
                        'solution': {var: round(float(root[i]), 6) for i, var in enumerate(variables)},
                        'converged': converged,
                        'iterations': iterations,
                        'iteration_history': iteration_history,
                        'plot_json': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    }

                except Exception as e:
                    logger.error(f"Error al generar la gráfica para el sistema: {str(e)}")
                    return jsonify({'error': f"Error al generar la gráfica para el sistema: {str(e)}"}), 400

        else:
            # Métodos para una sola ecuación
            if method == 'bisection':
                try:
                    root, converged, iterations = bisection.bisection_method(f, a, b, max_iter, iteration_history)
                except Exception as e:
                    logger.error(f"Error en el método Bisección: {str(e)}")
                    return jsonify({'error': f"Error en el método Bisección: {str(e)}"}), 400
            elif method == 'newton':
                try:
                    # Método de Newton-Raphson
                    fprime = parse_derivative_equation(equation)
                    root, converged, iterations, iteration_history = newton_raphson.newton_raphsonMethod(f, fprime, initial_guess, max_iter, tol=1e-6)
                    if not converged:
                        logger.error('El método Newton-Raphson no convergió.')
                        return jsonify({'error': 'El método Newton-Raphson no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Newton-Raphson: {str(e)}")
                    return jsonify({'error': f"Error en el método Newton-Raphson: {str(e)}"}), 400
            elif method == 'secant':
                try:
                    # Método de la Secante
                    root, converged, iterations, iteration_history = secant.secant_method(f, x0_sec, x1_sec, max_iter, tol=1e-6)
                    if not converged:
                        logger.error('El método Secante no convergió.')
                        return jsonify({'error': 'El método Secante no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Secante: {str(e)}")
                    return jsonify({'error': f"Error en el método Secante: {str(e)}"}), 400
            elif method == 'fixed_point':
                try:
                    # Método de Punto Fijo
                    root, converged, iterations, iteration_history = fixed_point.fixed_point_method(g, initial_guess, max_iter, iteration_history, tol=1e-6)
                    if not converged:
                        logger.error('El método Punto Fijo no convergió.')
                        return jsonify({'error': 'El método Punto Fijo no convergió.'}), 400
                except Exception as e:
                    logger.error(f"Error en el método Punto Fijo: {str(e)}")
                    return jsonify({'error': f"Error en el método Punto Fijo: {str(e)}"}), 400

            # Preparar la gráfica para una sola ecuación
            try:
                # Función mejorada para crear layouts
                def create_enhanced_layout(title, x_label='x', y_label='f(x)'):
                    return go.Layout(
                        title=dict(
                            text=title,
                            x=0.5,
                            xanchor='center',
                            font=dict(size=24, family='Arial, sans-serif', color='#2c3e50')
                        ),
                        xaxis=dict(
                            title=dict(
                                text=x_label,
                                font=dict(size=16, family='Arial, sans-serif')
                            ),
                            gridcolor='#e0e0e0',
                            zerolinecolor='#2c3e50',
                            zerolinewidth=2
                        ),
                        yaxis=dict(
                            title=dict(
                                text=y_label,
                                font=dict(size=16, family='Arial, sans-serif')
                            ),
                            gridcolor='#e0e0e0',
                            zerolinecolor='#2c3e50',
                            zerolinewidth=2
                        ),
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        hovermode='closest',
                        showlegend=True,
                        legend=dict(
                            x=1.05,
                            y=1,
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            bordercolor='#2c3e50'
                        ),
                        margin=dict(l=80, r=80, t=100, b=80)
                    )

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
                            current_y = f(current_x)  # Usar f(x) para la gráfica
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
                            hovertemplate=f"Iteración {idx + 1}: x = {current_x:.6f}, f(x) = {current_y:.6f}<extra></extra>"
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

                # Definir el layout de la gráfica con animación
                layout = create_enhanced_layout('Visualización del Método Numérico')

                # Agregar controles de animación mejorados
                layout.update(
                    updatemenus=[{
                        "buttons": [
                            {
                                "args": [None, {
                                    "frame": {"duration": 700, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300, "easing": "cubic-in-out"}
                                }],
                                "label": "▶ Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }],
                                "label": "⏸ Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 10},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "y": 1.1,
                        "xanchor": "right",
                        "yanchor": "top"
                    }],
                    sliders=[{
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 16},
                            "prefix": "Iteración: ",
                            "visible": True,
                            "xanchor": "right"
                        },
                        "transition": {"duration": 300, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [
                                    [str(k)],
                                    {"frame": {"duration": 700, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 300}}
                                ],
                                "label": f"Iteración {k + 1}",
                                "method": "animate"
                            } for k in range(len(frames))
                        ]
                    }]
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
        expr, f = parse_equation(equation)

        # Buscar intervalo válido
        a, b = find_valid_interval(f)

        return jsonify({'a': a, 'b': b})
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Error inesperado al buscar el intervalo.")
        return jsonify({'error': 'Ocurrió un error inesperado al buscar el intervalo.'}), 500

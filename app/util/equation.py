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
# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
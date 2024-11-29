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
    
def validate(data):
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
    
    return None, None

def validate_system(data):
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
    return None, None
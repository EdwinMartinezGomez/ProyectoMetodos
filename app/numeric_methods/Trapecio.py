from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import sympy as sp
import re
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def trapezoidal_method(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    area = (h / 2) * np.sum(y[:-1] + y[1:])

    trapezoids = []
    for i in range(n):
        # Definir los vértices del trapezoide
        trapezoid_x = [x[i], x[i+1], x[i+1], x[i], x[i]]
        trapezoid_y = [0, 0, y[i+1], y[i], 0]  # Cierra hacia el eje x
        trapezoids.append({'x': trapezoid_x, 'y': trapezoid_y})
    
    return area, trapezoids

def calculate_trapezoidal_error(expr, a, b, n):
    """
    Calcula el error estimado para el método del trapecio.
    """
    x = sp.Symbol('x')
    try:
        second_derivative = sp.diff(expr, x, 2)
        f_double_prime = sp.lambdify(x, second_derivative, modules=['numpy'])
        # Evaluar en múltiples puntos para encontrar el máximo absoluto
        x_vals = np.linspace(a, b, 1000)
        f_double_prime_vals = np.abs(f_double_prime(x_vals))
        max_f_double_prime = np.max(f_double_prime_vals)
        error = ((b - a)**3) / (12 * n**2) * max_f_double_prime
        return error
    except Exception as e:
        logger.error(f"Error al calcular la derivada segunda para el error del trapecio: {str(e)}")
        raise ValueError("No se pudo calcular el error estimado para el método del trapecio.")
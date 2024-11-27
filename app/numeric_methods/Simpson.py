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
def calculate_simpson_error(expr, a, b, n):
    """
    Calcula el error estimado para el método de Simpson.
    """
    x = sp.Symbol('x')
    try:
        fourth_derivative = sp.diff(expr, x, 4)
        f_fourth_prime = sp.lambdify(x, fourth_derivative, modules=['numpy'])
        # Evaluar en múltiples puntos para encontrar el máximo absoluto
        x_vals = np.linspace(a, b, 1000)
        f_fourth_prime_vals = np.abs(f_fourth_prime(x_vals))
        max_f_fourth_prime = np.max(f_fourth_prime_vals)
        error = ((b - a)**5) / (180 * n**4) * max_f_fourth_prime
        return error
    except Exception as e:
        logger.error(f"Error al calcular la derivada cuarta para el error de Simpson: {str(e)}")
        raise ValueError("No se pudo calcular el error estimado para el método de Simpson.")

def simpson_method(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("El número de subintervalos (n) debe ser par para el método de Simpson.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    area = (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])

    # Crear parábolas para graficar
    parabolas = []
    for i in range(0, n, 2):
        x_parabola = np.linspace(x[i], x[i+2], 100)  # Más puntos para una curva suave
        y_parabola = (
            y[i] * ((x_parabola - x[i+1]) * (x_parabola - x[i+2])) / ((x[i] - x[i+1]) * (x[i] - x[i+2]))
            + y[i+1] * ((x_parabola - x[i]) * (x_parabola - x[i+2])) / ((x[i+1] - x[i]) * (x[i+1] - x[i+2]))
            + y[i+2] * ((x_parabola - x[i]) * (x_parabola - x[i+1])) / ((x[i+2] - x[i]) * (x[i+2] - x[i+1]))
        )
        parabolas.append({'x': x_parabola, 'y': y_parabola})
    
    return area, parabolas
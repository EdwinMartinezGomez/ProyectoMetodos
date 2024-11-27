
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
                'iteration': i,
                'x': round(float(x2), 6),
                'fx': round(float(f(x2)), 6),
                'error': round(float(error), 6)
            })
            logger.info(f"Secante Iteración {i}: x = {x2}, f(x) = {f(x2)}, error = {error}")
            if error < tol:
                return x2, True, i, iteration_history
            x0, x1 = x1, x2
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método Secante: {str(e)}")
            return None, False, i, iteration_history
    return x1, False, max_iter, iteration_history
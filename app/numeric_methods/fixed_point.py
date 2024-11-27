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
                'iteration': i,
                'x': round(float(x_next), 6),
                'fx': round(float(fx), 6),
                'error': round(float(error), 6)
            })
            logger.info(f"Punto Fijo Iteración {i}: x = {x_next}, f(x) = {fx}, error = {error}")
            if error < tol:
                converged = True
                break
            x_prev = x_next
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método de Punto Fijo: {str(e)}")
            return None, False, i, iteration_history

    return x_next, converged, i, iteration_history
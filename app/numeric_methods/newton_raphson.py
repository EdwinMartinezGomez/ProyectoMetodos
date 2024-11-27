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
def newton_raphsonMethod(f, fprime, x0, max_iter=100, tol=1e-6):
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
                'iteration': i,
                'x': round(float(x_next), 6),
                'fx': round(float(fx), 6),
                'error': round(float(error), 6)
            })
            logger.info(f"Newton-Raphson Iteración {i}: x = {x_next}, f(x) = {fx}, error = {error}")
            if error < tol:
                return x_next, True, i, iteration_history
            x_prev = x_next
        except Exception as e:
            logger.error(f"Error en la iteración {i} del método Newton-Raphson: {str(e)}")
            return None, False, i, iteration_history
    return x_prev, False, max_iter, iteration_history
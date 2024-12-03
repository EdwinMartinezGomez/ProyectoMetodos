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
            'iteration': i,
            'x': round(float(c), 6),
            'fx': round(float(fc), 6),
            'error': round(float(error), 6)
        })
        logger.info(f"Bisección Iteración {i}: x = {c}, f(x) = {fc}, error = {error}")
        if abs(fc) < tol or error < tol:
            converged = True
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, converged, i, iteration_history
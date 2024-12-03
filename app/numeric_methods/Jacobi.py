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
def jacobi_method(A, b, x0, max_iter, tol=1e-6):
    n = len(b)
    x = x0.copy()
    iteration_history = []

    for i in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - s) / A[j][j]
        
        # Calcular el error como la norma infinita
        error = np.linalg.norm(x_new - x, ord=np.inf)
        
        # Almacenar el historial
        iteration_history.append({
            'iteration': i,
            'x': [round(float(val), 6) for val in x_new],
            'error': round(float(error), 6)
        })
        logger.info(f"Jacobi Iteración {i}: x = {x_new}, error = {error}")

        if error < tol:
            converged = True
            break
        x = x_new.copy()
    else:
        converged = False

    logger.debug("Resultado final: x = %s, converged = %s, iterations = %d", x, converged, i)
    return x.tolist(), converged, i, iteration_history
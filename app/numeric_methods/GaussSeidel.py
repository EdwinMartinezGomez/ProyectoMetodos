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
def gauss_seidel_method(A, b, x0, max_iter, tol=1e-6, iteration_history=None):
    """
    Implementación del método de Gauss-Seidel para resolver sistemas de ecuaciones lineales A x = b.
    """
    n = len(A)
    x = np.array(x0, dtype=float)
    converged = False

    for i in range(1, max_iter + 1):
        x_old = x.copy()
        for j in range(n):
            s = sum(A[j][k] * x[k] for k in range(n) if k != j)
            if A[j][j] == 0:
                raise ZeroDivisionError(f"División por cero detectada en la fila {j}.")
            x[j] = (b[j] - s) / A[j][j]
        
        # Calcular el error como la norma infinita
        error = np.linalg.norm(x - x_old, ord=np.inf)
        
        # Almacenar el historial
        if iteration_history is not None:
            iteration_history.append({
                'iteration': i,
                'x': [round(float(val), 6) for val in x],
                'error': round(float(error), 6)
            })
        logger.info(f"Gauss-Seidel Iteración {i}: x = {x}, error = {error}")

        if error < tol:
            converged = True
            break

    return x.tolist(), converged, i

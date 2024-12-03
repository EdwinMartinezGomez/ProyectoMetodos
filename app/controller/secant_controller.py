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
from app.util import equation as eq
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

def controller_secant(data):
    if not data or 'equation' not in data or 'x0' not in data or 'x1' not in data or 'iterations' not in data:
        return jsonify({'error': 'Faltan campos requeridos: equation, x0, x1, iterations'}), 400

    equation = data['equation']
    x0 = float(data['x0'])
    x1 = float(data['x1'])
    max_iter = int(data['iterations'])

    try:
        expr, f = eq.parse_equation(equation)
        iteration_history = []  # Inicializa iteration_history
        root, converged, iterations, iteration_history = secant.secant_method(f, x0, x1, max_iter)

        # Preparar los datos para el gráfico
        x_vals = np.linspace(min(x0, x1) - 10, max(x0, x1) + 10, 1000)
        y_vals = f(x_vals)

        trace_function = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='f(x)',
            line=dict(color='blue')
        )

        # Traza de las iteraciones
        secant_lines = [
            go.Scatter(
                x=[entry['x'], entry['x']],
                y=[0, entry['fx']],
                mode='lines+markers',
                line=dict(color='orange', dash='dash'),
                marker=dict(size=8),
                name=f'Iteración {i+1}'
            )
            for i, entry in enumerate(iteration_history)
        ]

        # Layout del gráfico
        layout = go.Layout(
            title="Convergencia del Método de la Secante",
            xaxis=dict(title='x'),
            yaxis=dict(title='f(x)'),
            plot_bgcolor='#f0f0f0'
        )

        # Generar la figura
        fig = go.Figure(data=[trace_function] + secant_lines, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        if root is not None:
            root_rounded = round(root, 6)
        else:
            root_rounded = None

        response = {
            'root': root_rounded,
            'converged': converged,
            'iterations': iterations,
            'iteration_history': iteration_history,
            'plot_json': graphJSON
        }
        logging.debug("Returning response")
        return jsonify(response)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500
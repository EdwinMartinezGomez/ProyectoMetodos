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
from app.util import equation
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

def controller_jacobi(data):
    if not data or 'matrix' not in data or 'vector' not in data or 'initial_guess' not in data or 'iterations' not in data:
        return jsonify({'error': 'Faltan campos requeridos: matrix, vector, initial_guess, iterations'}), 400

    A = np.array(data['matrix'])
    b = np.array(data['vector'])
    x0 = np.array(data['initial_guess'])
    max_iter = int(data['iterations'])

    try:
        root, converged, iterations, iteration_history = jacobi.jacobi_method(A, b, x0, max_iter)

        # Preparar los datos para la gráfica
        iterations_range = list(range(len(iteration_history)))
        traces = []
        for i in range(len(root)):
            trace = go.Scatter(
                x=iterations_range,
                y=[entry['x'][i] for entry in iteration_history],
                mode='lines+markers',
                name=f'Variable {i+1}',
                marker=dict(size=10)
            )
            traces.append(trace)

        layout = go.Layout(
            title="Convergencia del Método Jacobi",
            xaxis=dict(title='Iteración'),
            yaxis=dict(title='Valor de la Variable'),
            plot_bgcolor='#f0f0f0'
        )

        fig = go.Figure(data=traces, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        response = {
            'root': [round(val, 6) for val in root],
            'converged': converged,
            'iterations': iterations,
            'iteration_history': iteration_history,
            'plot_json': graphJSON
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
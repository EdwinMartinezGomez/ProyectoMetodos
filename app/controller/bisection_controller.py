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

def controller_bisection(data):
    if not data or 'equation' not in data or 'a' not in data or 'b' not in data or 'iterations' not in data:
        return jsonify({'error': 'Faltan campos requeridos: equation, a, b, iterations'}), 400

    equation = data['equation']
    a = float(data['a'])
    b = float(data['b'])
    max_iter = int(data['iterations'])

    try:
        expr, f = eq.parse_equation(equation)
        root, converged, iterations, iteration_history = bisection.bisection_method(f, a, b, max_iter)

        # Preparar los datos para el gráfico
        x_vals = np.linspace(a, b, 1000)
        y_vals = f(x_vals)

        # Traza de la función
        trace_function = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='f(x)',
            line=dict(color='blue')
        )

        # Traza de las iteraciones (puntos medios)
        trace_iterations = go.Scatter(
            x=[entry['mid'] for entry in iteration_history],
            y=[entry['f(mid)'] for entry in iteration_history],
            mode='markers+text',
            name='Iteraciones',
            marker=dict(size=10, color='red'),
            text=[f"Iteración {i+1}" for i in range(len(iteration_history))],
            textposition='top center'
        )

        # Layout del gráfico
        layout = go.Layout(
            title="Convergencia del Método de Bisección",
            xaxis=dict(title='x'),
            yaxis=dict(title='f(x)'),
            plot_bgcolor='#f0f0f0'
        )

        # Generar la figura
        fig = go.Figure(data=[trace_function, trace_iterations], layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        response = {
            'root': round(root, 6),
            'converged': converged,
            'iterations': iterations,
            'iteration_history': iteration_history,
            'plot_json': graphJSON
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
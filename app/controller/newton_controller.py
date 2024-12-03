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

def controller_newton(data):
    if not data or 'equation' not in data or 'initial_guess' not in data or 'iterations' not in data:
        return jsonify({'error': 'Faltan campos requeridos: equation, initial_guess, iterations'}), 400

    equation = data['equation']
    initial_guess = float(data['initial_guess'])
    max_iter = int(data['iterations'])

    try:
        expr, f = eq.parse_equation(equation)
        f_prime = eq.parse_derivative_equation(equation)
        root, converged, iterations, iteration_history = newton_raphson.newton_raphsonMethod(f, f_prime, initial_guess, max_iter)

        # Preparar los datos para el gráfico
        x_vals = np.linspace(initial_guess - 10, initial_guess + 10, 1000)
        y_vals = f(x_vals)

        trace_function = go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='blue'))

        # Traza de las iteraciones
        iteration_traces = [
            go.Scatter(
                x=[entry['x'], entry['x']],
                y=[0, entry['f(x)']],
                mode='lines+markers',
                name=f'Iteración {i+1}',
                line=dict(color='orange', dash='dot'),
                marker=dict(size=8)
            )
            for i, entry in enumerate(iteration_history)
        ]

        layout = go.Layout(
            title="Convergencia del Método Newton-Raphson",
            xaxis=dict(title='x'),
            yaxis=dict(title='f(x)'),
            plot_bgcolor='#f0f0f0'
        )

        fig = go.Figure(data=[trace_function] + iteration_traces, layout=layout)
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

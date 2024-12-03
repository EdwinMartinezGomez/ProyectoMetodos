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

def controller_trapecio(data):
    if not data or 'equation' not in data or 'a' not in data or 'b' not in data or 'n' not in data:
        return jsonify({'error': 'Faltan campos requeridos: equation, a, b, n'}), 400

    equation = data['equation']
    a = float(data['a'])
    b = float(data['b'])
    n = int(data['n'])

    try:
        expr, f = eq.parse_equation(equation)
        area, trapezoids = trapecio.trapezoidal_method(f, a, b, n)

        # Traza de la función y las áreas bajo la curva
        x_vals = np.linspace(a, b, 1000)
        y_vals = f(x_vals)

        trace_function = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='f(x)',
            line=dict(color='blue')
        )

        trapezoid_shapes = [
            go.Scatter(
                x=shape['x'],
                y=shape['y'],
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)',
                mode='lines',
                line=dict(color='rgba(255, 165, 0.5)'),
                showlegend=False
            )
            for shape in trapezoids
        ]

        layout = go.Layout(
            title="Integración usando el Método del Trapecio",
            xaxis=dict(title='x'),
            yaxis=dict(title='f(x)'),
            plot_bgcolor='#f0f0f0'
        )

        fig = go.Figure(data=[trace_function] + trapezoid_shapes, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        response = {
            'area': round(area, 6),
            'plot_json': graphJSON
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
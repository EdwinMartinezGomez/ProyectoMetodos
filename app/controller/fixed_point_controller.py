from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from app.numeric_methods import fixed_point
from app.util import equation as eq
import numpy as np
import plotly
import plotly.graph_objs as go
import json
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
from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

def controller_fixed(data):
    if not data or 'gFunction' not in data or 'initial_guess' not in data or 'iterations' not in data:
        return jsonify({'error': 'Faltan campos requeridos: gFunction, initial_guess, iterations'}), 400

    gFunction = data['gFunction']
    initial_guess = float(data['initial_guess'])
    max_iter = int(data['iterations'])

    try:
        logging.debug("Parsing g function")
        g = eq.parse_g_function(gFunction)
        
        logging.debug("Calling fixed_point_method")
        iteration_history = []  # Inicializa iteration_history
        root, converged, iterations, iteration_history = fixed_point.fixed_point_method(g, initial_guess, max_iter, iteration_history)

        logging.debug("Preparing data for plot")
        x_vals = [entry['x'] for entry in iteration_history]
        g_vals = [entry['fx'] for entry in iteration_history]  # Asegúrate de usar la clave correcta

        trace_iteration = go.Scatter(
            x=x_vals,
            y=g_vals,
            mode='lines+markers',
            name='Iteraciones',
            marker=dict(size=10, color='red')
        )

        layout = go.Layout(
            title="Convergencia del Método de Punto Fijo",
            xaxis=dict(title='x'),
            yaxis=dict(title='g(x)'),
            plot_bgcolor='#f0f0f0'
        )

        fig = go.Figure(data=[trace_iteration], layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        response = {
            'root': round(root, 6),
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
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
def parse_equations(equations, variables):
    A = []
    b = []
    var_symbols = sp.symbols(variables)
    for eq in equations:
        try:
            lhs, rhs = eq.split('=')
            sympy_eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            A.append([float(sympy_eq.lhs.coeff(var)) for var in var_symbols])
            b.append(float(sympy_eq.rhs))
        except Exception as e:
            logging.error("Error al analizar la ecuación '%s': %s", eq, str(e))
            raise
    return np.array(A), np.array(b)

def controller_jacobi(data):
    logging.debug("Datos recibidos: %s", data)
    
    if not data or 'equations' not in data or 'variables' not in data or 'initial_guess' not in data or 'iterations' not in data:
        logging.error("Faltan campos requeridos: equations, variables, initial_guess, iterations")
        return jsonify({'error': 'Faltan campos requeridos: equations, variables, initial_guess, iterations'}), 400

    try:
        equations = data['equations']
        variables = data['variables']
        initial_guess = data['initial_guess']
        max_iter = int(data['iterations'])
        
        logging.debug("Ecuaciones: %s", equations)
        logging.debug("Variables: %s", variables)
        logging.debug("Vector inicial x0: %s", initial_guess)
        logging.debug("Número máximo de iteraciones: %d", max_iter)

        A, b = parse_equations(equations, variables)
        x0 = np.array(initial_guess)
        
        logging.debug("Matriz A: %s", A)
        logging.debug("Vector b: %s", b)

        root, converged, iterations, iteration_history = jacobi.jacobi_method(A, b, x0, max_iter)
        
        if root is None or iteration_history is None:
            logging.error("El método Jacobi devolvió un valor None")
            return jsonify({'error': 'El método Jacobi devolvió un valor None'}), 500

        logging.debug("Raíz encontrada: %s", root)
        logging.debug("Convergencia: %s", converged)
        logging.debug("Número de iteraciones: %d", iterations)
        logging.debug("Historial de iteraciones: %s", iteration_history)

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
        logging.debug("Respuesta enviada: %s", response)
        return jsonify(response)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500
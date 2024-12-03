import plotly
import plotly.graph_objs as go
import json
import sympy as sp
import re
import logging
import numpy as np
from flask import Blueprint, request, jsonify, render_template
from app.numeric_methods import Broyden as broyden
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# Definir las transformaciones incluyendo 'convert_xor'
transformations = (
    standard_transformations +
    (implicit_multiplication_application,) +
    (convert_xor,)
)

# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def parse_equations(equations, variables):
    f = []
    var_symbols = sp.symbols(variables)
    for eq in equations:
        try:
            lhs, rhs = eq.split('=')
            lhs_expr = parse_expr(lhs, transformations=transformations)
            rhs_expr = parse_expr(rhs, transformations=transformations)
            sympy_eq = sp.Eq(lhs_expr, rhs_expr)
            f.append(sympy_eq)
        except Exception as e:
            logging.error("Error al analizar la ecuación '%s': %s", eq, str(e))
            raise
    return f, var_symbols

def controller_broyden(data):
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

        f, var_symbols = parse_equations(equations, variables)
        x0 = np.array(initial_guess, dtype=float)
        
        logging.debug("Funciones f: %s", f)
        logging.debug("Símbolos de variables: %s", var_symbols)

        root, converged, iterations, iteration_history = broyden.broyden_method(f, var_symbols, x0, max_iter)
        
        if root is None or iteration_history is None:
            logging.error("El método Broyden devolvió un valor None")
            return jsonify({'error': 'El método Broyden devolvió un valor None'}), 500

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
            title="Convergencia del Método Broyden",
            xaxis=dict(title='Iteración'),
            yaxis=dict(title='Valor de la Variable'),
            plot_bgcolor='#f0f0f0'
        )

        fig = go.Figure(data=traces, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        response = {
            'root': [round(float(val), 6) for val in root],
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
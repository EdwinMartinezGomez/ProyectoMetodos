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

# Controlador para el método de bisección
# Nuevo controlador simplificado para el método de bisección
# Configurar la gráfica principal con las mejoras deseadas
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

# Controlador para el método de bisección
# Nuevo controlador simplificado para el método de bisección
def controller_bisection(data):
    try:
        # Verificar si faltan campos requeridos
        required_fields = ['equation', 'a', 'b', 'iterations']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Falta el campo requerido: {field}'}), 400

        # Extraer y convertir los datos
        equation = data['equation']
        a = float(data['a'])
        b = float(data['b'])
        max_iter = int(data['iterations'])

        # Parsear la ecuación
        expr, f = eq.parse_equation(equation)
        iteration_history = []  # Historial de iteraciones

        # Ejecutar método de bisección
        root, converged, iterations, iteration_history = bisection.bisection_method(
            f, a, b, max_iter, iteration_history
        )

        # Generar datos para el gráfico
        x_vals = np.linspace(a, b, 1000)
        y_vals = f(x_vals)

        # Traza de la función
        traces = [go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Función',
            line=dict(color='blue')
        )]

        # Añadir líneas tangentes y puntos de iteración
        for i, entry in enumerate(iteration_history):
            x_tangent = entry['x']
            y_tangent = f(x_tangent)

            # Calcular pendiente (derivada numérica) en el punto
            delta_x = 1e-5
            slope = (f(x_tangent + delta_x) - f(x_tangent)) / delta_x
            tangent_line = slope * (x_vals - x_tangent) + y_tangent

            # Añadir línea tangente
            traces.append(go.Scatter(
                x=x_vals,
                y=tangent_line,
                mode='lines',
                name=f'Tangente Iter {i+1}',
                line=dict(dash='dash', color=f'rgb({50 + i * 20}, 100, 150)')
            ))

            # Añadir punto de intersección
            traces.append(go.Scatter(
                x=[x_tangent],
                y=[y_tangent],
                mode='markers+text',
                name=f'Iteración {i+1}',
                text=[f'Iter {i+1}'],
                textposition="top center",
                marker=dict(size=10, color='red')
            ))

        # Crear layout del gráfico
        layout = go.Layout(
            title="Convergencia del Método de Bisección con Tangentes",
            xaxis=dict(title='x'),
            yaxis=dict(title='f(x)'),
            plot_bgcolor='#f0f0f0'
        )

        # Crear figura y serializar a JSON
        fig = go.Figure(data=traces, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Respuesta JSON
        response = {
            'root': round(root, 6),
            'converged': converged,
            'iterations': iterations,
            'iteration_history': iteration_history,
            'plot_json': graphJSON
        }
        return jsonify(response)

    except Exception as e:
        logger.error("Error en el controlador de bisección: %s", str(e))
        return jsonify({'error': str(e)}), 500

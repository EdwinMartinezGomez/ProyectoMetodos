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


def render_integration_plot(f, a, b, n, method, extra_shapes):
    x_vals = np.linspace(a, b, 1000)
    y_vals = f(x_vals)

    function_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(color='blue')
    )
    
    data_traces = [function_trace]

    # Agregar las áreas bajo la curva (trapezoides o parábolas)
    for shape in extra_shapes:
        trace = go.Scatter(
            x=shape['x'],
            y=shape['y'],
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.2)',
            mode='lines',
            line=dict(color='rgba(0, 100, 255, 0.5)'),
            showlegend=False
        )
        data_traces.append(trace)

    layout = go.Layout(
        title=f"Integración usando el Método {method.capitalize()}",
        xaxis=dict(title='x'),
        yaxis=dict(title='f(x)'),
        plot_bgcolor='#f0f0f0'
    )

    fig = go.Figure(data=data_traces, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_plot(method, is_system, iteration_history=None, variables=None, f=None, root=None, a=None, b=None, x0=None, x1=None, integration_data=None):
    """
    Genera una gráfica interactiva basada en el método numérico.

    Args:
        method (str): Método utilizado ('jacobi', 'gauss_seidel', 'bisection', etc.).
        is_system (bool): Si es un sistema de ecuaciones (True) o una sola ecuación (False).
        iteration_history (list, optional): Historial de iteraciones con valores de variables o aproximaciones.
        variables (list, optional): Lista de nombres de las variables (solo para sistemas).
        f (function, optional): Función matemática (para ecuaciones individuales o integración).
        root (float, optional): Raíz calculada (solo para métodos de una ecuación).
        a (float, optional): Límite inferior del intervalo (solo para métodos de una ecuación/integración).
        b (float, optional): Límite superior del intervalo (solo para métodos de una ecuación/integración).
        x0 (float, optional): Estimación inicial (solo para métodos de una ecuación).
        x1 (float, optional): Segunda estimación inicial (solo para método Secante).
        integration_data (dict, optional): Datos específicos para integración (e.g., trapezoides o parábolas).

    Returns:
        str: Gráfica en formato JSON lista para renderizar con Plotly.
    """
    try:
        if is_system:
            # Graficar sistemas de ecuaciones (Jacobi, Gauss-Seidel, Broyden)
            if not iteration_history or not variables:
                raise ValueError("Faltan datos de iteraciones o variables para sistemas de ecuaciones.")
            
            iterations_numbers = list(range(1, len(iteration_history) + 1))
            plot_traces = []

            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c']
            for var_idx, var in enumerate(variables):
                var_values = [entry['x'][var_idx] for entry in iteration_history]
                trace = go.Scatter(
                    x=iterations_numbers,
                    y=var_values,
                    mode='lines+markers',
                    name=var,
                    line=dict(color=colors[var_idx % len(colors)], width=3),
                    marker=dict(size=8)
                )
                plot_traces.append(trace)

            layout = go.Layout(
                title='Convergencia de Variables por Iteración',
                xaxis=dict(title='Iteración'),
                yaxis=dict(title='Valor de las Variables'),
                plot_bgcolor='#ffffff'
            )
            fig = go.Figure(data=plot_traces, layout=layout)

        elif method in ['trapezoidal', 'simpson']:
            # Graficar integración numérica
            if integration_data is None:
                raise ValueError("Los datos de integración son requeridos para métodos de integración.")
            
            x_vals = integration_data['x_vals']
            y_vals = integration_data['y_vals']
            shapes = integration_data['shapes']  # Datos para trapezoides/parábolas
            
            traces = [
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name='f(x)',
                    line=dict(color='blue')
                )
            ]
            for shape in shapes:
                traces.append(
                    go.Scatter(
                        x=shape['x'],
                        y=shape['y'],
                        mode='lines',
                        fill='tozeroy',
                        name=shape['label'],
                        line=dict(color='rgba(0, 123, 255, 0.5)')
                    )
                )

            layout = go.Layout(
                title='Integración Numérica',
                xaxis=dict(title='x'),
                yaxis=dict(title='f(x)'),
                plot_bgcolor='#ffffff'
            )
            fig = go.Figure(data=traces, layout=layout)

        else:
            # Graficar métodos de una sola ecuación (Bisección, Secante, Newton-Raphson, Punto Fijo)
            if f is None:
                raise ValueError("La función f debe ser proporcionada para métodos de una ecuación.")
            
            plot_a = a if a is not None else (x0 - 10 if x0 is not None else -10)
            plot_b = b if b is not None else (x1 + 10 if x1 is not None else 10)
            x_vals = np.linspace(plot_a, plot_b, 1000)
            y_vals = [f(xi) for xi in x_vals]

            traces = [
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    name='f(x)',
                    line=dict(color='blue')
                )
            ]

            if iteration_history:
                iteration_x = []
                iteration_y = []
                for iter_data in iteration_history:
                    current_x = iter_data['x']
                    current_y = f(current_x)
                    iteration_x.append(current_x)
                    iteration_y.append(current_y)
                    traces.append(
                        go.Scatter(
                            x=iteration_x.copy(),
                            y=iteration_y.copy(),
                            mode='lines+markers',
                            name='Iteración',
                            line=dict(color='orange', dash='dash')
                        )
                    )

            if root is not None:
                traces.append(
                    go.Scatter(
                        x=[root],
                        y=[f(root)],
                        mode='markers',
                        name='Raíz',
                        marker=dict(color='green', size=10, symbol='star')
                    )
                )

            layout = go.Layout(
                title=f'Convergencia ({method})',
                xaxis=dict(title='x'),
                yaxis=dict(title='f(x)'),
                plot_bgcolor='#ffffff'
            )
            fig = go.Figure(data=traces, layout=layout)

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        raise RuntimeError(f"Error al generar la gráfica: {str(e)}")

from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import sympy as sp
import re
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def broyden_method(F, J_initial, x0, max_iter=100, tol=1e-6, iteration_history=None):
    """
    Implementación del Método de Broyden para sistemas de ecuaciones no lineales.

    Args:
        F: Función que toma un vector x y devuelve F(x).
        J_initial: Matriz jacobiana inicial.
        x0: Estimación inicial (lista o array).
        max_iter: Número máximo de iteraciones.
        tol: Tolerancia para el criterio de convergencia.
        iteration_history: Lista para almacenar el historial de iteraciones.

    Returns:
        x: Solución encontrada.
        converged: Booleano indicando si el método convergió.
        iterations: Número de iteraciones realizadas.
    """
    x = np.array(x0, dtype=float)
    J = np.array(J_initial, dtype=float)
    converged = False

    for i in range(1, max_iter + 1):
        try:
            # Resolver J * delta = -F(x)
            delta = np.linalg.solve(J, -F(x))
        except np.linalg.LinAlgError:
            logger.error(f"Jacobian singular en la iteración {i}.")
            return None, False, i

        x_new = x + delta
        F_new = F(x_new)
        error = np.linalg.norm(delta, ord=np.inf)

        if iteration_history is not None:
            iteration_history.append({
                'iteration': i,
                'x': [round(float(val), 6) for val in x_new],
                'F(x)': [round(float(val), 6) for val in F_new],
                'error': round(float(error), 6)
            })

        logger.info(f"Broyden Iteración {i}: x = {x_new}, F(x) = {F_new}, error = {error}")

        if error < tol:
            converged = True
            x = x_new
            break

        # Actualizar la matriz Jacobiana usando la actualización de Broyden
        y = F_new - F(x)
        delta = delta.reshape(-1, 1)
        y = y.reshape(-1, 1)
        J_delta = J @ delta
        denom = (delta.T @ delta)[0, 0]
        if denom == 0:
            logger.error(f"Denominador cero en la actualización de Jacobiano en la iteración {i}.")
            return None, False, i
        J += ((y - J_delta) @ delta.T) / denom

        x = x_new

    return x.tolist(), converged, i

def render_broyden_plot(iteration_history, variables):
    """
    Genera una gráfica animada de la convergencia del Método de Broyden.

    Args:
        iteration_history: Lista con el historial de iteraciones.
        variables: Lista de nombres de variables.

    Returns:
        plot_json: JSON para Plotly.
    """
    data_traces = []
    frames = []
    iterations = [entry['iteration'] for entry in iteration_history]

    # Colores para cada variable
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#1abc9c']

    # Crear trazas iniciales (vacías)
    for idx, var in enumerate(variables):
        trace = go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name=var,
            line=dict(color=colors[idx % len(colors)], width=2)
        )
        data_traces.append(trace)

    # Crear frames para cada iteración
    for i, entry in enumerate(iteration_history):
        frame_data = []
        for idx, var in enumerate(variables):
            frame_data.append(
                go.Scatter(
                    x=iterations[:i+1],
                    y=[iter_val for iter_val in [h['x'][idx] for h in iteration_history[:i+1]]],
                    mode='lines+markers',
                    name=var,
                    line=dict(color=colors[idx % len(colors)], width=2)
                )
            )
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Layout con controles de animación
    layout = go.Layout(
        title='Convergencia del Método de Broyden',
        xaxis=dict(title='Iteración', range=[min(iterations), max(iterations)]),
        yaxis=dict(title='Valor de la Variable'),
        plot_bgcolor='#f0f0f0',
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=1.05,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 300}}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Iteración: "},
                pad={"t": 50},
                steps=[
                    dict(method='animate',
                         args=[[str(k)],
                               {"frame": {"duration": 700, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}}],
                         label=str(k+1))
                    for k in range(len(frames))
                ]
            )
        ]
    )

    fig = go.Figure(data=data_traces, layout=layout, frames=frames)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

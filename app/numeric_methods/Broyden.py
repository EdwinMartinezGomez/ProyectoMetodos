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
   
def render_broyden_plot(exprs, variables, root):
    """
    Genera una gráfica mejorada de las funciones del sistema con cuadrícula y líneas de eje en los orígenes.

    Args:
        exprs: Lista de expresiones SymPy que representan las ecuaciones del sistema.
        variables: Lista de nombres de las variables (strings).
        root: Lista o array con las soluciones encontradas para cada variable.

    Returns:
        plot_json: JSON para Plotly.
    """
    if len(variables) != 2 or len(exprs) != 2:
        raise ValueError("Actualmente, solo se soporta la visualización para sistemas de dos ecuaciones y dos variables.")

    var1, var2 = variables
    x_sym, y_sym = sp.symbols(variables)

    # Crear funciones lambda para las ecuaciones
    f1 = sp.lambdify((x_sym, y_sym), exprs[0], modules=['numpy'])
    f2 = sp.lambdify((x_sym, y_sym), exprs[1], modules=['numpy'])

    # Definir el rango de la gráfica
    x_min, x_max = root[0] - 10, root[0] + 10
    y_min, y_max = root[1] - 10, root[1] + 10

    # Crear una malla de puntos
    x = np.linspace(x_min, x_max, 500)
    y = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x, y)

    # Evaluar las ecuaciones en la malla
    Z1 = f1(X, Y)
    Z2 = f2(X, Y)

    # Dibujar las curvas de las ecuaciones evaluadas
    curve1 = go.Contour(
        x=x,
        y=y,
        z=Z1,
        colorscale=[[0, 'blue'], [1, 'blue']],
        line=dict(width=2),
        contours=dict(start=0, end=0, coloring="lines"),
        name=f'{sp.pretty(exprs[0])} = 0'
    )

    curve2 = go.Contour(
        x=x,
        y=y,
        z=Z2,
        colorscale=[[0, 'red'], [1, 'red']],
        line=dict(width=2),
        contours=dict(start=0, end=0, coloring="lines"),
        name=f'{sp.pretty(exprs[1])} = 0'
    )

    # Agregar la solución encontrada
    solution_trace = go.Scatter(
        x=[root[0]],
        y=[root[1]],
        mode='markers+text',
        name='Solución',
        marker=dict(color='green', size=10, symbol='star'),
        text=[f'Solución: ({root[0]:.6f}, {root[1]:.6f})'],
        textposition='top right',
        hovertemplate='Solución: (%{x:.6f}, %{y:.6f})<extra></extra>'
    )

    # Líneas del eje en los orígenes
    x_axis = go.Scatter(
        x=np.linspace(x_min, x_max, 500),
        y=[0] * 500,
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        name='Eje X'
    )

    y_axis = go.Scatter(
        x=[0] * 500,
        y=np.linspace(y_min, y_max, 500),
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        name='Eje Y'
    )

    # Definir el layout con cuadrícula y ejes
    layout = go.Layout(
        title='Intersección de las Funciones del Sistema',
        xaxis=dict(
            title=var1,
            range=[x_min, x_max],
            showgrid=True,  # Mostrar cuadrícula
            zeroline=True,  # Línea en el origen
            zerolinecolor='gray',  # Color de la línea en el origen
            gridcolor='lightgray'  # Color de la cuadrícula
        ),
        yaxis=dict(
            title=var2,
            range=[y_min, y_max],
            showgrid=True,  # Mostrar cuadrícula
            zeroline=True,  # Línea en el origen
            zerolinecolor='gray',  # Color de la línea en el origen
            gridcolor='lightgray'  # Color de la cuadrícula
        ),
        width=900,
        height=900,
        showlegend=True,
        plot_bgcolor='white'
    )

    # Crear la figura y devolverla como JSON
    fig = go.Figure(data=[curve1, curve2, solution_trace, x_axis, y_axis], layout=layout)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

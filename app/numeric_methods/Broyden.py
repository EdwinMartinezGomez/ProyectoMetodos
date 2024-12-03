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
def broyden_method(f, var_symbols, x0, max_iter, tol=1e-6):
    n = len(var_symbols)
    x = x0.copy()
    iteration_history = []

    # Inicializar la matriz Jacobiana aproximada
    J_inv = np.eye(n)

    for i in range(1, max_iter + 1):
        f_val = np.array([float(eq.subs(dict(zip(var_symbols, x)))) for eq in f])
        delta_x = -J_inv @ f_val
        x_new = x + delta_x
        
        # Calcular el error como la norma infinita
        error = np.linalg.norm(delta_x, ord=np.inf)
        
        # Almacenar el historial
        iteration_history.append({
            'iteration': i,
            'x': [round(float(val), 6) for val in x_new],
            'error': round(float(error), 6)
        })
        logger.info(f"Broyden Iteración {i}: x = {x_new}, error = {error}")

        if error < tol:
            converged = True
            break

        # Actualizar la matriz Jacobiana aproximada
        f_val_new = np.array([float(eq.subs(dict(zip(var_symbols, x_new)))) for eq in f])
        y = f_val_new - f_val
        J_inv += np.outer(delta_x - J_inv @ y, delta_x) / np.dot(delta_x, delta_x)

        x = x_new.copy()
    else:
        converged = False

    logger.debug("Resultado final: x = %s, converged = %s, iterations = %d", x, converged, i)
    return x.tolist(), converged, i, iteration_history
   
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

from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from app.numeric_methods import newton_raphson
from app.util import equation as eq
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import sympy as sp
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
    try:
        # Verificar si faltan campos requeridos
        required_fields = ['equation', 'initial_guess', 'iterations']
        for field in required_fields:
            if field not in data:
                logger.error(f'Faltan campos requeridos: {field}')
                return jsonify({'error': f'Faltan campos requeridos: {field}'}), 400

        # Extraer y convertir los datos
        equation = data['equation']
        try:
            initial_guess = float(data['initial_guess'])
            max_iter = int(data['iterations'])
        except ValueError:
            logger.error('initial_guess debe ser un número y iterations debe ser un entero.')
            return jsonify({'error': 'initial_guess debe ser un número y iterations debe ser un entero.'}), 400

        # Parsear la función f(x) y su derivada f'(x)
        try:
            expr, f = eq.parse_equation(equation)
            f_prime = eq.parse_derivative_equation(equation)  # Corrección: solo se asigna f_prime
        except Exception as e:
            logger.error(f"Error al parsear la ecuación o su derivada: {str(e)}")
            return jsonify({'error': f"Error al parsear la ecuación o su derivada: {str(e)}"}), 400

        # Inicializar el historial de iteraciones
        iteration_history = []

        # Ejecutar el método de Newton-Raphson
        try:
            root, converged, iterations, iteration_history = newton_raphson.newton_raphsonMethod(
                f, f_prime, initial_guess, max_iter, iteration_history
            )
        except ValueError as ve:
            logger.error(str(ve))
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logger.error(f"Error en el método de Newton-Raphson: {str(e)}")
            return jsonify({'error': f"Error en el método de Newton-Raphson: {str(e)}"}), 500

        # Preparar la gráfica
        try:
            # Función para crear layouts mejorados
            def create_enhanced_layout(title, x_label='Iteración', y_label='x'):
                return go.Layout(
                    title=dict(
                        text=title,
                        x=0.5,
                        xanchor='center',
                        font=dict(size=24, family='Arial, sans-serif', color='#2c3e50')
                    ),
                    xaxis=dict(
                        title=dict(
                            text=x_label,
                            font=dict(size=16, family='Arial, sans-serif')
                        ),
                        gridcolor='#e0e0e0',
                        zerolinecolor='#2c3e50',
                        zerolinewidth=2
                    ),
                    yaxis=dict(
                        title=dict(
                            text=y_label,
                            font=dict(size=16, family='Arial, sans-serif')
                        ),
                        gridcolor='#e0e0e0',
                        zerolinecolor='#2c3e50',
                        zerolinewidth=2
                    ),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(
                        x=1.05,
                        y=1,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='#2c3e50'
                    ),
                    margin=dict(l=80, r=80, t=100, b=80)
                )

            # Generar valores para la gráfica de f(x)
            # Definir el rango de la gráfica basado en las iteraciones
            all_x = [entry['x'] for entry in iteration_history]
            plot_a, plot_b = min(all_x) - 5, max(all_x) + 5
            x_vals = np.linspace(plot_a, plot_b, 1000)
            try:
                y_vals = [f(xi) for xi in x_vals]
            except Exception as e:
                logger.error(f"Error al evaluar la función para la gráfica: {str(e)}")
                y_vals = [float('nan') for _ in x_vals]

            # Trace de la función f(x)
            trace_function = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='f(x)',
                line=dict(color='blue'),
                hoverinfo='none'
            )

            # Inicializar las trazas de iteraciones
            data_traces = [trace_function]

            # Inicializar los frames para la animación
            frames = []
            iteration_numbers = list(range(1, len(iteration_history) + 1))
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'yellow', 'black', 'gray']

            for i, entry in enumerate(iteration_history):
                current_x = entry['x']
                current_fx = entry['fx']

                # Calcular la pendiente (derivada) en el punto actual
                try:
                    slope = f_prime(current_x)
                except Exception as e:
                    logger.error(f"Error al calcular la derivada en x={current_x}: {str(e)}")
                    slope = 0

                # Calcular la línea tangente en el punto actual
                tangent_x = np.linspace(current_x - 5, current_x + 5, 100)
                tangent_y = slope * (tangent_x - current_x) + current_fx

                # Trace de la línea tangente
                tangent_trace = go.Scatter(
                    x=tangent_x,
                    y=tangent_y,
                    mode='lines',
                    name=f'Tangente Iter {i+1}',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    opacity=0.5,
                    showlegend=False
                )

                # Trace del punto actual
                point_trace = go.Scatter(
                    x=[current_x],
                    y=[current_fx],
                    mode='markers+text',
                    name=f'Iteración {i+1}',
                    marker=dict(color=colors[i % len(colors)], size=12, symbol='circle'),
                    text=[f"{i+1}"],
                    textposition='top center',
                    hovertemplate=f"Iteración {i+1}: x = {current_x:.6f}, f(x) = {current_fx:.6f}<extra></extra>"
                )

                # Crear frame con las trazas hasta la iteración actual
                frame = go.Frame(
                    data=[trace_function, tangent_trace, point_trace],
                    name=f'frame{i+1}'
                )
                frames.append(frame)

                # Agregar las trazas al data_traces para la visualización inicial
                data_traces.extend([tangent_trace, point_trace])

            # Definir el layout de la gráfica con animación
            layout = create_enhanced_layout('Convergencia del Método de Newton-Raphson', 'Iteración', 'x')

            # Agregar controles de animación
            layout.update(
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {
                                "frame": {"duration": 700, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "cubic-in-out"}
                            }],
                            "label": "▶ Reproducir",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }],
                            "label": "❚❚ Pausar",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "y": 1.1,
                    "xanchor": "right",
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 16},
                        "prefix": "Iteración: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f'frame{k}'],
                                {"frame": {"duration": 700, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}
                            ],
                            "label": f"Iteración {k}",
                            "method": "animate"
                        } for k in range(1, len(frames) + 1)
                    ]
                }]
            )

            # Crear la figura
            fig = go.Figure(data=data_traces, layout=layout, frames=frames)

            # Serializar la figura a JSON
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Preparar la respuesta JSON
            response = {
                'root': round(root, 6),
                'converged': converged,
                'iterations': iterations,
                'iteration_history': iteration_history,
                'plot_json': graphJSON
            }

            logger.debug("Returning response")
            return jsonify(response)

        except Exception as e:
            logger.exception("Error inesperado en el controlador de Newton-Raphson.")
            return jsonify({'error': 'Ocurrió un error inesperado durante el cálculo.'}), 500

    except Exception as e:
        logger.exception("Error inesperado en el controlador de Newton-Raphson.")
        return jsonify({'error': 'Ocurrió un error inesperado durante el cálculo.'}), 500

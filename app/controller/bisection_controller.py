from flask import Blueprint, request, jsonify, render_template
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from app.numeric_methods import bisection
from app.util import equation as eq
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import logging

# Configuración del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Definir las transformaciones incluyendo 'convert_xor'
transformations = (
    standard_transformations +
    (implicit_multiplication_application,) +
    (convert_xor,)
)

def controller_bisection(data):
    try:
        # Verificar si faltan campos requeridos
        required_fields = ['equation', 'a', 'b', 'iterations']
        for field in required_fields:
            if field not in data:
                logger.error(f'Falta el campo requerido: {field}')
                return jsonify({'error': f'Falta el campo requerido: {field}'}), 400

        # Extraer y convertir los datos
        equation = data['equation']
        try:
            a = float(data['a'])
            b = float(data['b'])
            max_iter = int(data['iterations'])
        except ValueError:
            logger.error('Los valores de a, b deben ser números y iterations debe ser un entero.')
            return jsonify({'error': 'Los valores de a, b deben ser números y iterations debe ser un entero.'}), 400

        # Parsear la ecuación
        try:
            expr, f = eq.parse_equation(equation)
        except Exception as e:
            logger.error(f"Error al parsear la ecuación: {str(e)}")
            return jsonify({'error': f"Error al parsear la ecuación: {str(e)}"}), 400

        # Inicializar el historial de iteraciones
        iteration_history = []

        # Ejecutar el método de bisección
        try:
            root, converged, iterations, iteration_history = bisection.bisection_method(
                f, a, b, max_iter, iteration_history
            )
        except ValueError as ve:
            logger.error(str(ve))
            return jsonify({'error': str(ve)}), 400
        except Exception as e:
            logger.error(f"Error en el método de bisección: {str(e)}")
            return jsonify({'error': f"Error en el método de bisección: {str(e)}"}), 500

        # Preparar la gráfica
        try:
            # Función para crear layouts mejorados
            def create_enhanced_layout(title, x_label='x', y_label='f(x)'):
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

            # Definir el rango de la gráfica con un margen
            margin = (b - a) * 0.1  # 10% de margen
            plot_a, plot_b = a - margin, b + margin

            # Evaluar los valores de y para la gráfica
            x_vals = np.linspace(plot_a, plot_b, 1000)
            y_vals = []
            for xi in x_vals:
                try:
                    yi = f(xi)
                    y_vals.append(yi)
                except Exception as e:
                    logger.warning(f"f({xi}) no está definido: {e}")
                    y_vals.append(float('nan'))

            # Trace de la función principal
            function_trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name='f(x)',
                line=dict(color='blue'),
                hoverinfo='none'
            )

            # Trace de rastro inicial (vacío)
            trace_rastro_initial = go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name='Rastro',
                line=dict(color='orange', dash='dash'),
                hoverinfo='none'
            )

            # Trace inicial para las aproximaciones (vacío)
            approx_trace_initial = go.Scatter(
                x=[],
                y=[],
                mode='markers+text',
                name='Aproximación',
                marker=dict(color='red', size=10),
                text=[],
                textposition='top center',
                hoverinfo='none'
            )

            # Inicializar data_traces con las trazas base
            data_traces = [function_trace, trace_rastro_initial, approx_trace_initial]

            # Inicializar los frames y las listas de iteración
            frames = []
            iteration_x = []
            iteration_y = []

            # Agregar las iteraciones a los traces para la animación
            if iteration_history:
                for idx, iter_data in enumerate(iteration_history):
                    current_x = iter_data['x']
                    try:
                        current_y = f(current_x)
                    except Exception as e:
                        logger.warning(f"f({current_x}) no está definido: {e}")
                        current_y = float('nan')
                    
                    iteration_x.append(current_x)
                    iteration_y.append(current_y)

                    # Trace de la aproximación actual
                    approx_trace = go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode='markers+text',
                        name='Iteración',
                        marker=dict(color='red', size=10),
                        text=[f"{idx + 1}"],
                        textposition='top center',
                        hovertemplate=f"Iteración {idx + 1}: x = {current_x:.6f}, f(x) = {current_y:.6f}<extra></extra>"
                    )

                    # Trace de rastro acumulado
                    trace_rastro = go.Scatter(
                        x=iteration_x.copy(),
                        y=iteration_y.copy(),
                        mode='lines',
                        name='Rastro',
                        line=dict(color='orange', dash='dash'),
                        hoverinfo='none'
                    )

                    # Crear frame con las trazas de iteración
                    frames.append(go.Frame(
                        data=[trace_rastro, approx_trace],
                        traces=[1, 2],
                        name=str(idx)
                    ))

            # Agregar trace de la raíz final
            if root is not None:
                try:
                    y_root = f(root)
                except Exception as e:
                    logger.error(f"Error al evaluar f(root): {str(e)}")
                    y_root = float('nan')

                root_trace = go.Scatter(
                    x=[root],
                    y=[y_root],
                    mode='markers',
                    name='Raíz',
                    marker=dict(color='green', size=12, symbol='star'),
                    hoverinfo='x+y'
                )
                data_traces.append(root_trace)

            # Definir el layout de la gráfica con animación
            layout = create_enhanced_layout('Visualización del Método de Bisección')

            # Agregar controles de animación mejorados
            layout.update(
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {
                                "frame": {"duration": 700, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "cubic-in-out"}
                            }],
                            "label": "▶️ Reproducir",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }],
                            "label": "⏸️ Pausar",
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
                                [str(k)],
                                {"frame": {"duration": 700, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}
                            ],
                            "label": f"Iteración {k + 1}",
                            "method": "animate"
                        } for k in range(len(frames))
                    ]
                }]
            )

            # Crear la figura con los datos y frames
            fig = go.Figure(data=data_traces, layout=layout, frames=frames)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Preparar la respuesta JSON
            response = {
                'root': round(root, 6),
                'converged': converged,
                'iterations': iterations,
                'iteration_history': iteration_history,
                'plot_json': graphJSON
            }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error al generar la gráfica para la ecuación: {str(e)}")
            return jsonify({'error': f"Error al generar la gráfica para la ecuación: {str(e)}"}), 400

    except Exception as e:
        logger.exception("Error inesperado en el controlador de bisección.")
        return jsonify({'error': 'Ocurrió un error inesperado durante el cálculo.'}), 500

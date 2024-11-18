import unittest
from app.routes import render_integration_plot  # Asegúrate de que el módulo sea correcto
import numpy as np
import json
import plotly.graph_objects as go

class TestRenderIntegrationPlot(unittest.TestCase):
    def test_render_integration_plot(self):
        # Parámetros para el método trapezoidal
        equation = lambda x: np.exp(x**4)
        a, b = -1, 1
        n = 100
        method = 'trapezoidal'

        # Lista de formas para extra_shapes
        extra_shapes = [
            {'x': [-1, -0.5, 0], 'y': [0, 0.5, 1]},
            {'x': [0, 0.5, 1], 'y': [1, 0.5, 0]}
        ]

        # Usar la función render_integration_plot para generar la gráfica
        try:
            plot_json = render_integration_plot(equation, a, b, n, method, extra_shapes)
        except Exception as e:
            self.fail(f"Error al ejecutar render_integration_plot: {e}")

        # Convertir el JSON generado de nuevo a un objeto Plotly
        try:
            fig_dict = json.loads(plot_json)
            fig = go.Figure(fig_dict)
        except json.JSONDecodeError as e:
            self.fail(f"Error al decodificar el JSON generado: {e}")

        # Verificar que los datos de la gráfica no están vacíos
        self.assertGreater(len(fig.data), 0, "La gráfica no contiene datos.")

        # Mostrar la gráfica
        fig.show()

# Ejecutar el test
if __name__ == '__main__':
    unittest.main()

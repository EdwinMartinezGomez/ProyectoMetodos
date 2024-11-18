# tests/test_plot.py
import unittest
import json
import plotly.graph_objects as go
from app.routes import render_integration_plot  # Cambia app.utils al módulo donde esté definida esta función

class TestRenderIntegrationPlot(unittest.TestCase):
    def test_render_integration_plot_trapezoidal(self):
        # Define una función simple f(x) = x^2
        def f(x): 
            return x ** 2

        # Define parámetros para el test
        a, b, n = 0, 5, 10
        method = 'trapezoidal'

        # Llamar a la función
        plot_json = render_integration_plot(f, a, b, n, method)

        # Validar que plot_json es un string serializable en JSON
        self.assertIsInstance(plot_json, str)

        # Mostrar la gráfica
        plot_data = json.loads(plot_json)
        fig = go.Figure(plot_data)
        fig.show()  # Esto abrirá el gráfico en el navegador

if __name__ == '__main__':
    unittest.main()

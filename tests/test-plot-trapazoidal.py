import unittest
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad

class TestTrapezoidalPlot(unittest.TestCase):
    def trapezoidal_method(self, f, a, b, n):
        """Método trapezoidal para calcular el área bajo una curva."""
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        area = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
        return area, x, y

    def test_plot_trapezoidal(self):
        # Parámetros
        equation = lambda x: np.exp(x**4)  # Definir función f(x)
        a, b = -1, 1
        n = 100

        # Calcular el área usando el método trapezoidal
        area, x, y = self.trapezoidal_method(equation, a, b, n)

        # Valores de referencia para comparación
        true_area, _ = quad(equation, a, b)

        # Comprobar que el área calculada está cerca del valor real
        self.assertAlmostEqual(area, true_area, delta=0.01)

        # Generar la gráfica usando Plotly
        fig = go.Figure()

        # Trazar la función original
        x_full = np.linspace(a, b, 1000)
        y_full = equation(x_full)
        fig.add_trace(go.Scatter(x=x_full, y=y_full, mode='lines', name='f(x) = e^(x^4)'))

        # Agregar trapezoides
        for i in range(n):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = equation(x0), equation(x1)
            fig.add_trace(go.Scatter(
                x=[x0, x0, x1, x1],
                y=[0, y0, y1, 0],
                fill='toself',
                mode='lines',
                line=dict(width=0.5, color='rgba(0,100,80,0.2)'),
                name=f'Trapezoide {i+1}'
            ))

        # Configurar la gráfica
        fig.update_layout(
            title=f'Método Trapezoidal: Área = {area:.6f}',
            xaxis_title='x',
            yaxis_title='f(x)',
            showlegend=False
        )

        # Mostrar la gráfica
        fig.show()

# Ejecutar el test
if __name__ == '__main__':
    unittest.main()

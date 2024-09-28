from typing import Callable, Tuple
import matplotlib.pyplot as plt

def simpson(function: str, lower_limit: float, upper_limit: float, intervals: int) -> Tuple[float, float, plt.Figure]:
    """
    Calculates the approximate integral of a function using Simpson's rule.

    Parameters:
    function (str): The function to integrate.
    lower_limit (float): The lower limit of integration.
    upper_limit (float): The upper limit of integration.
    intervals (int): The number of intervals to use in the approximation.

    Returns:
    A tuple containing the approximate integral, the percent error, and the graph of the function and approximation.
    """
    # Check if number of intervals is even
    if intervals % 2 != 0:
        print("El número de intervalos debe ser par.")
        return None, None, None

    # Parse function string into a callable function
    try:
        f = lambda x: eval(function)
    except:
        print("Error al ingresar la función.")
        return None, None, None

    # Calculate interval width and x values
    h = (upper_limit - lower_limit) / intervals
    x_values = [lower_limit + i * h for i in range(intervals + 1)]

    # Calculate y values
    y_values = [f(x) for x in x_values]

    # Calculate approximate integral using Simpson's rule
    approx = (h / 3) * (y_values[0] + 4 * sum(y_values[1:-1:2]) + 2 * sum(y_values[2:-1:2]) + y_values[-1])

    # Calculate actual integral using scipy.integrate.quad
    try:
        import scipy.integrate as spi
        actual, _ = spi.quad(f, lower_limit, upper_limit)
    except:
        print("Error al calcular la integral real.")
        actual = None

    # Calculate percent error
    if actual is not None:
        error_percent = abs((actual - approx) / actual) * 100
    else:
        error_percent = None

    # Create graph
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label="Función")
    ax.plot(x_values, y_values, "bo")
    ax.plot(x_values[::2], y_values[::2], "g-", label="Trapecios")
    ax.plot(x_values[1::2], y_values[1::2], "r-", label="Simpson")
    ax.legend()
    ax.set_title("Aproximación de integral usando Simpson")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return approx, error_percent, fig

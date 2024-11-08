import pytest
from app.numeric_methods import (
    bisection_method,
    newton_raphson,
    secant_method,
    fixed_point_method
)

def test_bisection():
    # Prueba para x^2 - 4 = 0
    result = bisection_method("x**2 - 4", 0, 3)
    assert abs(result['root'] - 2.0) < 1e-6
    assert result['convergence'] is True

def test_newton():
    # Prueba para x^2 - 4 = 0
    result = newton_raphson("x**2 - 4", x0=3)
    assert abs(result['root'] - 2.0) < 1e-6
    assert result['convergence'] is True

def test_secant():
    # Prueba para x^2 - 4 = 0
    result = secant_method("x**2 - 4", x0=1, x1=3)
    assert abs(result['root'] - 2.0) < 1e-6
    assert result['convergence'] is True

def test_fixed_point():
    # Prueba para x = cos(x)
    result = fixed_point_method("cos(x)", x0=0)
    assert result['convergence'] is True
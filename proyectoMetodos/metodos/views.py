import numpy as np
import math
from django.shortcuts import render
from .forms import SecanteForm

def f(x, funcionMath):
    """Evalúa la función dada en el punto x."""
    try:
        return eval(funcionMath, {"x": x, "math": math, "np": np})
    except Exception as e:
        print(f"Error evaluating functionMath: {e}")
        return None
def secante(funcion, p0, p1, tol, n):
    """Implementa el método de la secante con dos puntos iniciales."""
    result = ""
    
    for i in range(n):
        f_p0 = f(p0, funcion)
        f_p1 = f(p1, funcion)
        
        if f_p0 is None or f_p1 is None:
            result += "Error al evaluar la función.\n"
            return result
        
        # Comprobación de división por cero
        if f_p1 - f_p0 == 0:
            result += "Error: División por cero.\n"
            return result
        
        # Fórmula del método de la secante
        p2 = p1 - f_p1 * (p1 - p0) / (f_p1 - f_p0)
        result += f"Iteración {i + 1}: p0 = {p0}, p1 = {p1}, p2 = {p2}\n"
        
        # Condición de convergencia
        if abs(p2 - p1) < tol:
            result += f"La raíz es {p2} después de {i + 1} iteraciones\n"
            return result
        
        p0 = p1
        p1 = p2
    
    result += "El método no converge después de las iteraciones especificadas.\n"
    return result


def secante_view(request):
    """Vista para el método de la secante."""
    if request.method == 'POST':
        form = SecanteForm(request.POST)
        if form.is_valid():
            funcion = form.cleaned_data['funcion']
            p0 = form.cleaned_data['p0']
            p1 = form.cleaned_data['p1']  # Obtener el segundo punto inicial
            tol = form.cleaned_data['tol']
            n = form.cleaned_data['n']
            resultado = secante(funcion, p0, p1, tol, n)  # Pasar p1 a la función
            return render(request, 'metodos/resultados.html', {'resultado': resultado, 'form': form})
    else:
        form = SecanteForm()

    return render(request, 'metodos/secante.html', {'form': form})


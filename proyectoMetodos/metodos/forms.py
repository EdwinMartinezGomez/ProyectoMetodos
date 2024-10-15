from django import forms

class SecanteForm(forms.Form):
    funcion = forms.CharField(label='Función f(x)', max_length=100)
    p0 = forms.FloatField(label='Punto inicial (p0)')
    p1 = forms.FloatField(label='Segundo punto inicial (p1)')  # Nuevo campo
    tol = forms.FloatField(label='Tolerancia', initial=0.0001)
    n = forms.IntegerField(label='Número máximo de iteraciones', initial=1000)

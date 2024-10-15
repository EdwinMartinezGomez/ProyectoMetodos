from django.urls import path
from .views import secante_view

urlpatterns = [
    path('secante/', secante_view, name='secante'),
]

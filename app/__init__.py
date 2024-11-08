from flask import Flask
from .routes import main  # Asegúrate de que esta línea esté presente

def create_app():
    app = Flask(__name__)

    # Aquí puedes configurar tu aplicación, como cargar configuraciones o inicializar extensiones.

    # Registra el blueprint
    app.register_blueprint(main)

    return app

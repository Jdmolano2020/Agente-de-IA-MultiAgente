import os

# Configurar la variable de entorno para OpenAI API Key
# https://platform.openai.com/account/api-keys
# os.environ['OPENAI_API_KEY'] = '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'  
# <-- Reemplaza por tu clave real
import sys
import asyncio
import logging
from logger import setup_logging


# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.api import api_bp, initialize_agent

# Configurar logging
logger = setup_logging()

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Habilitar CORS para todas las rutas
CORS(app)

# Registrar blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(api_bp, url_prefix='/api/agent')

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


# Inicializar el agente al arrancar la aplicación
@app.before_request
def startup():
    """Inicializa el agente multi-LLM al arrancar la aplicación."""
    try:
        logger.info("Iniciando aplicación Multi-LLM Agent...")
        # Ejecutar inicialización del agente en un hilo separado
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(initialize_agent())
            if success:
                logger.info("✓ Aplicación iniciada correctamente")
            else:
                logger.warning("⚠ Aplicación iniciada con funcionalidad limitada")
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error en startup: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

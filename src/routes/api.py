"""
API Routes para el Multi-LLM Agent.
Proporciona endpoints REST para interactuar con el agente.
"""

import asyncio
import json
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
from logger import setup_logging
from src.models.multi_llm_agent import MultiLLMAgent


# Configurar logging
logger = setup_logging()

# Crear blueprint para la API
api_bp = Blueprint('api', __name__)

# Instancia global del agente (se inicializará al arrancar)
agent: MultiLLMAgent = None


@api_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Endpoint de verificación de salud."""
    return jsonify({
        'status': 'healthy',
        'service': 'Multi-LLM Agent API',
        'version': '1.0.0'
    })


@api_bp.route('/status', methods=['GET'])
@cross_origin()
def get_status():
    """Obtiene el estado del sistema."""
    try:
        if agent is None:
            return jsonify({
                'status': 'not_initialized',
                'message': 'Agente no inicializado'
            }), 503
        
        # Ejecutar en el loop de eventos
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status = loop.run_until_complete(agent.get_system_status())
            return jsonify(status)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error obteniendo estado: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@api_bp.route('/query', methods=['POST'])
@cross_origin()
def process_query():
    """Procesa una consulta usando el agente multi-LLM."""
    try:
        if agent is None:
            return jsonify({
                'error': 'Agente no inicializado',
                'message': 'El agente multi-LLM no está disponible'
            }), 503
        
        # Validar datos de entrada
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'Se requiere un JSON válido'
            }), 400
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({
                'error': 'Prompt requerido',
                'message': 'El campo prompt es obligatorio'
            }), 400
        
        parameters = data.get('parameters', {})
        session_id = data.get('session_id')
        
        # Procesar consulta
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                agent.process_query(prompt, parameters, session_id)
            )
            
            # Convertir respuesta a formato JSON serializable
            result = {
                'success': True,
                'response': {
                    'text': response.refined_response.text,
                    'confidence_score': response.refined_response.confidence_score,
                    'source_models': response.refined_response.source_models,
                    'refinement_methods': response.refined_response.refinement_methods
                },
                'metadata': {
                    'total_processing_time': response.total_processing_time,
                    'successful_models': response.metadata['successful_models'],
                    'failed_models': response.metadata['failed_models'],
                    'consensus_score': response.comparison_result.consensus_score,
                    'response_count': len(response.processing_result.responses)
                },
                'analysis': {
                    'clusters': response.comparison_result.clusters,
                    'outliers': response.comparison_result.outliers,
                    'quality_scores': response.comparison_result.quality_scores
                }
            }
            
            # Añadir análisis de discrepancias si está disponible
            if response.discrepancy_analysis:
                result['analysis']['discrepancies'] = {
                    'contradictions_count': len(response.discrepancy_analysis.contradictions),
                    'factual_inconsistencies_count': len(response.discrepancy_analysis.factual_inconsistencies),
                    'semantic_differences_count': len(response.discrepancy_analysis.semantic_differences)
                }
            
            return jsonify(result)
            
        finally:
            loop.close()
            
    except ValueError as e:
        return jsonify({
            'error': 'Datos inválidos',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Error procesando consulta: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@api_bp.route('/feedback', methods=['POST'])
@cross_origin()
def submit_feedback():
    """Envía retroalimentación sobre una respuesta."""
    try:
        if agent is None:
            return jsonify({
                'error': 'Agente no inicializado'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Datos inválidos'
            }), 400
        
        response_id = data.get('response_id')
        feedback = data.get('feedback', {})
        
        if not response_id:
            return jsonify({
                'error': 'response_id requerido'
            }), 400
        
        # Procesar retroalimentación
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(
                agent.add_feedback(response_id, feedback)
            )
            
            return jsonify({
                'success': success,
                'message': 'Retroalimentación recibida'
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error procesando retroalimentación: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@api_bp.route('/configure', methods=['POST'])
@cross_origin()
def configure_agent():
    """Configura las estrategias de refinamiento del agente."""
    try:
        if agent is None:
            return jsonify({
                'error': 'Agente no inicializado'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Datos inválidos'
            }), 400
        
        strategies = data.get('strategies', {})
        
        # Configurar estrategias
        agent.configure_refinement_strategies(strategies)
        
        return jsonify({
            'success': True,
            'message': 'Configuración actualizada'
        })
        
    except Exception as e:
        logger.error(f"Error configurando agente: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


@api_bp.route('/models', methods=['GET'])
@cross_origin()
def get_available_models():
    """Obtiene información sobre los modelos disponibles."""
    try:
        if agent is None:
            return jsonify({
                'error': 'Agente no inicializado'
            }), 503
        
        # Obtener estadísticas de adaptadores
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status = loop.run_until_complete(agent.get_system_status())
            adapter_stats = status.get('system_stats', {}).get('adapter_stats', {})
            
            models = []
            for model_name, stats in adapter_stats.items():
                models.append({
                    'name': model_name,
                    'available': stats.get('is_available', False),
                    'total_requests': stats.get('total_requests', 0),
                    'error_rate': stats.get('error_rate', 0),
                    'average_latency': stats.get('average_latency', 0)
                })
            
            return jsonify({
                'models': models,
                'total_models': len(models),
                'available_models': sum(1 for m in models if m['available'])
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500


# Función para inicializar el agente
async def initialize_agent(config: dict = None):
    """Inicializa el agente multi-LLM."""
    global agent
    
    try:
        logger.info("Inicializando agente multi-LLM...")
        
        # Configuración por defecto: SIEMPRE incluye OpenAI y Hugging Face
        default_config = {
            'timeout': 30.0,
            'enable_discrepancy_analysis': True,
            'min_responses_required': 1,
            'llms': [
                {
                    'type': 'openai',
                    'model_name': 'gpt-3.5-turbo',
                    'config': {}
                },
                {
                    'type': 'huggingface',
                    'model_name': 'distilgpt2',
                    'config': {
                        'device': 'cpu',
                        'max_length': 128,
                        'use_pipeline': True
                    }
                }
            ]
        }
        
        # Usar configuración personalizada si se proporciona
        agent_config = config or default_config
        
        agent = MultiLLMAgent(agent_config)
        success = await agent.initialize()
        
        if success:
            logger.info("Agente multi-LLM inicializado correctamente")
        else:
            logger.error("Error inicializando agente multi-LLM")
            agent = None
        
        return success
        
    except Exception as e:
        logger.error(f"Error en initialize_agent: {str(e)}")
        agent = None
        return False


def get_agent():
    """Obtiene la instancia del agente."""
    return agent


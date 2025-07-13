"""
Agente Principal Multi-LLM.
Integra todos los componentes para proporcionar respuestas refinadas y de alta calidad.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..llm_adapter import BaseLLMAdapter, LLMResponse
from ..openai_adapter import OpenAIAdapter
from ..huggingface_adapter import HuggingFaceAdapter
from ..request_manager import RequestManager, ProcessingResult
from ..response_comparator import ResponseComparator, ComparisonResult, DiscrepancyAnalysis
from ..response_refiner import ResponseRefiner, RefinedResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Respuesta final del agente multi-LLM."""
    refined_response: RefinedResponse
    comparison_result: ComparisonResult
    discrepancy_analysis: Optional[DiscrepancyAnalysis]
    processing_result: ProcessingResult
    total_processing_time: float
    metadata: Dict[str, Any]


class MultiLLMAgent:
    """Agente principal que coordina múltiples LLM para generar respuestas mejoradas."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.adapters: List[BaseLLMAdapter] = []
        self.request_manager: Optional[RequestManager] = None
        self.response_comparator: Optional[ResponseComparator] = None
        self.response_refiner: Optional[ResponseRefiner] = None
        self.is_initialized = False
        
        # Configuración por defecto
        self.timeout = self.config.get('timeout', 30.0)
        self.enable_discrepancy_analysis = self.config.get('enable_discrepancy_analysis', True)
        self.min_responses_required = self.config.get('min_responses_required', 1)
        
    async def initialize(self) -> bool:
        """Inicializa el agente y todos sus componentes."""
        try:
            logger.info("Inicializando Multi-LLM Agent...")
            
            # 1. Configurar adaptadores de LLM
            await self._setup_llm_adapters()
            
            # 2. Inicializar componentes principales
            self.request_manager = RequestManager(self.adapters, timeout=self.timeout)
            self.response_comparator = ResponseComparator()
            self.response_refiner = ResponseRefiner()
            
            # 3. Inicializar adaptadores
            adapter_results = await self.request_manager.initialize_adapters()
            available_adapters = sum(adapter_results.values())
            
            if available_adapters == 0:
                logger.error("No hay adaptadores de LLM disponibles")
                return False
            
            # 4. Inicializar comparador de respuestas
            comparator_initialized = await self.response_comparator.initialize()
            if not comparator_initialized:
                logger.warning("Comparador de respuestas no inicializado, funcionalidad limitada")
            
            self.is_initialized = True
            logger.info(f"Multi-LLM Agent inicializado con {available_adapters} adaptadores")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando Multi-LLM Agent: {str(e)}")
            return False
    
    async def _setup_llm_adapters(self):
        """Configura los adaptadores de LLM basándose en la configuración."""
        
        # Configuración por defecto de LLM
        default_llms = [
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
        
        # Usar configuración personalizada si está disponible
        llm_configs = self.config.get('llms', default_llms)
        
        for llm_config in llm_configs:
            try:
                adapter = self._create_adapter(llm_config)
                if adapter:
                    self.adapters.append(adapter)
                    logger.info(f"Adaptador configurado: {adapter.model_name}")
            except Exception as e:
                logger.error(f"Error configurando adaptador {llm_config}: {str(e)}")
    
    def _create_adapter(self, llm_config: Dict[str, Any]) -> Optional[BaseLLMAdapter]:
        """Crea un adaptador basándose en la configuración."""
        
        llm_type = llm_config.get('type', '').lower()
        model_name = llm_config.get('model_name', '')
        config = llm_config.get('config', {})
        
        if llm_type == 'openai':
            return OpenAIAdapter(model_name, config)
        elif llm_type == 'huggingface':
            return HuggingFaceAdapter(model_name, config)
        else:
            logger.warning(f"Tipo de LLM no soportado: {llm_type}")
            return None
    
    async def process_query(self, prompt: str, parameters: Dict[str, Any] = None,
                          session_id: Optional[str] = None) -> AgentResponse:
        """Procesa una consulta y devuelve una respuesta refinada."""
        
        if not self.is_initialized:
            raise RuntimeError("Agente no inicializado")
        
        start_time = time.time()
        
        try:
            logger.info(f"Procesando consulta: {prompt[:100]}...")
            
            # 1. Distribuir solicitud a todos los LLM
            processing_result = await self.request_manager.process_request(
                prompt, parameters, session_id
            )
            
            if len(processing_result.responses) < self.min_responses_required:
                raise RuntimeError(f"Insuficientes respuestas: {len(processing_result.responses)}")
            
            # 2. Comparar respuestas
            comparison_result = None
            discrepancy_analysis = None
            
            if len(processing_result.responses) > 1 and self.response_comparator.is_initialized:
                comparison_result = await self.response_comparator.compare_responses(
                    processing_result.responses
                )
                
                # 3. Analizar discrepancias (opcional)
                if self.enable_discrepancy_analysis:
                    discrepancy_analysis = await self.response_comparator.analyze_discrepancies(
                        processing_result.responses
                    )
            else:
                # Crear resultado de comparación básico para una sola respuesta
                comparison_result = self._create_single_response_comparison(
                    processing_result.responses[0]
                )
            
            # 4. Refinar respuestas
            refined_response = await self.response_refiner.refine_responses(
                processing_result.responses,
                comparison_result,
                discrepancy_analysis
            )
            
            total_time = time.time() - start_time
            
            # 5. Crear respuesta final
            metadata = {
                'total_processing_time': total_time,
                'llm_processing_time': processing_result.total_latency,
                'refinement_time': total_time - processing_result.total_latency,
                'successful_models': len(processing_result.responses),
                'failed_models': len(processing_result.failed_models),
                'prompt_length': len(prompt)
            }
            
            logger.info(f"Consulta procesada en {total_time:.2f}s con confianza {refined_response.confidence_score:.2f}")
            
            return AgentResponse(
                refined_response=refined_response,
                comparison_result=comparison_result,
                discrepancy_analysis=discrepancy_analysis,
                processing_result=processing_result,
                total_processing_time=total_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {str(e)}")
            raise
    
    def _create_single_response_comparison(self, response: LLMResponse) -> ComparisonResult:
        """Crea un resultado de comparación básico para una sola respuesta."""
        import numpy as np
        
        return ComparisonResult(
            similarity_matrix=np.array([[1.0]]),
            clusters=[[0]],
            consensus_score=1.0,
            outliers=[],
            quality_scores={f"response_0_{response.model_name}": 0.8},
            metadata={
                'response_count': 1,
                'model_names': [response.model_name],
                'average_similarity': 1.0,
                'similarity_std': 0.0
            }
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema."""
        
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        # Estadísticas del gestor de solicitudes
        system_stats = self.request_manager.get_system_stats()
        
        # Estado de los componentes
        component_status = {
            'request_manager': self.request_manager is not None,
            'response_comparator': self.response_comparator is not None and self.response_comparator.is_initialized,
            'response_refiner': self.response_refiner is not None,
            'total_adapters': len(self.adapters),
            'available_adapters': sum(1 for adapter in self.adapters if adapter.is_available())
        }
        
        return {
            'status': 'initialized',
            'component_status': component_status,
            'system_stats': system_stats,
            'config': {
                'timeout': self.timeout,
                'enable_discrepancy_analysis': self.enable_discrepancy_analysis,
                'min_responses_required': self.min_responses_required
            }
        }
    
    async def add_feedback(self, response_id: str, feedback: Dict[str, Any]) -> bool:
        """Añade retroalimentación del usuario para mejora continua."""
        
        # TODO: Implementar sistema de retroalimentación
        # Por ahora, solo registrar la retroalimentación
        
        logger.info(f"Retroalimentación recibida para {response_id}: {feedback}")
        
        # En una implementación completa, esto podría:
        # 1. Almacenar la retroalimentación en base de datos
        # 2. Ajustar pesos de los modelos
        # 3. Actualizar estrategias de refinamiento
        # 4. Reentrenar componentes si es necesario
        
        return True
    
    def configure_refinement_strategies(self, strategies: Dict[str, Dict[str, Any]]):
        """Configura las estrategias de refinamiento."""
        
        if self.response_refiner:
            for strategy_name, config in strategies.items():
                if strategy_name in self.response_refiner.strategies:
                    strategy = self.response_refiner.strategies[strategy_name]
                    strategy.weight = config.get('weight', strategy.weight)
                    strategy.enabled = config.get('enabled', strategy.enabled)
                    strategy.parameters.update(config.get('parameters', {}))
                    
                    logger.info(f"Estrategia {strategy_name} actualizada: peso={strategy.weight}, habilitada={strategy.enabled}")
    
    async def shutdown(self):
        """Cierra el agente y libera recursos."""
        
        logger.info("Cerrando Multi-LLM Agent...")
        
        # Limpiar recursos de adaptadores si es necesario
        for adapter in self.adapters:
            if hasattr(adapter, 'cleanup'):
                try:
                    await adapter.cleanup()
                except Exception as e:
                    logger.warning(f"Error limpiando adaptador {adapter.model_name}: {str(e)}")
        
        self.is_initialized = False
        logger.info("Multi-LLM Agent cerrado")


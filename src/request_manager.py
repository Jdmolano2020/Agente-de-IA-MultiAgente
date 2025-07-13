"""
Módulo de Gestión de Solicitudes.
Maneja la recepción, validación y distribución de solicitudes a los LLM.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
from logger import setup_logging
from .llm_adapter import BaseLLMAdapter, LLMResponse

# Configurar logging
logger = setup_logging()

@dataclass
class UserRequest:
    """Clase para encapsular una solicitud del usuario."""
    id: str
    prompt: str
    parameters: Dict[str, Any]
    timestamp: float
    session_id: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ProcessingResult:
    """Resultado del procesamiento de una solicitud."""
    request_id: str
    responses: List[LLMResponse]
    failed_models: List[str]
    total_latency: float
    timestamp: float


class RequestManager:
    """Gestor de solicitudes que distribuye consultas a múltiples LLM."""
    
    def __init__(self, adapters: List[BaseLLMAdapter], timeout: float = 30.0):
        self.adapters = {adapter.model_name: adapter for adapter in adapters}
        self.timeout = timeout
        self.active_requests: Dict[str, UserRequest] = {}
        self.request_history: List[ProcessingResult] = []
        
    async def initialize_adapters(self) -> Dict[str, bool]:
        """Inicializa todos los adaptadores."""
        logger.info("Inicializando adaptadores de LLM...")
        
        initialization_results = {}
        tasks = []
        
        for name, adapter in self.adapters.items():
            if 'huggingface' in name.lower() or 'distilgpt2' in name.lower():
                logger.info(f"[FORZADO] Inicializando adaptador Hugging Face: {name}")
            else:
                logger.info(f"Inicializando adaptador: {name}")
            task = asyncio.create_task(adapter.initialize())
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                result = await task
                initialization_results[name] = result
                if result:
                    logger.info(f"✓ {name}: Inicializado correctamente")
                else:
                    logger.warning(f"✗ {name}: Falló la inicialización")
            except Exception as e:
                initialization_results[name] = False
                logger.error(f"✗ {name}: Error en inicialización: {str(e)}")
        
        available_count = sum(initialization_results.values())
        logger.info(f"Adaptadores disponibles: {available_count}/{len(self.adapters)}")
        
        return initialization_results
    
    def validate_request(self, prompt: str, parameters: Dict[str, Any] = None) -> bool:
        """Valida una solicitud del usuario."""
        if not prompt or not isinstance(prompt, str):
            return False
        
        if len(prompt.strip()) == 0:
            return False
        
        if len(prompt) > 10000:  # Límite de longitud
            return False
        
        return True
    
    async def process_request(self, prompt: str, parameters: Dict[str, Any] = None, 
                            session_id: Optional[str] = None) -> ProcessingResult:
        """Procesa una solicitud distribuyéndola a todos los LLM disponibles."""
        
        # Validar solicitud
        if not self.validate_request(prompt, parameters):
            raise ValueError("Solicitud inválida")
        
        # Crear solicitud
        request = UserRequest(
            id=str(uuid.uuid4()),
            prompt=prompt,
            parameters=parameters or {},
            timestamp=time.time(),
            session_id=session_id
        )
        
        self.active_requests[request.id] = request
        
        try:
            result = await self._distribute_to_llms(request)
            self.request_history.append(result)
            return result
            
        finally:
            # Limpiar solicitud activa
            self.active_requests.pop(request.id, None)
    
    async def _distribute_to_llms(self, request: UserRequest) -> ProcessingResult:
        """Distribuye la solicitud a todos los LLM disponibles en paralelo."""
        start_time = time.time()
        
        # Filtrar adaptadores disponibles
        available_adapters = {
            name: adapter for name, adapter in self.adapters.items()
            if adapter.is_available()
        }
        
        if not available_adapters:
            raise RuntimeError("No hay adaptadores de LLM disponibles")
        
        logger.info(f"Distribuyendo solicitud {request.id} a {len(available_adapters)} LLM")
        
        # Crear tareas para cada LLM
        tasks = []
        for name, adapter in available_adapters.items():
            task = asyncio.create_task(
                adapter.safe_generate_response(
                    request.prompt,
                    timeout=self.timeout,
                    **request.parameters
                )
            )
            tasks.append((name, task))
        
        # Esperar a que todas las tareas terminen
        responses = []
        failed_models = []
        
        for name, task in tasks:
            try:
                response = await task
                if response is not None:
                    responses.append(response)
                    logger.info(f"✓ {name}: Respuesta recibida")
                else:
                    failed_models.append(name)
                    logger.warning(f"✗ {name}: No se recibió respuesta")
                    
            except Exception as e:
                failed_models.append(name)
                logger.error(f"✗ {name}: Error: {str(e)}")
        
        total_latency = time.time() - start_time
        
        logger.info(f"Solicitud {request.id} completada: {len(responses)} respuestas, "
                   f"{len(failed_models)} fallos, {total_latency:.2f}s")
        
        return ProcessingResult(
            request_id=request.id,
            responses=responses,
            failed_models=failed_models,
            total_latency=total_latency,
            timestamp=time.time()
        )
    
    def get_adapter_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estadísticas de todos los adaptadores."""
        return {name: adapter.get_stats() for name, adapter in self.adapters.items()}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema."""
        total_requests = len(self.request_history)
        
        if total_requests == 0:
            return {
                'total_requests': 0,
                'average_latency': 0,
                'success_rate': 0,
                'adapter_stats': self.get_adapter_stats()
            }
        
        total_latency = sum(result.total_latency for result in self.request_history)
        successful_requests = sum(
            1 for result in self.request_history if len(result.responses) > 0
        )
        
        return {
            'total_requests': total_requests,
            'average_latency': total_latency / total_requests,
            'success_rate': successful_requests / total_requests,
            'active_requests': len(self.active_requests),
            'adapter_stats': self.get_adapter_stats()
        }


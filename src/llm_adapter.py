"""
Módulo base para los adaptadores de LLM.
Define la interfaz común que deben implementar todos los adaptadores.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import time
from logger import setup_logging


# Configurar logging
logger = setup_logging()


class LLMResponse:
    """Clase para encapsular la respuesta de un LLM."""
    
    def __init__(self, text: str, model_name: str, latency: float, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.model_name = model_name
        self.latency = latency
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la respuesta a un diccionario."""
        return {
            'text': self.text,
            'model_name': self.model_name,
            'latency': self.latency,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class BaseLLMAdapter(ABC):
    """Clase base abstracta para todos los adaptadores de LLM."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.is_initialized = False
        self.error_count = 0
        self.total_requests = 0
        self.total_latency = 0.0
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Inicializa el adaptador del LLM."""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Genera una respuesta usando el LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el LLM está disponible."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del adaptador."""
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0
        return {
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.total_requests if self.total_requests > 0 else 0,
            'average_latency': avg_latency,
            'is_available': self.is_available()
        }
    
    async def safe_generate_response(self, prompt: str, timeout: float = 30.0, **kwargs) -> Optional[LLMResponse]:
        """Genera una respuesta de manera segura con manejo de errores y timeout."""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Aplicar timeout
            response = await asyncio.wait_for(
                self.generate_response(prompt, **kwargs),
                timeout=timeout
            )
            
            latency = time.time() - start_time
            self.total_latency += latency
            
            logger.info(f"{self.model_name}: Respuesta generada en {latency:.2f}s")
            return response
            
        except asyncio.TimeoutError:
            self.error_count += 1
            logger.error(f"{self.model_name}: Timeout después de {timeout}s")
            return None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"{self.model_name}: Error generando respuesta: {str(e)}")
            return None


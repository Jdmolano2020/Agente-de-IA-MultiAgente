"""
Adaptador para modelos de OpenAI (GPT).
"""

import os
import openai
from typing import Dict, Any
from .llm_adapter import BaseLLMAdapter, LLMResponse
import time
from logger import setup_logging


# Configurar logging
logger = setup_logging()


class OpenAIAdapter(BaseLLMAdapter):
    """Adaptador para modelos de OpenAI."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", config: Dict[str, Any] = None):
        super().__init__(model_name, config or {})
        self.client = None
        self.api_key = config.get('api_key') if config else None
        self.base_url = config.get('base_url') if config else None
        
    async def initialize(self) -> bool:
        """Inicializa el cliente de OpenAI."""
        try:
            # Usar la clave de API del entorno o la configuración
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            base_url = self.base_url or os.getenv('OPENAI_API_BASE')
            
            if not api_key:
                logger.warning(f"{self.model_name}: No se encontró API key de OpenAI")
                return False
            
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # Probar la conexión
            await self.client.models.list()
            self.is_initialized = True
            logger.info(f"{self.model_name}: Inicializado correctamente con OpenAI")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name}: Error en inicialización con OpenAI: {str(e)}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Genera una respuesta usando OpenAI."""
        if not self.is_initialized:
            raise RuntimeError(f"{self.model_name}: Adaptador no inicializado")
        
        start_time = time.time()
        
        try:
            # Configuración por defecto
            params = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7),
            }
            
            # Actualizar con parámetros adicionales
            params.update(kwargs.get('openai_params', {}))
            
            response = await self.client.chat.completions.create(**params)
            
            latency = time.time() - start_time
            
            text = response.choices[0].message.content
            metadata = {
                'usage': response.usage.model_dump() if response.usage else {},
                'finish_reason': response.choices[0].finish_reason,
                'model': response.model
            }
            
            return LLMResponse(
                text=text,
                model_name=self.model_name,
                latency=latency,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"{self.model_name}: Error generando respuesta: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Verifica si el adaptador está disponible."""
        return self.is_initialized and self.client is not None


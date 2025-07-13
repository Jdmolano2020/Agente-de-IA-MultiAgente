"""
Adaptador para modelos de Hugging Face.
Soporta modelos como BLOOM, Falcon, LLaMA, etc.
"""

import asyncio
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from .llm_adapter import BaseLLMAdapter, LLMResponse
import time
from logger import setup_logging


# Configurar logging
logger = setup_logging()

class HuggingFaceAdapter(BaseLLMAdapter):
    """Adaptador para modelos de Hugging Face."""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config or {})
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = config.get('device', 'auto') if config else 'auto'
        self.max_length = config.get('max_length', 512) if config else 512
        self.use_pipeline = config.get('use_pipeline', True) if config else True
        
    async def initialize(self) -> bool:
        """Inicializa el modelo de Hugging Face."""
        try:
            logger.info(f"{self.model_name}: Iniciando carga del modelo Hugging Face.")
            
            # Ejecutar la carga en un hilo separado para no bloquear
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            
            self.is_initialized = True
            logger.info(f"{self.model_name}: Modelo cargado correctamente Hugging Face.")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name}: Error cargando modelo Hugging Face.: {str(e)}")
            return False
    
    def _load_model(self):
        """Carga el modelo de manera síncrona."""
        try:
            if self.use_pipeline:
                # Usar pipeline para simplicidad
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
            else:
                # Cargar tokenizer y modelo por separado
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                
                # Configurar pad_token si no existe
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
        except Exception as e:
            logger.error(f"{self.model_name}: Error en _load_model: {str(e)}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Genera una respuesta usando el modelo de Hugging Face."""
        if not self.is_initialized:
            raise RuntimeError(f"{self.model_name}: Adaptador no inicializado de Hugging Face")
        
        start_time = time.time()
        
        try:
            # Ejecutar la generación en un hilo separado
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._generate_text, prompt, kwargs)
            
            latency = time.time() - start_time
            
            metadata = {
                'prompt_length': len(prompt),
                'response_length': len(text),
                'device': str(self.device)
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
    
    def _generate_text(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Genera texto de manera síncrona."""
        try:
            if self.use_pipeline and self.pipeline:
                # Usar pipeline
                max_new_tokens = kwargs.get('max_new_tokens', 200)
                temperature = kwargs.get('temperature', 0.7)
                do_sample = kwargs.get('do_sample', True)
                
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    return_full_text=False,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                return outputs[0]['generated_text'].strip()
                
            else:
                # Usar tokenizer y modelo directamente
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                
                if torch.cuda.is_available() and hasattr(self.model, 'device'):
                    inputs = inputs.to(self.model.device)
                
                max_new_tokens = kwargs.get('max_new_tokens', 200)
                temperature = kwargs.get('temperature', 0.7)
                do_sample = kwargs.get('do_sample', True)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decodificar solo la parte nueva
                new_tokens = outputs[0][inputs.shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                return response.strip()
                
        except Exception as e:
            logger.error(f"{self.model_name}: Error en _generate_text: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Verifica si el adaptador está disponible."""
        if self.use_pipeline:
            return self.is_initialized and self.pipeline is not None
        else:
            return self.is_initialized and self.tokenizer is not None and self.model is not None


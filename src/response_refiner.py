"""
Módulo de Refinamiento de Respuestas.
Implementa algoritmos para fusionar, corregir y mejorar respuestas de múltiples LLM.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from logger import setup_logging
from collections import Counter

from .llm_adapter import LLMResponse
from .response_comparator import ComparisonResult, DiscrepancyAnalysis

# Configurar logging
logger = setup_logging()


@dataclass
class RefinedResponse:
    """Respuesta refinada y mejorada."""
    text: str
    confidence_score: float
    source_models: List[str]
    refinement_methods: List[str]
    metadata: Dict[str, Any]
    original_responses: List[LLMResponse]


@dataclass
class RefinementStrategy:
    """Estrategia de refinamiento."""
    name: str
    weight: float
    enabled: bool
    parameters: Dict[str, Any]


class ResponseRefiner:
    """Refinador de respuestas que mejora la calidad mediante fusión inteligente."""
    
    def __init__(self):
        self.strategies = {
            'consensus_voting': RefinementStrategy(
                name='consensus_voting',
                weight=0.3,
                enabled=True,
                parameters={'threshold': 0.6}
            ),
            'quality_weighted': RefinementStrategy(
                name='quality_weighted',
                weight=0.25,
                enabled=True,
                parameters={'min_quality': 0.3}
            ),
            'length_normalization': RefinementStrategy(
                name='length_normalization',
                weight=0.15,
                enabled=True,
                parameters={'target_length_range': (100, 1000)}
            ),
            'semantic_fusion': RefinementStrategy(
                name='semantic_fusion',
                weight=0.2,
                enabled=True,
                parameters={'similarity_threshold': 0.7}
            ),
            'error_correction': RefinementStrategy(
                name='error_correction',
                weight=0.1,
                enabled=True,
                parameters={'confidence_threshold': 0.5}
            )
        }
    
    async def refine_responses(self, responses: List[LLMResponse], 
                             comparison_result: ComparisonResult,
                             discrepancy_analysis: Optional[DiscrepancyAnalysis] = None) -> RefinedResponse:
        """Refina múltiples respuestas en una respuesta mejorada."""
        
        if not responses:
            raise ValueError("No hay respuestas para refinar")
        
        if len(responses) == 1:
            return self._create_single_response_result(responses[0])
        
        logger.info(f"Refinando {len(responses)} respuestas")
        
        # Aplicar estrategias de refinamiento
        refinement_results = {}
        applied_methods = []
        
        # 1. Votación por consenso
        if self.strategies['consensus_voting'].enabled:
            consensus_result = await self._apply_consensus_voting(responses, comparison_result)
            refinement_results['consensus'] = consensus_result
            applied_methods.append('consensus_voting')
        
        # 2. Ponderación por calidad
        if self.strategies['quality_weighted'].enabled:
            quality_result = await self._apply_quality_weighting(responses, comparison_result)
            refinement_results['quality'] = quality_result
            applied_methods.append('quality_weighted')
        
        # 3. Fusión semántica
        if self.strategies['semantic_fusion'].enabled:
            semantic_result = await self._apply_semantic_fusion(responses, comparison_result)
            refinement_results['semantic'] = semantic_result
            applied_methods.append('semantic_fusion')
        
        # 4. Corrección de errores
        if self.strategies['error_correction'].enabled and discrepancy_analysis:
            error_result = await self._apply_error_correction(responses, discrepancy_analysis)
            refinement_results['error_correction'] = error_result
            applied_methods.append('error_correction')
        
        # Combinar resultados de todas las estrategias
        final_result = await self._combine_refinement_results(
            responses, refinement_results, comparison_result
        )
        
        # Aplicar normalización de longitud si es necesario
        if self.strategies['length_normalization'].enabled:
            final_result = await self._apply_length_normalization(final_result)
            applied_methods.append('length_normalization')
        
        # Calcular puntuación de confianza final
        confidence_score = self._calculate_final_confidence(
            final_result, comparison_result, refinement_results
        )
        
        # Crear metadatos
        metadata = {
            'original_count': len(responses),
            'consensus_score': comparison_result.consensus_score,
            'refinement_strategies': applied_methods,
            'strategy_weights': {name: strategy.weight for name, strategy in self.strategies.items()},
            'quality_scores': comparison_result.quality_scores,
            'clusters': comparison_result.clusters,
            'outliers': comparison_result.outliers
        }
        
        return RefinedResponse(
            text=final_result,
            confidence_score=confidence_score,
            source_models=[r.model_name for r in responses],
            refinement_methods=applied_methods,
            metadata=metadata,
            original_responses=responses
        )
    
    def _create_single_response_result(self, response: LLMResponse) -> RefinedResponse:
        """Crea un resultado para una sola respuesta."""
        return RefinedResponse(
            text=response.text,
            confidence_score=0.8,  # Confianza moderada para respuesta única
            source_models=[response.model_name],
            refinement_methods=['single_response'],
            metadata={'single_response': True},
            original_responses=[response]
        )
    
    async def _apply_consensus_voting(self, responses: List[LLMResponse], 
                                    comparison_result: ComparisonResult) -> str:
        """Aplica votación por consenso basada en clusters."""
        
        # Encontrar el cluster más grande (mayor consenso)
        largest_cluster = max(comparison_result.clusters, key=len)
        
        if len(largest_cluster) == 1:
            # No hay consenso claro, usar la respuesta de mejor calidad
            quality_scores = comparison_result.quality_scores
            best_idx = max(range(len(responses)), 
                          key=lambda i: list(quality_scores.values())[i])
            return responses[best_idx].text
        
        # Fusionar respuestas del cluster de consenso
        cluster_responses = [responses[i] for i in largest_cluster]
        
        # Estrategia simple: usar la respuesta más larga del cluster
        # (asumiendo que más detalle = mejor)
        best_response = max(cluster_responses, key=lambda r: len(r.text))
        
        return best_response.text
    
    async def _apply_quality_weighting(self, responses: List[LLMResponse], 
                                     comparison_result: ComparisonResult) -> str:
        """Aplica ponderación basada en calidad."""
        
        quality_scores = list(comparison_result.quality_scores.values())
        
        # Encontrar la respuesta con mejor puntuación de calidad
        best_idx = np.argmax(quality_scores)
        best_response = responses[best_idx]
        
        # Si la calidad es muy baja, intentar mejorar combinando con otras
        if quality_scores[best_idx] < 0.5:
            # Combinar con la segunda mejor
            sorted_indices = np.argsort(quality_scores)[::-1]
            if len(sorted_indices) > 1:
                second_best = responses[sorted_indices[1]]
                # Fusión simple: tomar la primera mitad de la mejor y la segunda mitad de la segunda
                mid_point = len(best_response.text) // 2
                combined_text = best_response.text[:mid_point] + " " + second_best.text[mid_point:]
                return combined_text
        
        return best_response.text
    
    async def _apply_semantic_fusion(self, responses: List[LLMResponse], 
                                   comparison_result: ComparisonResult) -> str:
        """Aplica fusión semántica de respuestas similares."""
        
        # Agrupar respuestas por clusters semánticos
        clusters = comparison_result.clusters
        
        if len(clusters) == 1:
            # Todas las respuestas son similares, fusionar inteligentemente
            return await self._fuse_similar_responses([responses[i] for i in clusters[0]])
        
        # Múltiples clusters, elegir el mejor cluster y fusionar
        cluster_scores = []
        for cluster in clusters:
            cluster_responses = [responses[i] for i in cluster]
            avg_quality = np.mean([
                list(comparison_result.quality_scores.values())[i] for i in cluster
            ])
            cluster_scores.append((len(cluster), avg_quality))
        
        # Elegir cluster con mejor combinación de tamaño y calidad
        best_cluster_idx = max(range(len(clusters)), 
                              key=lambda i: cluster_scores[i][0] * cluster_scores[i][1])
        
        best_cluster = clusters[best_cluster_idx]
        best_cluster_responses = [responses[i] for i in best_cluster]
        
        return await self._fuse_similar_responses(best_cluster_responses)
    
    async def _fuse_similar_responses(self, responses: List[LLMResponse]) -> str:
        """Fusiona respuestas semánticamente similares."""
        
        if len(responses) == 1:
            return responses[0].text
        
        # Estrategia de fusión: combinar información única de cada respuesta
        texts = [r.text for r in responses]
        
        # Dividir en oraciones
        all_sentences = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            all_sentences.extend([s.strip() for s in sentences if s.strip()])
        
        # Eliminar duplicados manteniendo orden
        unique_sentences = []
        seen = set()
        for sentence in all_sentences:
            sentence_lower = sentence.lower()
            if sentence_lower not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence_lower)
        
        # Reconstruir texto fusionado
        fused_text = '. '.join(unique_sentences)
        if fused_text and not fused_text.endswith('.'):
            fused_text += '.'
        
        return fused_text
    
    async def _apply_error_correction(self, responses: List[LLMResponse], 
                                    discrepancy_analysis: DiscrepancyAnalysis) -> str:
        """Aplica corrección de errores basada en análisis de discrepancias."""
        
        # Identificar respuestas con menos contradicciones
        contradiction_counts = Counter()
        for contradiction in discrepancy_analysis.contradictions:
            contradiction_counts[contradiction[0]] += 1
            contradiction_counts[contradiction[1]] += 1
        
        # Elegir respuesta con menos contradicciones
        best_idx = 0
        min_contradictions = float('inf')
        
        for i in range(len(responses)):
            contradictions = contradiction_counts.get(i, 0)
            if contradictions < min_contradictions:
                min_contradictions = contradictions
                best_idx = i
        
        return responses[best_idx].text
    
    async def _combine_refinement_results(self, responses: List[LLMResponse],
                                        refinement_results: Dict[str, str],
                                        comparison_result: ComparisonResult) -> str:
        """Combina los resultados de diferentes estrategias de refinamiento."""
        
        if not refinement_results:
            # Fallback: usar la respuesta de mejor calidad
            quality_scores = list(comparison_result.quality_scores.values())
            best_idx = np.argmax(quality_scores)
            return responses[best_idx].text
        
        # Si solo hay un resultado, usarlo
        if len(refinement_results) == 1:
            return list(refinement_results.values())[0]
        
        # Estrategia de combinación: votar por la respuesta más común
        # o usar la del método con mayor peso
        
        method_weights = {}
        for method_name in refinement_results.keys():
            if method_name == 'consensus':
                method_weights[method_name] = self.strategies['consensus_voting'].weight
            elif method_name == 'quality':
                method_weights[method_name] = self.strategies['quality_weighted'].weight
            elif method_name == 'semantic':
                method_weights[method_name] = self.strategies['semantic_fusion'].weight
            elif method_name == 'error_correction':
                method_weights[method_name] = self.strategies['error_correction'].weight
            else:
                method_weights[method_name] = 0.1
        
        # Elegir el resultado del método con mayor peso
        best_method = max(method_weights.keys(), key=lambda k: method_weights[k])
        return refinement_results[best_method]
    
    async def _apply_length_normalization(self, text: str) -> str:
        """Aplica normalización de longitud al texto."""
        
        target_range = self.strategies['length_normalization'].parameters['target_length_range']
        min_length, max_length = target_range
        
        current_length = len(text)
        
        if current_length < min_length:
            # Texto muy corto, no modificar (podría ser una respuesta válida corta)
            return text
        
        if current_length > max_length:
            # Texto muy largo, truncar inteligentemente
            sentences = re.split(r'[.!?]+', text)
            
            # Mantener oraciones completas hasta llegar cerca del límite
            truncated_sentences = []
            current_len = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if current_len + len(sentence) + 1 <= max_length:
                    truncated_sentences.append(sentence)
                    current_len += len(sentence) + 1
                else:
                    break
            
            if truncated_sentences:
                result = '. '.join(truncated_sentences)
                if not result.endswith('.'):
                    result += '.'
                return result
        
        return text
    
    def _calculate_final_confidence(self, final_text: str, 
                                  comparison_result: ComparisonResult,
                                  refinement_results: Dict[str, str]) -> float:
        """Calcula la puntuación de confianza final."""
        
        confidence = 0.0
        
        # Factor 1: Consenso entre respuestas originales (30%)
        consensus_factor = comparison_result.consensus_score * 0.3
        confidence += consensus_factor
        
        # Factor 2: Calidad promedio de respuestas originales (25%)
        avg_quality = np.mean(list(comparison_result.quality_scores.values()))
        quality_factor = avg_quality * 0.25
        confidence += quality_factor
        
        # Factor 3: Número de métodos de refinamiento aplicados (20%)
        method_factor = min(len(refinement_results) / 4, 1.0) * 0.2
        confidence += method_factor
        
        # Factor 4: Longitud apropiada del resultado final (15%)
        length = len(final_text)
        if 100 <= length <= 1000:
            length_factor = 0.15
        elif 50 <= length <= 1500:
            length_factor = 0.1
        else:
            length_factor = 0.05
        confidence += length_factor
        
        # Factor 5: Ausencia de outliers (10%)
        outlier_factor = (1 - len(comparison_result.outliers) / len(comparison_result.metadata['response_count'])) * 0.1
        confidence += outlier_factor
        
        return min(confidence, 1.0)


"""
Módulo de Comparación de Respuestas.
Implementa algoritmos para comparar, analizar y detectar discrepancias entre respuestas de LLM.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
from logger import setup_logging


from .llm_adapter import LLMResponse

# Configurar logging
logger = setup_logging()


@dataclass
class ComparisonResult:
    """Resultado de la comparación entre respuestas."""
    similarity_matrix: np.ndarray
    clusters: List[List[int]]
    consensus_score: float
    outliers: List[int]
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class DiscrepancyAnalysis:
    """Análisis de discrepancias entre respuestas."""
    contradictions: List[Tuple[int, int, str]]
    factual_inconsistencies: List[Tuple[int, str]]
    semantic_differences: List[Tuple[int, int, float]]
    length_variations: Dict[str, Any]


class ResponseComparator:
    """Comparador de respuestas que analiza similitudes y discrepancias."""
    
    def __init__(self, similarity_model: str = "all-MiniLM-L6-v2"):
        self.similarity_model_name = similarity_model
        self.similarity_model = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Inicializa el modelo de similitud semántica."""
        try:
            logger.info(f"Cargando modelo de similitud: {self.similarity_model_name}")
            
            # Cargar en un hilo separado para no bloquear
            loop = asyncio.get_event_loop()
            self.similarity_model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.similarity_model_name)
            )
            
            self.is_initialized = True
            logger.info("Modelo de similitud cargado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo de similitud: {str(e)}")
            return False
    
    async def compare_responses(self, responses: List[LLMResponse]) -> ComparisonResult:
        """Compara múltiples respuestas y genera un análisis completo."""
        if not self.is_initialized:
            raise RuntimeError("Comparador no inicializado")
        
        if len(responses) < 2:
            raise ValueError("Se necesitan al menos 2 respuestas para comparar")
        
        logger.info(f"Comparando {len(responses)} respuestas")
        
        # Extraer textos
        texts = [response.text for response in responses]
        
        # Calcular similitudes semánticas
        similarity_matrix = await self._calculate_semantic_similarity(texts)
        
        # Detectar clusters de respuestas similares
        clusters = self._detect_clusters(similarity_matrix)
        
        # Calcular puntuación de consenso
        consensus_score = self._calculate_consensus_score(similarity_matrix)
        
        # Identificar outliers
        outliers = self._identify_outliers(similarity_matrix)
        
        # Calcular puntuaciones de calidad
        quality_scores = await self._calculate_quality_scores(responses)
        
        # Metadatos adicionales
        metadata = {
            'response_count': len(responses),
            'model_names': [r.model_name for r in responses],
            'average_similarity': float(np.mean(similarity_matrix)),
            'similarity_std': float(np.std(similarity_matrix))
        }
        
        return ComparisonResult(
            similarity_matrix=similarity_matrix,
            clusters=clusters,
            consensus_score=consensus_score,
            outliers=outliers,
            quality_scores=quality_scores,
            metadata=metadata
        )
    
    async def analyze_discrepancies(self, responses: List[LLMResponse]) -> DiscrepancyAnalysis:
        """Analiza discrepancias específicas entre respuestas."""
        texts = [response.text for response in responses]
        
        # Detectar contradicciones
        contradictions = await self._detect_contradictions(texts)
        
        # Detectar inconsistencias factuales
        factual_inconsistencies = await self._detect_factual_inconsistencies(texts)
        
        # Calcular diferencias semánticas
        semantic_differences = await self._calculate_semantic_differences(texts)
        
        # Analizar variaciones de longitud
        length_variations = self._analyze_length_variations(texts)
        
        return DiscrepancyAnalysis(
            contradictions=contradictions,
            factual_inconsistencies=factual_inconsistencies,
            semantic_differences=semantic_differences,
            length_variations=length_variations
        )
    
    async def _calculate_semantic_similarity(self, texts: List[str]) -> np.ndarray:
        """Calcula la matriz de similitud semántica entre textos."""
        loop = asyncio.get_event_loop()
        
        # Generar embeddings
        embeddings = await loop.run_in_executor(
            None,
            self.similarity_model.encode,
            texts
        )
        
        # Calcular similitud coseno
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def _detect_clusters(self, similarity_matrix: np.ndarray, threshold: float = 0.7) -> List[List[int]]:
        """Detecta clusters de respuestas similares."""
        n_responses = similarity_matrix.shape[0]
        
        if n_responses <= 2:
            return [[i] for i in range(n_responses)]
        
        # Usar clustering basado en similitud
        # Convertir similitud a distancia
        distance_matrix = 1 - similarity_matrix
        
        # Determinar número óptimo de clusters
        max_clusters = min(n_responses, 4)
        
        try:
            # Usar K-means en el espacio de embeddings
            # Para esto necesitamos reconstruir los embeddings aproximados
            # Usaremos un enfoque simplificado basado en threshold
            
            clusters = []
            assigned = set()
            
            for i in range(n_responses):
                if i in assigned:
                    continue
                
                cluster = [i]
                assigned.add(i)
                
                for j in range(i + 1, n_responses):
                    if j not in assigned and similarity_matrix[i, j] >= threshold:
                        cluster.append(j)
                        assigned.add(j)
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Error en clustering: {str(e)}, usando clusters individuales")
            return [[i] for i in range(n_responses)]
    
    def _calculate_consensus_score(self, similarity_matrix: np.ndarray) -> float:
        """Calcula una puntuación de consenso basada en la similitud promedio."""
        # Excluir la diagonal (similitud consigo mismo)
        n = similarity_matrix.shape[0]
        if n <= 1:
            return 1.0
        
        # Suma de similitudes excluyendo diagonal
        total_similarity = np.sum(similarity_matrix) - np.trace(similarity_matrix)
        max_possible = n * (n - 1)  # n*(n-1) comparaciones
        
        consensus_score = total_similarity / max_possible if max_possible > 0 else 0.0
        
        return float(consensus_score)
    
    def _identify_outliers(self, similarity_matrix: np.ndarray, threshold: float = 0.3) -> List[int]:
        """Identifica respuestas que son outliers (muy diferentes del resto)."""
        n = similarity_matrix.shape[0]
        outliers = []
        
        for i in range(n):
            # Calcular similitud promedio con otras respuestas
            other_similarities = [similarity_matrix[i, j] for j in range(n) if i != j]
            
            if other_similarities:
                avg_similarity = np.mean(other_similarities)
                if avg_similarity < threshold:
                    outliers.append(i)
        
        return outliers
    
    async def _calculate_quality_scores(self, responses: List[LLMResponse]) -> Dict[str, float]:
        """Calcula puntuaciones de calidad para cada respuesta."""
        quality_scores = {}
        
        for i, response in enumerate(responses):
            score = 0.0
            
            # Factor 1: Longitud (ni muy corta ni muy larga)
            length = len(response.text)
            if 50 <= length <= 2000:
                score += 0.3
            elif length > 20:
                score += 0.1
            
            # Factor 2: Estructura (párrafos, puntuación)
            if self._has_good_structure(response.text):
                score += 0.2
            
            # Factor 3: Coherencia (sin repeticiones excesivas)
            if not self._has_excessive_repetition(response.text):
                score += 0.2
            
            # Factor 4: Latencia (respuestas más rápidas son mejores)
            if response.latency < 10.0:
                score += 0.1
            elif response.latency < 30.0:
                score += 0.05
            
            # Factor 5: Completitud (no termina abruptamente)
            if self._is_complete_response(response.text):
                score += 0.2
            
            quality_scores[f"response_{i}_{response.model_name}"] = min(score, 1.0)
        
        return quality_scores
    
    def _has_good_structure(self, text: str) -> bool:
        """Verifica si el texto tiene buena estructura."""
        # Verificar puntuación básica
        has_punctuation = any(p in text for p in '.!?')
        
        # Verificar que no sea una sola línea muy larga
        lines = text.split('\n')
        has_paragraphs = len(lines) > 1 or len(text) < 500
        
        return has_punctuation and has_paragraphs
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Detecta repeticiones excesivas en el texto."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Contar palabras repetidas consecutivas
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                consecutive_repeats += 1
        
        # Si más del 20% son repeticiones consecutivas
        return consecutive_repeats / len(words) > 0.2
    
    def _is_complete_response(self, text: str) -> bool:
        """Verifica si la respuesta parece completa."""
        text = text.strip()
        
        # Verificar que termine con puntuación apropiada
        ends_properly = text.endswith(('.', '!', '?', ':', '"', "'"))
        
        # Verificar que no termine con palabras incompletas comunes
        incomplete_endings = ['and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at']
        last_word = text.split()[-1].lower() if text.split() else ""
        not_incomplete = last_word not in incomplete_endings
        
        return ends_properly and not_incomplete
    
    async def _detect_contradictions(self, texts: List[str]) -> List[Tuple[int, int, str]]:
        """Detecta contradicciones entre respuestas."""
        contradictions = []
        
        # Palabras que indican negación o contradicción
        negation_patterns = [
            r'\b(no|not|never|none|nothing|nobody|nowhere)\b',
            r'\b(false|incorrect|wrong|untrue)\b',
            r'\b(impossible|cannot|can\'t|won\'t|wouldn\'t)\b'
        ]
        
        affirmation_patterns = [
            r'\b(yes|true|correct|right|accurate)\b',
            r'\b(always|definitely|certainly|surely)\b',
            r'\b(possible|can|will|would)\b'
        ]
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1, text2 = texts[i].lower(), texts[j].lower()
                
                # Buscar patrones contradictorios
                has_negation_1 = any(re.search(pattern, text1) for pattern in negation_patterns)
                has_affirmation_1 = any(re.search(pattern, text1) for pattern in affirmation_patterns)
                
                has_negation_2 = any(re.search(pattern, text2) for pattern in negation_patterns)
                has_affirmation_2 = any(re.search(pattern, text2) for pattern in affirmation_patterns)
                
                # Detectar contradicción simple
                if (has_negation_1 and has_affirmation_2) or (has_affirmation_1 and has_negation_2):
                    contradictions.append((i, j, "Contradicción en afirmación/negación"))
        
        return contradictions
    
    async def _detect_factual_inconsistencies(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Detecta posibles inconsistencias factuales."""
        inconsistencies = []
        
        # Patrones para detectar números, fechas, nombres
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
        
        for i, text in enumerate(texts):
            # Buscar números que podrían ser inconsistentes
            numbers = re.findall(number_pattern, text)
            dates = re.findall(date_pattern, text)
            
            # Verificaciones básicas de consistencia
            if len(numbers) > 10:  # Demasiados números podría indicar confusión
                inconsistencies.append((i, "Exceso de datos numéricos"))
            
            if len(dates) > 5:  # Demasiadas fechas
                inconsistencies.append((i, "Exceso de fechas"))
        
        return inconsistencies
    
    async def _calculate_semantic_differences(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """Calcula diferencias semánticas significativas."""
        differences = []
        
        similarity_matrix = await self._calculate_semantic_similarity(texts)
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = similarity_matrix[i, j]
                difference = 1 - similarity
                
                # Solo reportar diferencias significativas
                if difference > 0.5:
                    differences.append((i, j, float(difference)))
        
        return differences
    
    def _analyze_length_variations(self, texts: List[str]) -> Dict[str, Any]:
        """Analiza variaciones en la longitud de las respuestas."""
        lengths = [len(text) for text in texts]
        
        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'length_ratio': max(lengths) / min(lengths) if min(lengths) > 0 else float('inf'),
            'lengths': lengths
        }


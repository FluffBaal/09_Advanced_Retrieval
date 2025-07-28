"""
Ensemble retriever that combines multiple retrieval methods.
"""

from typing import List, Optional
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever


def create_ensemble_retriever(
    retrievers: List[BaseRetriever],
    weights: Optional[List[float]] = None,
    id_key: Optional[str] = None,
    c: int = 60
) -> EnsembleRetriever:
    """
    Create an ensemble retriever that combines multiple retrievers.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
    retrievers, leveraging the strengths of each approach.
    
    Args:
        retrievers: List of retrievers to combine
        weights: Optional weights for each retriever (equal weights if None)
        id_key: Key to use for document IDs
        c: Constant for RRF algorithm (default: 60)
        
    Returns:
        An EnsembleRetriever instance
    """
    if weights is None:
        # Equal weighting by default
        weights = [1.0 / len(retrievers)] * len(retrievers)
    
    ensemble_kwargs = {
        "retrievers": retrievers,
        "weights": weights,
        "c": c
    }
    
    if id_key is not None:
        ensemble_kwargs["id_key"] = id_key
    
    return EnsembleRetriever(**ensemble_kwargs)
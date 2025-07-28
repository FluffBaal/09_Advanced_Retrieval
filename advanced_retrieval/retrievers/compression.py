"""
Contextual compression retriever with reranking capabilities.
"""

from typing import Optional
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
try:
    from langchain_cohere import CohereRerank
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


def create_contextual_compression_retriever(
    base_retriever: BaseRetriever,
    model: str = "rerank-v3.5",
    top_n: Optional[int] = None,
    cohere_api_key: Optional[str] = None
) -> ContextualCompressionRetriever:
    """
    Create a contextual compression retriever using Cohere's reranking model.
    
    This retriever first fetches documents using the base retriever, then
    reranks them using Cohere's reranking model to improve relevance.
    
    Args:
        base_retriever: The base retriever to fetch initial documents
        model: Cohere rerank model to use (default: "rerank-v3.5")
        top_n: Number of documents to return after reranking
        cohere_api_key: Optional Cohere API key (uses environment variable if not provided)
        
    Returns:
        A ContextualCompressionRetriever instance
    """
    compressor_kwargs = {"model": model}
    if cohere_api_key:
        compressor_kwargs["cohere_api_key"] = cohere_api_key
    if top_n:
        compressor_kwargs["top_n"] = top_n
    
    if not COHERE_AVAILABLE:
        raise ImportError(
            "langchain-cohere is required for contextual compression. "
            "Install it with: pip install langchain-cohere"
        )
    
    compressor = CohereRerank(**compressor_kwargs)
    
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
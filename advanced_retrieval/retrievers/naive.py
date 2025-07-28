"""
Naive retriever implementation using simple cosine similarity.
"""

from typing import Optional, Dict, Any
from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def create_naive_retriever(
    vectorstore: Qdrant,
    k: int = 10,
    search_type: str = "similarity",
    mmr_lambda: float = 0.5,
    fetch_k: int = 50,
    search_kwargs: Optional[Dict[str, Any]] = None
) -> BaseRetriever:
    """
    Create a naive retriever with optional Maximum Marginal Relevance (MMR).
    
    Based on 2025 best practices, supports both similarity and MMR search.
    
    Args:
        vectorstore: The vector store to search in
        k: Number of documents to retrieve (default: 10)
        search_type: "similarity" or "mmr" (default: "similarity")
        mmr_lambda: Diversity parameter for MMR (0=max diversity, 1=max relevance)
        fetch_k: Number of documents to fetch before MMR reranking
        search_kwargs: Additional search parameters
        
    Returns:
        A LangChain retriever instance
    """
    if search_kwargs is None:
        search_kwargs = {}
    
    search_kwargs["k"] = k
    
    # Configure search type based on 2025 best practices
    if search_type == "mmr":
        # Maximum Marginal Relevance for diversity
        search_kwargs["lambda_mult"] = mmr_lambda
        search_kwargs["fetch_k"] = fetch_k
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
    else:
        # Standard similarity search
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
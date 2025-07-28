"""
BM25 retriever implementation for keyword-based search.
"""

from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def create_bm25_retriever(
    documents: List[Document],
    k: int = 10,
    **kwargs
) -> BM25Retriever:
    """
    Create a BM25 retriever for keyword-based document retrieval.
    
    BM25 is a bag-of-words retrieval function that ranks documents based on
    the query terms appearing in each document.
    
    Args:
        documents: List of documents to search through
        k: Number of documents to retrieve (default: 10)
        **kwargs: Additional parameters for BM25Retriever
        
    Returns:
        A BM25Retriever instance
    """
    retriever = BM25Retriever.from_documents(documents, **kwargs)
    retriever.k = k
    return retriever
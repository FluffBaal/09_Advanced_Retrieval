"""
Advanced Retrieval Library
A comprehensive library for advanced retrieval methods in RAG applications.
"""

from .retrievers import (
    create_naive_retriever,
    create_bm25_retriever,
    create_contextual_compression_retriever,
    create_multi_query_retriever,
    create_parent_document_retriever,
    create_ensemble_retriever,
    RetrieverFactory
)

from .chains import create_rag_chain, RAGChainFactory

from .evaluation import RagasEvaluator, evaluate_retrievers

from .utils import (
    load_documents,
    load_loan_complaints_data,
    generate_test_data
)

__version__ = "0.1.0"

__all__ = [
    # Retrievers
    "create_naive_retriever",
    "create_bm25_retriever",
    "create_contextual_compression_retriever",
    "create_multi_query_retriever",
    "create_parent_document_retriever",
    "create_ensemble_retriever",
    "RetrieverFactory",
    
    # Chains
    "create_rag_chain",
    "RAGChainFactory",
    
    # Evaluation
    "RagasEvaluator",
    "evaluate_retrievers",
    
    # Utils
    "load_documents",
    "load_loan_complaints_data",
    "generate_test_data"
]
"""
Factory class for creating different types of retrievers.
"""

from typing import Dict, Any, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Qdrant

from .naive import create_naive_retriever
from .bm25 import create_bm25_retriever
from .compression import create_contextual_compression_retriever
from .multi_query import create_multi_query_retriever
from .parent_document import create_parent_document_retriever
from .ensemble import create_ensemble_retriever
from .hype import create_hype_retriever


class RetrieverFactory:
    """Factory class for creating different types of retrievers."""
    
    def __init__(
        self,
        documents: List[Document],
        embeddings: Embeddings,
        vectorstore: Optional[Qdrant] = None,
        collection_name: str = "default_collection"
    ):
        """
        Initialize the retriever factory.
        
        Args:
            documents: List of documents for retrieval
            embeddings: Embedding model to use
            vectorstore: Optional pre-created vector store
            collection_name: Name for vector store collection
        """
        self.documents = documents
        self.embeddings = embeddings
        self.collection_name = collection_name
        
        # Create vector store if not provided
        if vectorstore is None:
            self.vectorstore = Qdrant.from_documents(
                documents,
                embeddings,
                location=":memory:",
                collection_name=collection_name
            )
        else:
            self.vectorstore = vectorstore
    
    def create_retriever(
        self,
        retriever_type: str,
        **kwargs
    ) -> BaseRetriever:
        """
        Create a retriever of the specified type.
        
        Args:
            retriever_type: Type of retriever to create
            **kwargs: Additional arguments for the specific retriever
            
        Returns:
            A retriever instance
        """
        if retriever_type == "naive":
            return self.create_naive(**kwargs)
        elif retriever_type == "bm25":
            return self.create_bm25(**kwargs)
        elif retriever_type == "compression":
            return self.create_compression(**kwargs)
        elif retriever_type == "multi_query":
            return self.create_multi_query(**kwargs)
        elif retriever_type == "parent_document":
            return self.create_parent_document(**kwargs)
        elif retriever_type == "ensemble":
            return self.create_ensemble(**kwargs)
        elif retriever_type == "hype":
            return self.create_hype(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def create_naive(self, k: int = 10, **kwargs) -> BaseRetriever:
        """Create a naive retriever."""
        return create_naive_retriever(self.vectorstore, k=k, **kwargs)
    
    def create_bm25(self, k: int = 10, **kwargs) -> BaseRetriever:
        """Create a BM25 retriever."""
        return create_bm25_retriever(self.documents, k=k, **kwargs)
    
    def create_compression(
        self,
        base_retriever: Optional[BaseRetriever] = None,
        **kwargs
    ) -> BaseRetriever:
        """Create a compression retriever."""
        if base_retriever is None:
            base_retriever = self.create_naive()
        return create_contextual_compression_retriever(base_retriever, **kwargs)
    
    def create_multi_query(
        self,
        llm: BaseLLM,
        base_retriever: Optional[BaseRetriever] = None,
        **kwargs
    ) -> BaseRetriever:
        """Create a multi-query retriever."""
        if base_retriever is None:
            base_retriever = self.create_naive()
        return create_multi_query_retriever(base_retriever, llm, **kwargs)
    
    def create_parent_document(
        self,
        chunk_size: int = 750,
        **kwargs
    ) -> BaseRetriever:
        """Create a parent document retriever."""
        return create_parent_document_retriever(
            self.documents,
            self.embeddings,
            chunk_size=chunk_size,
            **kwargs
        )
    
    def create_ensemble(
        self,
        retriever_types: Optional[List[str]] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        Create an ensemble retriever.
        
        Args:
            retriever_types: List of retriever types to include
            llm: LLM for multi-query retriever (if included)
            **kwargs: Additional arguments
        """
        if retriever_types is None:
            retriever_types = ["naive", "bm25"]
        
        retrievers = []
        for rtype in retriever_types:
            if rtype == "multi_query" and llm is None:
                raise ValueError("LLM required for multi_query retriever")
            
            if rtype == "multi_query":
                retrievers.append(self.create_multi_query(llm))
            else:
                retrievers.append(self.create_retriever(rtype))
        
        return create_ensemble_retriever(retrievers, **kwargs)
    
    def create_hype(
        self,
        llm: BaseLLM,
        num_questions: int = 3,
        k: int = 10,
        **kwargs
    ) -> BaseRetriever:
        """Create a Hypothetical Document Embedding (HyPE) retriever."""
        return create_hype_retriever(
            self.documents,
            llm,
            self.embeddings,
            num_questions=num_questions,
            k=k,
            **kwargs
        )
    
    def create_all_retrievers(
        self,
        llm: BaseLLM,
        include_ensemble: bool = True
    ) -> Dict[str, BaseRetriever]:
        """
        Create all available retrievers.
        
        Args:
            llm: LLM for multi-query retriever
            include_ensemble: Whether to include ensemble retriever
            
        Returns:
            Dictionary mapping retriever names to instances
        """
        retrievers = {
            "naive": self.create_naive(),
            "bm25": self.create_bm25(),
            "compression": self.create_compression(),
            "multi_query": self.create_multi_query(llm),
            "parent_document": self.create_parent_document()
        }
        
        if include_ensemble:
            retrievers["ensemble"] = self.create_ensemble(
                retriever_types=list(retrievers.keys()),
                llm=llm
            )
        
        return retrievers
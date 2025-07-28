"""
Parent document retriever for small-to-big retrieval strategy.
"""

from typing import List, Optional
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings


def create_parent_document_retriever(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str = "parent_documents",
    chunk_size: int = 750,
    chunk_overlap: int = 50,
    k: int = 10
) -> ParentDocumentRetriever:
    """
    Create a parent document retriever with small-to-big strategy.
    
    This retriever splits documents into small chunks for precise matching,
    but returns the full parent documents for comprehensive context.
    
    Args:
        documents: List of parent documents
        embeddings: Embedding model to use
        collection_name: Name for the Qdrant collection
        chunk_size: Size of child chunks (default: 750)
        chunk_overlap: Overlap between chunks (default: 50)
        k: Number of documents to retrieve (default: 10)
        
    Returns:
        A ParentDocumentRetriever instance
    """
    # Create vector store with a placeholder document
    # The ParentDocumentRetriever will manage the actual documents
    placeholder_doc = Document(page_content="placeholder", metadata={"type": "placeholder"})
    vectorstore = Qdrant.from_documents(
        [placeholder_doc],
        embeddings,
        location=":memory:",
        collection_name=collection_name
    )
    
    # Create child splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create document store
    store = InMemoryStore()
    
    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        k=k
    )
    
    # Add documents
    retriever.add_documents(documents)
    
    return retriever
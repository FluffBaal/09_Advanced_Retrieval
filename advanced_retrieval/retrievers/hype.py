"""
Hypothetical Document Embeddings (HyPE) retriever implementation.

Based on 2025 best practices, this retriever generates hypothetical questions
at indexing time and performs question-to-question matching during retrieval.
"""

from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Qdrant
import asyncio


class HypotheticalDocumentEmbeddingRetriever(BaseRetriever):
    """
    Hypothetical Document Embedding (HyPE) retriever.
    
    This retriever generates hypothetical questions for each document chunk
    at indexing time, then retrieves based on question-to-question similarity.
    """
    
    vectorstore: Any
    llm: BaseLLM
    embeddings: Embeddings
    num_questions: int = 3
    k: int = 10
    
    def __init__(
        self,
        documents: List[Document],
        llm: BaseLLM,
        embeddings: Embeddings,
        num_questions: int = 3,
        k: int = 10,
        collection_name: str = "hype_retriever"
    ):
        """
        Initialize the HyPE retriever.
        
        Args:
            documents: Documents to index
            llm: LLM for generating hypothetical questions
            embeddings: Embeddings model
            num_questions: Number of questions to generate per document
            k: Number of documents to retrieve
            collection_name: Name for the vector store collection
        """
        super().__init__()
        self.llm = llm
        self.embeddings = embeddings
        self.num_questions = num_questions
        self.k = k
        
        # Generate hypothetical questions and create vector store
        self.vectorstore = self._create_hype_index(documents, collection_name)
    
    def _generate_hypothetical_questions(self, document: Document) -> List[str]:
        """Generate hypothetical questions for a document."""
        prompt = f"""Given the following document content, generate {self.num_questions} diverse questions that this document would answer well.

Document content:
{document.page_content[:1000]}

Generate exactly {self.num_questions} questions, one per line:"""
        
        try:
            response = self.llm.invoke(prompt)
            questions = [q.strip() for q in response.content.split('\n') if q.strip()]
            return questions[:self.num_questions]
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Fallback to simple extraction
            return [f"What information is provided about {document.page_content[:50]}...?"]
    
    def _create_hype_index(
        self,
        documents: List[Document],
        collection_name: str
    ) -> Qdrant:
        """Create vector store with hypothetical questions."""
        hype_documents = []
        
        print("Generating hypothetical questions for documents...")
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"Processing document {i}/{len(documents)}")
            
            # Generate hypothetical questions
            questions = self._generate_hypothetical_questions(doc)
            
            # Create new documents with questions as content
            for question in questions:
                hype_doc = Document(
                    page_content=question,
                    metadata={
                        **doc.metadata,
                        "original_content": doc.page_content,
                        "is_hypothetical": True,
                        "source_doc_index": i
                    }
                )
                hype_documents.append(hype_doc)
        
        print(f"Created {len(hype_documents)} hypothetical question documents")
        
        # Create vector store
        vectorstore = Qdrant.from_documents(
            hype_documents,
            self.embeddings,
            location=":memory:",
            collection_name=collection_name
        )
        
        return vectorstore
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents using question-to-question matching."""
        # Search for similar questions
        similar_questions = self.vectorstore.similarity_search(query, k=self.k * 2)
        
        # Deduplicate and get original documents
        seen_indices = set()
        relevant_docs = []
        
        for q_doc in similar_questions:
            source_index = q_doc.metadata.get("source_doc_index")
            if source_index not in seen_indices:
                seen_indices.add(source_index)
                # Reconstruct original document
                original_doc = Document(
                    page_content=q_doc.metadata["original_content"],
                    metadata={k: v for k, v in q_doc.metadata.items() 
                             if k not in ["original_content", "is_hypothetical", "source_doc_index"]}
                )
                relevant_docs.append(original_doc)
                
                if len(relevant_docs) >= self.k:
                    break
        
        return relevant_docs
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Async version of get_relevant_documents."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_relevant_documents, query
        )


def create_hype_retriever(
    documents: List[Document],
    llm: BaseLLM,
    embeddings: Embeddings,
    num_questions: int = 3,
    k: int = 10
) -> HypotheticalDocumentEmbeddingRetriever:
    """
    Create a Hypothetical Document Embedding retriever.
    
    This advanced retriever generates hypothetical questions for each document
    at indexing time, enabling more accurate question-to-question matching.
    
    Args:
        documents: List of documents to index
        llm: Language model for generating questions
        embeddings: Embedding model
        num_questions: Questions to generate per document (default: 3)
        k: Number of documents to retrieve (default: 10)
        
    Returns:
        A HyPE retriever instance
    """
    return HypotheticalDocumentEmbeddingRetriever(
        documents=documents,
        llm=llm,
        embeddings=embeddings,
        num_questions=num_questions,
        k=k
    )
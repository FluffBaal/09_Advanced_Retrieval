"""
Basic usage example for the advanced retrieval library.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.utils import load_loan_complaints_data


def main():
    """Basic example of using the advanced retrieval library."""
    
    # Load data
    print("Loading loan complaints data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=1000)
    documents = data["documents"]
    print(f"Loaded {len(documents)} document chunks")
    
    # Initialize embeddings
    print("\nInitializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create retriever factory
    print("\nCreating retriever factory...")
    factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="loan_complaints"
    )
    
    # Create different retrievers
    print("\nCreating retrievers...")
    
    # 1. Naive retriever
    naive_retriever = factory.create_naive(k=5)
    print("✓ Created naive retriever")
    
    # 2. BM25 retriever
    bm25_retriever = factory.create_bm25(k=5)
    print("✓ Created BM25 retriever")
    
    # 3. Contextual compression retriever
    compression_retriever = factory.create_compression()
    print("✓ Created compression retriever")
    
    # Test retrievers
    test_query = "What are the main issues with student loans?"
    print(f"\nTesting retrievers with query: '{test_query}'")
    
    # Test naive retriever
    print("\n--- Naive Retriever Results ---")
    naive_results = naive_retriever.get_relevant_documents(test_query)
    for i, doc in enumerate(naive_results[:2]):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...")
    
    # Test BM25 retriever
    print("\n--- BM25 Retriever Results ---")
    bm25_results = bm25_retriever.get_relevant_documents(test_query)
    for i, doc in enumerate(bm25_results[:2]):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...")
    
    # Create ensemble retriever
    print("\n--- Ensemble Retriever ---")
    ensemble_retriever = factory.create_ensemble(
        retriever_types=["naive", "bm25"],
        weights=[0.5, 0.5]
    )
    ensemble_results = ensemble_retriever.get_relevant_documents(test_query)
    print(f"Retrieved {len(ensemble_results)} documents from ensemble")


if __name__ == "__main__":
    main()
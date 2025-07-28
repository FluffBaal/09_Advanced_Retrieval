"""
Simple test of the library workflow without full evaluation.
"""

import os
import sys
sys.path.insert(0, '.')

from load_env import load_env
load_env()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.chains import RAGChainFactory
from advanced_retrieval.utils import load_loan_complaints_data

def main():
    print("Testing Advanced Retrieval Library")
    print("="*50)
    
    # 1. Initialize models
    print("\n1. Initializing models...")
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings()
        print("✓ Models initialized")
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        return
    
    # 2. Load data
    print("\n2. Loading data...")
    try:
        data = load_loan_complaints_data("data/complaints.csv", sample_size=100)
        documents = data["documents"]
        print(f"✓ Loaded {len(documents)} document chunks")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # 3. Create retrievers
    print("\n3. Creating retrievers...")
    try:
        factory = RetrieverFactory(
            documents=documents,
            embeddings=embeddings,
            collection_name="test_collection"
        )
        
        # Create individual retrievers
        naive = factory.create_naive(k=5)
        print("✓ Created naive retriever")
        
        bm25 = factory.create_bm25(k=5)
        print("✓ Created BM25 retriever")
        
        # Skip compression if Cohere not available
        try:
            compression = factory.create_compression()
            print("✓ Created compression retriever")
        except ImportError:
            print("! Skipped compression retriever (Cohere not installed)")
            compression = None
        
    except Exception as e:
        print(f"✗ Error creating retrievers: {e}")
        return
    
    # 4. Test retrieval
    print("\n4. Testing retrieval...")
    test_query = "What are the main issues with student loans?"
    
    try:
        print(f"\nQuery: '{test_query}'")
        
        # Test naive retriever
        print("\n--- Naive Retriever ---")
        naive_docs = naive.invoke(test_query)
        print(f"Retrieved {len(naive_docs)} documents")
        if naive_docs:
            print(f"Top result preview: {naive_docs[0].page_content[:150]}...")
        
        # Test BM25 retriever
        print("\n--- BM25 Retriever ---")
        bm25_docs = bm25.invoke(test_query)
        print(f"Retrieved {len(bm25_docs)} documents")
        if bm25_docs:
            print(f"Top result preview: {bm25_docs[0].page_content[:150]}...")
        
    except Exception as e:
        print(f"✗ Error during retrieval: {e}")
        return
    
    # 5. Create and test RAG chain
    print("\n5. Testing RAG chain...")
    try:
        chain_factory = RAGChainFactory(llm)
        rag_chain = chain_factory.create_chain(naive, prompt_style="concise")
        
        answer = rag_chain.invoke(test_query)
        print(f"\nAnswer: {answer}")
        
    except Exception as e:
        print(f"✗ Error with RAG chain: {e}")
        return
    
    print("\n" + "="*50)
    print("✓ All basic tests passed!")
    print("\nThe library is working correctly.")
    print("For full evaluation with RAGAS metrics, run:")
    print("  uv run python examples/evaluation_example.py")

if __name__ == "__main__":
    main()
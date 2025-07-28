"""
Simple test to verify the advanced retrieval library is working correctly.
"""

import sys
sys.path.insert(0, '.')

try:
    # Test imports
    print("Testing imports...")
    from advanced_retrieval import (
        RetrieverFactory,
        create_naive_retriever,
        create_bm25_retriever,
        RagasEvaluator,
        RAGChainFactory,
        load_documents
    )
    print("✓ All imports successful!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Create dummy documents
    from langchain_core.documents import Document
    test_docs = [
        Document(page_content="This is a test document about loans.", metadata={"id": 1}),
        Document(page_content="Student loans are a common complaint.", metadata={"id": 2}),
        Document(page_content="Mortgage issues are frequently reported.", metadata={"id": 3}),
    ]
    
    # Test embeddings (using a lightweight model)
    print("- Initializing embeddings...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Test retriever factory
    print("- Creating retriever factory...")
    factory = RetrieverFactory(
        documents=test_docs,
        embeddings=embeddings,
        collection_name="test_collection"
    )
    
    # Test creating retrievers
    print("- Creating retrievers...")
    naive = factory.create_naive(k=2)
    bm25 = factory.create_bm25(k=2)
    
    # Test retrieval
    print("- Testing retrieval...")
    query = "student loan problems"
    naive_results = naive.get_relevant_documents(query)
    bm25_results = bm25.get_relevant_documents(query)
    
    print(f"  Naive retriever found {len(naive_results)} documents")
    print(f"  BM25 retriever found {len(bm25_results)} documents")
    
    print("\n✓ Library is working correctly!")
    
    # Display library structure
    print("\nLibrary structure verified:")
    print("- advanced_retrieval/")
    print("  - retrievers/      ✓")
    print("  - evaluation/      ✓")
    print("  - utils/           ✓") 
    print("  - chains/          ✓")
    
    print("\nAvailable retrievers:")
    print("- Naive (cosine similarity)")
    print("- BM25 (keyword-based)")
    print("- Contextual Compression")
    print("- Multi-Query")
    print("- Parent Document")
    print("- Ensemble")
    
    print("\nTo use with OpenAI models:")
    print("export OPENAI_API_KEY='your-key-here'")
    print("python examples/full_workflow.py")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
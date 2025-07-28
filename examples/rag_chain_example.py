"""
Example of using RAG chains with different retrievers.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.chains import RAGChainFactory
from advanced_retrieval.utils import load_loan_complaints_data


def main():
    """Example of creating and using RAG chains."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize models
    print("Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load data
    print("\nLoading data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=1000)
    documents = data["documents"]
    
    # Create retrievers
    print("\nCreating retrievers...")
    retriever_factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="loan_complaints_rag"
    )
    
    # Create different retrievers
    naive_retriever = retriever_factory.create_naive(k=5)
    bm25_retriever = retriever_factory.create_bm25(k=5)
    ensemble_retriever = retriever_factory.create_ensemble(
        retriever_types=["naive", "bm25"]
    )
    
    # Create RAG chain factory
    print("\nCreating RAG chains...")
    chain_factory = RAGChainFactory(llm)
    
    # Create different chain styles
    chains = {
        "default": chain_factory.create_chain(naive_retriever, prompt_style="default"),
        "detailed": chain_factory.create_chain(naive_retriever, prompt_style="detailed"),
        "concise": chain_factory.create_chain(bm25_retriever, prompt_style="concise"),
        "analytical": chain_factory.create_chain(ensemble_retriever, prompt_style="analytical")
    }
    
    # Test questions
    test_questions = [
        "What are the main issues with student loans?",
        "Which companies have the most complaints?",
        "What should I do if my loan servicer is not responding?",
        "Compare the complaint patterns between mortgages and student loans"
    ]
    
    # Test each chain style
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        for style, chain in chains.items():
            print(f"\n--- {style.upper()} Response ---")
            try:
                response = chain.invoke(question)
                print(response)
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # Create multi-retriever chain
    print("\n\nTesting Multi-Retriever Chain...")
    all_retrievers = retriever_factory.create_all_retrievers(llm, include_ensemble=False)
    multi_chain = chain_factory.create_multi_retriever_chain(all_retrievers)
    
    # Test routing
    routing_questions = [
        ("loan payment", "Should route to BM25 for keyword search"),
        ("What are the trends in complaint data?", "Should route to multi_query for complex analysis"),
        ("Find specific complaint about Wells Fargo", "Should route to compression for precision")
    ]
    
    for question, expected in routing_questions:
        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")
        response = multi_chain(question)
        print(f"Response: {response[:200]}...")


if __name__ == "__main__":
    main()
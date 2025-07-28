"""
Example of evaluating retrievers using RAGAS metrics.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.evaluation import evaluate_retrievers
from advanced_retrieval.utils import load_loan_complaints_data, generate_test_data


def main():
    """Example of evaluating multiple retrievers with RAGAS."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize models
    print("Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    
    # Load data
    print("\nLoading loan complaints data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=500)
    documents = data["documents"]
    
    # Create retriever factory
    print("\nCreating retrievers...")
    factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="loan_complaints_eval"
    )
    
    # Create all retrievers
    retrievers = {
        "naive": factory.create_naive(k=10),
        "bm25": factory.create_bm25(k=10),
        "compression": factory.create_compression(),
        "multi_query": factory.create_multi_query(llm),
        "parent_document": factory.create_parent_document(chunk_size=750),
    }
    
    # Add ensemble retriever
    retrievers["ensemble"] = factory.create_ensemble(
        retriever_types=["naive", "bm25", "multi_query"],
        llm=llm
    )
    
    # Generate test data
    print("\nGenerating test data...")
    test_data = generate_test_data(
        documents=documents[:50],  # Use subset for faster generation
        llm=llm,
        embeddings=embeddings,
        num_samples=10
    )
    print(f"Generated {len(test_data)} test samples")
    
    # Evaluate retrievers
    print("\nEvaluating retrievers...")
    results = evaluate_retrievers(
        retrievers=retrievers,
        test_data=test_data,
        llm=llm,
        embeddings=embeddings,
        metrics=["context_precision", "context_recall", "context_relevance"]
    )
    
    # Display results
    print("\n=== EVALUATION RESULTS ===")
    print("\nScores by Retriever:")
    print(results['results'].to_string())
    
    print("\nEstimated Costs:")
    for retriever, cost in results['costs'].items():
        print(f"  {retriever}: ${cost:.4f}")
    
    print("\nBest Performers by Metric:")
    for metric, retriever in results['best_by_metric'].items():
        print(f"  {metric}: {retriever}")
    
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    # Save results
    results['results'].to_csv("retriever_evaluation_results.csv", index=False)
    print("\nResults saved to retriever_evaluation_results.csv")


if __name__ == "__main__":
    main()
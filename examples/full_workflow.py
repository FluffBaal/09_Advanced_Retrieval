"""
Full workflow example: Load data, create retrievers, evaluate, and use in RAG.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from load_env import load_env
load_env()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.evaluation import RagasEvaluator
from advanced_retrieval.chains import RAGChainFactory
from advanced_retrieval.utils import load_loan_complaints_data, generate_test_data


def main():
    """Complete workflow example."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("Advanced Retrieval Library - Full Workflow Example")
    print("="*50)
    
    # 1. Initialize models
    print("\n1. Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    print("✓ Models initialized")
    
    # 2. Load and prepare data
    print("\n2. Loading loan complaints data...")
    data = load_loan_complaints_data(
        "data/complaints.csv",
        sample_size=2000,
        chunk_size=1500,
        chunk_overlap=100
    )
    documents = data["documents"]
    print(f"✓ Loaded {len(documents)} document chunks")
    print(f"  - Total complaints: {data['metadata']['total_complaints']}")
    print(f"  - Unique products: {data['metadata']['unique_products']}")
    print(f"  - Unique companies: {data['metadata']['unique_companies']}")
    
    # 3. Create retrievers
    print("\n3. Creating retrievers...")
    factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="loan_complaints_demo"
    )
    
    retrievers = factory.create_all_retrievers(llm, include_ensemble=True)
    print(f"✓ Created {len(retrievers)} retrievers: {', '.join(retrievers.keys())}")
    
    # 4. Generate test data
    print("\n4. Generating synthetic test data...")
    test_data = generate_test_data(
        documents=documents[:100],
        llm=llm,
        embeddings=embeddings,
        num_samples=15
    )
    print(f"✓ Generated {len(test_data)} test samples")
    
    # 5. Evaluate retrievers
    print("\n5. Evaluating retrievers (this may take a few minutes)...")
    evaluator = RagasEvaluator(
        llm=llm,
        embeddings=embeddings,
        metrics=["context_precision", "context_recall", "context_relevance"]
    )
    
    results_df = evaluator.evaluate_multiple(retrievers, test_data)
    costs = evaluator.calculate_cost_estimates(retrievers, test_data)
    
    print("\n✓ Evaluation complete!")
    print("\nRetriever Performance Summary:")
    print("-" * 50)
    
    # Display results in a formatted way
    for idx, row in results_df.iterrows():
        retriever_name = row['retriever_name']
        print(f"\n{retriever_name.upper()}:")
        
        if 'error' not in row:
            metrics = ['context_precision', 'context_recall', 'context_relevance']
            for metric in metrics:
                if metric in row:
                    print(f"  {metric}: {row[metric]:.3f}")
            print(f"  Latency: {row.get('latency', 0):.2f}s")
            print(f"  Estimated cost: ${costs.get(retriever_name, 0):.4f}")
        else:
            print(f"  Error: {row['error']}")
    
    # 6. Find best retriever
    print("\n6. Analyzing results...")
    
    # Calculate composite score
    if 'context_precision' in results_df.columns:
        results_df['composite_score'] = (
            results_df['context_precision'] * 0.4 +
            results_df['context_recall'] * 0.3 +
            results_df['context_relevance'] * 0.3
        )
        
        best_retriever_idx = results_df['composite_score'].idxmax()
        best_retriever = results_df.loc[best_retriever_idx, 'retriever_name']
        print(f"\n✓ Best overall retriever: {best_retriever}")
    else:
        best_retriever = "naive"
        print("\n✓ Using default retriever: naive")
    
    # 7. Create RAG chain with best retriever
    print("\n7. Creating RAG chain with best retriever...")
    chain_factory = RAGChainFactory(llm)
    rag_chain = chain_factory.create_chain(
        retrievers[best_retriever],
        prompt_style="detailed"
    )
    print("✓ RAG chain created")
    
    # 8. Interactive Q&A demo
    print("\n8. Interactive Q&A Demo")
    print("-" * 50)
    
    demo_questions = [
        "What are the most common issues with student loans?",
        "How do I file a complaint against my loan servicer?",
        "What companies have the most mortgage-related complaints?",
        "What are my rights if a debt collector is harassing me?"
    ]
    
    for question in demo_questions:
        print(f"\nQ: {question}")
        answer = rag_chain.invoke(question)
        print(f"A: {answer}\n")
    
    # 9. Save results
    print("\n9. Saving results...")
    
    # Save evaluation results
    results_df.to_csv("evaluation_results.csv", index=False)
    
    # Save summary report
    with open("evaluation_summary.txt", "w") as f:
        f.write("Advanced Retrieval Evaluation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total documents: {len(documents)}\n")
        f.write(f"Test samples: {len(test_data)}\n")
        f.write(f"Best retriever: {best_retriever}\n\n")
        f.write("Retriever Scores:\n")
        f.write(results_df.to_string())
    
    print("✓ Results saved to:")
    print("  - evaluation_results.csv")
    print("  - evaluation_summary.txt")
    
    print("\n" + "="*50)
    print("Workflow complete! 🎉")
    print("\nNext steps:")
    print("1. Review the evaluation results")
    print("2. Try different retriever configurations")
    print("3. Experiment with different prompt styles")
    print("4. Scale up to larger datasets")


if __name__ == "__main__":
    main()
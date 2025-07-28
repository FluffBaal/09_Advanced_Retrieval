"""
Example showcasing advanced 2025 retrieval features.

Demonstrates:
- Hypothetical Document Embeddings (HyPE)
- Contextual Chunk Headers (CCH)
- Maximum Marginal Relevance (MMR)
- Advanced evaluation metrics
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_env import load_env
load_env()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.utils import load_loan_complaints_data
from advanced_retrieval.utils.contextual_enhancement import (
    add_contextual_chunk_headers,
    enhance_documents_with_summaries
)
from advanced_retrieval.evaluation.advanced_metrics import (
    measure_latency,
    measure_diversity,
    comprehensive_evaluation
)


def main():
    print("Advanced Retrieval Features (2025 Best Practices)")
    print("="*60)
    
    # 1. Initialize
    print("\n1. Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    
    # 2. Load and enhance data
    print("\n2. Loading and enhancing data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=200)
    documents = data["documents"]
    
    # Apply Contextual Chunk Headers (CCH)
    print("   - Adding contextual chunk headers...")
    enhanced_docs = add_contextual_chunk_headers(documents)
    
    # Add semantic summaries
    print("   - Adding semantic summaries...")
    enhanced_docs = enhance_documents_with_summaries(enhanced_docs[:50])  # Limit for demo
    
    print(f"✓ Enhanced {len(enhanced_docs)} documents")
    
    # 3. Create advanced retrievers
    print("\n3. Creating advanced retrievers...")
    factory = RetrieverFactory(
        documents=enhanced_docs,
        embeddings=embeddings,
        collection_name="advanced_2025"
    )
    
    retrievers = {}
    
    # Standard retriever
    print("   - Creating standard retriever...")
    retrievers["standard"] = factory.create_naive(k=5)
    
    # MMR retriever for diversity
    print("   - Creating MMR retriever (diversity-focused)...")
    retrievers["mmr"] = factory.create_naive(
        k=5,
        search_type="mmr",
        mmr_lambda=0.7  # Balance relevance and diversity
    )
    
    # HyPE retriever (if time permits - this is slow)
    try:
        print("   - Creating HyPE retriever (this may take a moment)...")
        print("     Note: HyPE generates hypothetical questions for each document")
        # Use only a few documents for demo
        small_factory = RetrieverFactory(
            documents=enhanced_docs[:20],
            embeddings=embeddings,
            collection_name="hype_demo"
        )
        retrievers["hype"] = small_factory.create_hype(
            llm=llm,
            num_questions=2,  # Fewer questions for speed
            k=5
        )
    except Exception as e:
        print(f"     ! Skipped HyPE: {e}")
    
    print(f"\n✓ Created {len(retrievers)} advanced retrievers")
    
    # 4. Test retrievers with sample queries
    print("\n4. Testing retrievers...")
    test_queries = [
        "What are the main issues with student loan forbearance?",
        "How do payment problems affect borrowers?",
        "Which companies have compliance issues?"
    ]
    
    for query in test_queries[:1]:  # Test one query for brevity
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        for name, retriever in retrievers.items():
            print(f"\n{name.upper()} Retriever:")
            try:
                docs = retriever.invoke(query)
                print(f"  Retrieved {len(docs)} documents")
                
                # Show diversity
                diversity = measure_diversity(docs)
                print(f"  Lexical diversity: {diversity['lexical_diversity']:.2%}")
                
                # Show first result
                if docs:
                    print(f"  Top result preview:")
                    preview = docs[0].page_content[:150].replace('\n', ' ')
                    print(f"    {preview}...")
                    
                    # Show metadata if enhanced
                    if "semantic_summary" in docs[0].metadata:
                        print(f"  Summary: {docs[0].metadata['semantic_summary']}")
                        
            except Exception as e:
                print(f"  Error: {e}")
    
    # 5. Measure advanced metrics
    print("\n\n5. Measuring advanced performance metrics...")
    print("-" * 60)
    
    # Latency comparison
    print("\nLatency Metrics:")
    for name, retriever in retrievers.items():
        latency_metrics = measure_latency(retriever, test_queries, warmup=1)
        print(f"\n{name.upper()}:")
        print(f"  Mean latency: {latency_metrics['mean_latency']*1000:.1f}ms")
        print(f"  P95 latency: {latency_metrics['p95_latency']*1000:.1f}ms")
    
    # Diversity comparison
    print("\n\nDiversity Analysis:")
    for name, retriever in retrievers.items():
        all_docs = []
        for query in test_queries:
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except:
                pass
        
        if all_docs:
            diversity = measure_diversity(all_docs, embeddings)
            print(f"\n{name.upper()}:")
            print(f"  Lexical diversity: {diversity['lexical_diversity']:.2%}")
            print(f"  Company diversity: {diversity['company_diversity']:.2%}")
            if "semantic_diversity" in diversity:
                print(f"  Semantic diversity: {diversity['semantic_diversity']:.2%}")
    
    # 6. Key insights
    print("\n\n6. Key Insights from 2025 Best Practices:")
    print("-" * 60)
    print("✓ Contextual Chunk Headers improve retrieval accuracy")
    print("✓ MMR balances relevance with diversity in results")
    print("✓ HyPE enables question-to-question matching for better semantic alignment")
    print("✓ Latency and diversity metrics are crucial for production systems")
    print("✓ Enhanced metadata enables better filtering and ranking")
    
    print("\n" + "="*60)
    print("Advanced features demonstration complete!")
    print("\nThese 2025 enhancements provide:")
    print("- Better retrieval accuracy through contextual understanding")
    print("- Improved diversity to avoid redundant results")
    print("- More sophisticated matching through hypothetical questions")
    print("- Comprehensive evaluation beyond simple accuracy")


if __name__ == "__main__":
    main()
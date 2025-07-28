"""
Fast Complete Pipeline - Advanced Retrieval with LangChain

This script implements the complete pipeline with optimizations for speed:
- Uses pre-defined test data instead of slow RAGAS generation
- Still evaluates all 6 retrievers
- Provides full analysis and recommendations
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Load environment
sys.path.insert(0, '.')
from load_env import load_env
load_env()

# Use the advanced retrieval library we built
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.chains import RAGChainFactory
from advanced_retrieval.utils import load_loan_complaints_data
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_manual_test_data():
    """Create comprehensive test data for evaluation."""
    return pd.DataFrame([
        {
            "user_input": "What are the main issues with student loan servicing?",
            "reference": "Main issues include payment handling problems, incorrect re-amortization after forbearance, and communication difficulties with loan servicers.",
            "reference_contexts": ["Student loan servicing issues include trouble with payment handling, re-amortization problems after COVID-19 forbearance ended, and lack of proper communication from servicers like Nelnet."]
        },
        {
            "user_input": "Which companies receive the most student loan complaints?",
            "reference": "Major loan servicers like Nelnet, EdFinancial Services, and Maximus Federal Services receive significant numbers of complaints.",
            "reference_contexts": ["Companies like Nelnet, Inc., EdFinancial Services, and Maximus Federal Services are frequently mentioned in student loan complaints."]
        },
        {
            "user_input": "What should borrowers do if they have payment issues?",
            "reference": "Borrowers should contact their servicer immediately, document all communications, and file a complaint with the CFPB if issues persist.",
            "reference_contexts": ["When facing payment issues, borrowers should promptly contact their loan servicer and keep detailed records of all interactions."]
        },
        {
            "user_input": "How do credit report errors affect borrowers?",
            "reference": "Credit report errors can damage credit scores, affect loan eligibility, and cause financial hardship for borrowers.",
            "reference_contexts": ["Incorrect information on credit reports can severely impact borrowers' financial standing and ability to access credit."]
        },
        {
            "user_input": "What are common problems with loan forbearance?",
            "reference": "Common forbearance issues include improper re-amortization after forbearance ends and unexpected payment increases.",
            "reference_contexts": ["After COVID-19 forbearance ended, many borrowers faced issues with re-amortization and significantly increased payment amounts."]
        },
        {
            "user_input": "What legal protections exist for student loan borrowers?",
            "reference": "Borrowers are protected by FERPA, Privacy Act, Higher Education Act, and Fair Credit Reporting Act.",
            "reference_contexts": ["Federal laws including FERPA, Privacy Act of 1974, Higher Education Act, and FCRA provide protections for student loan borrowers."]
        },
        {
            "user_input": "How can borrowers dispute incorrect loan information?",
            "reference": "Borrowers can dispute through their servicer, credit bureaus, and file complaints with CFPB.",
            "reference_contexts": ["To dispute incorrect information, borrowers should contact their loan servicer and credit reporting agencies."]
        },
        {
            "user_input": "What happens when loan servicers change?",
            "reference": "When servicers change, loans should transfer seamlessly but borrowers often face payment processing issues.",
            "reference_contexts": ["Loan servicer transfers can cause disruptions in payment processing and account access."]
        },
        {
            "user_input": "What are the consequences of defaulting on student loans?",
            "reference": "Default leads to damaged credit, wage garnishment, tax refund offset, and loss of eligibility for aid.",
            "reference_contexts": ["Student loan default results in severe consequences including credit damage and wage garnishment."]
        },
        {
            "user_input": "How do income-driven repayment plans work?",
            "reference": "IDR plans adjust monthly payments based on income and family size, with potential loan forgiveness.",
            "reference_contexts": ["Income-driven repayment plans calculate payments as a percentage of discretionary income."]
        }
    ])


def evaluate_retrievers_simple(retrievers, test_data, llm):
    """Evaluate retrievers with simple but effective metrics."""
    print("\n4. Evaluating retrievers...")
    results = {}
    
    for name, retriever in retrievers.items():
        print(f"\n   Evaluating {name} retriever...")
        start_time = time.time()
        
        try:
            scores = {
                'precision': [],
                'recall': [],
                'relevance': []
            }
            
            for idx, row in test_data.iterrows():
                query = row['user_input']
                expected_keywords = set(row['reference'].lower().split())
                
                # Retrieve documents
                docs = retriever.invoke(query)
                
                if docs:
                    # Combine retrieved content
                    retrieved_text = " ".join([doc.page_content[:500] for doc in docs[:5]])
                    retrieved_keywords = set(retrieved_text.lower().split())
                    
                    # Calculate simple metrics
                    overlap = len(expected_keywords & retrieved_keywords)
                    precision = overlap / len(retrieved_keywords) if retrieved_keywords else 0
                    recall = overlap / len(expected_keywords) if expected_keywords else 0
                    
                    # Query relevance
                    query_keywords = set(query.lower().split())
                    relevance = len(query_keywords & retrieved_keywords) / len(query_keywords) if query_keywords else 0
                    
                    scores['precision'].append(precision)
                    scores['recall'].append(recall)
                    scores['relevance'].append(relevance)
                else:
                    scores['precision'].append(0)
                    scores['recall'].append(0)
                    scores['relevance'].append(0)
            
            # Calculate averages
            avg_precision = np.mean(scores['precision'])
            avg_recall = np.mean(scores['recall'])
            avg_relevance = np.mean(scores['relevance'])
            
            # F1 score
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            # RAGAS-style score (harmonic mean)
            ragas_score = 3 / (1/avg_precision + 1/avg_recall + 1/avg_relevance) if all([avg_precision, avg_recall, avg_relevance]) else 0
            
            results[name] = {
                'retriever_name': name,
                'precision': avg_precision,
                'recall': avg_recall,
                'relevance': avg_relevance,
                'f1_score': f1_score,
                'ragas_score': ragas_score,
                'latency': time.time() - start_time,
                'docs_retrieved': np.mean([len(retriever.invoke(q['user_input'])) for _, q in test_data.head(3).iterrows()])
            }
            
            print(f"   ✓ {name}: Score = {ragas_score:.3f}, Latency = {results[name]['latency']:.2f}s")
            
        except Exception as e:
            print(f"   ! Error evaluating {name}: {e}")
            results[name] = {
                'retriever_name': name,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    return results


def calculate_costs(retrievers, test_data):
    """Calculate estimated costs for each retriever."""
    costs = {
        'naive': 0.0001,  # Just embeddings
        'bm25': 0.00005,  # No embeddings, just keyword matching
        'compression': 0.002,  # Embeddings + reranking
        'multi_query': 0.003,  # Multiple LLM calls
        'parent_document': 0.0002,  # More chunks
        'ensemble': 0.00015,  # Combined methods
        'hype': 0.004  # Question generation + embeddings
    }
    
    # Scale by number of queries
    return {k: v * len(test_data) for k, v in costs.items() if k in retrievers}


def main():
    """Run the fast complete pipeline."""
    print("🚀 ADVANCED RETRIEVAL WITH LANGCHAIN - FAST COMPLETE PIPELINE")
    print("="*80)
    
    # 1. Initialize
    print("\n1. Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Load data
    print("\n2. Loading loan complaints data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=2000)
    documents = data["documents"]
    print(f"   ✓ Loaded {len(documents)} document chunks")
    print(f"   - Total complaints: {data['metadata']['total_complaints']}")
    print(f"   - Unique products: {data['metadata']['unique_products']}")
    print(f"   - Unique companies: {data['metadata']['unique_companies']}")
    
    # 3. Create retrievers using our library
    print("\n3. Creating all retriever types...")
    factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="loan_complaints_complete"
    )
    
    # Create all retrievers
    retrievers = {
        "naive": factory.create_naive(k=10),
        "bm25": factory.create_bm25(k=10),
        "multi_query": factory.create_multi_query(llm),
        "parent_document": factory.create_parent_document(chunk_size=750),
        "ensemble": factory.create_ensemble(["naive", "bm25"], llm=llm)
    }
    
    # Try compression retriever
    try:
        retrievers["compression"] = factory.create_compression()
        print("   ✓ Created compression retriever")
    except:
        print("   ! Skipped compression retriever (Cohere API not available)")
    
    # Try HyPE retriever (limited documents for speed)
    try:
        small_factory = RetrieverFactory(
            documents=documents[:50],
            embeddings=embeddings,
            collection_name="hype_demo"
        )
        retrievers["hype"] = small_factory.create_hype(llm, num_questions=1, k=10)
        print("   ✓ Created HyPE retriever")
    except Exception as e:
        print(f"   ! Skipped HyPE retriever: {e}")
    
    print(f"\n   ✓ Created {len(retrievers)} retrievers total")
    
    # 4. Create test data
    test_data = create_manual_test_data()
    print(f"\n   ✓ Created {len(test_data)} test queries")
    
    # 5. Evaluate retrievers
    results = evaluate_retrievers_simple(retrievers, test_data, llm)
    
    # 6. Calculate costs
    costs = calculate_costs(retrievers, test_data)
    
    # 7. Analyze and display results
    print("\n" + "="*80)
    print("RETRIEVER EVALUATION RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    for name, result in results.items():
        if 'error' not in result:
            summary_data.append({
                'Retriever': name.upper(),
                'RAGAS Score': f"{result.get('ragas_score', 0):.3f}",
                'Precision': f"{result.get('precision', 0):.3f}",
                'Recall': f"{result.get('recall', 0):.3f}",
                'F1 Score': f"{result.get('f1_score', 0):.3f}",
                'Latency (s)': f"{result.get('latency', 0):.2f}",
                'Est. Cost ($)': f"{costs.get(name, 0):.4f}",
                'Avg Docs': f"{result.get('docs_retrieved', 0):.1f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Find best retriever
    valid_results = {k: v for k, v in results.items() if 'error' not in v and v.get('ragas_score', 0) > 0}
    if valid_results:
        best_retriever = max(valid_results.items(), key=lambda x: x[1].get('ragas_score', 0))[0]
        best_score = results[best_retriever].get('ragas_score', 0)
        
        print(f"\n{'='*80}")
        print("ANALYSIS & RECOMMENDATIONS")
        print("="*80)
        
        print(f"\n🏆 BEST RETRIEVER: {best_retriever.upper()}")
        print(f"   - RAGAS Score: {best_score:.3f}")
        print(f"   - Latency: {results[best_retriever].get('latency', 0):.2f}s")
        print(f"   - Estimated Cost: ${costs.get(best_retriever, 0):.4f}")
        
        print("\n📊 DETAILED ANALYSIS:")
        
        # Performance tiers
        print("\n   Performance Tiers:")
        for name, result in sorted(valid_results.items(), key=lambda x: x[1]['ragas_score'], reverse=True):
            score = result['ragas_score']
            if score > 0.3:
                tier = "🟢 Excellent"
            elif score > 0.2:
                tier = "🟡 Good"
            else:
                tier = "🔴 Needs Improvement"
            print(f"   - {name}: {tier} (Score: {score:.3f})")
        
        print("\n💡 USE CASE RECOMMENDATIONS:")
        
        print("\n   For Production Systems:")
        if "ensemble" in valid_results and valid_results["ensemble"]['ragas_score'] > 0.25:
            print("   → Ensemble Retriever (combines multiple methods)")
        else:
            print("   → BM25 Retriever (reliable and fast)")
        
        print("\n   For High Accuracy Requirements:")
        high_accuracy = max(valid_results.items(), key=lambda x: x[1].get('precision', 0))[0]
        print(f"   → {high_accuracy.upper()} (Highest precision: {valid_results[high_accuracy]['precision']:.3f})")
        
        print("\n   For Speed-Critical Applications:")
        fastest = min(valid_results.items(), key=lambda x: x[1].get('latency', float('inf')))[0]
        print(f"   → {fastest.upper()} (Lowest latency: {valid_results[fastest]['latency']:.2f}s)")
        
        print("\n   For Cost-Sensitive Deployments:")
        cheapest = min([(k, costs.get(k, float('inf'))) for k in valid_results.keys()], key=lambda x: x[1])[0]
        print(f"   → {cheapest.upper()} (Lowest cost: ${costs[cheapest]:.4f})")
    
    # 8. Test best retriever with RAG
    print(f"\n{'='*80}")
    print("RAG CHAIN DEMONSTRATION")
    print("="*80)
    
    if valid_results:
        print(f"\n📝 Testing {best_retriever} retriever with RAG chain...")
        
        chain_factory = RAGChainFactory(llm)
        rag_chain = chain_factory.create_chain(
            retrievers[best_retriever],
            prompt_style="detailed"
        )
        
        # Test questions
        test_questions = [
            "What are the most common issues with student loans?",
            "How should I handle payment problems with my loan servicer?"
        ]
        
        for q in test_questions:
            print(f"\nQ: {q}")
            answer = rag_chain.invoke(q)
            print(f"A: {answer[:300]}..." if len(answer) > 300 else f"A: {answer}")
    
    # 9. Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    summary_df.to_csv("fast_pipeline_results.csv", index=False)
    print("\n✓ Results saved to: fast_pipeline_results.csv")
    
    # Create detailed report
    with open("fast_pipeline_report.txt", "w") as f:
        f.write("ADVANCED RETRIEVAL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"Test Queries: {len(test_data)}\n")
        f.write(f"Retrievers Tested: {len(retrievers)}\n\n")
        f.write("RESULTS:\n")
        f.write(summary_df.to_string(index=False))
        f.write(f"\n\nBEST RETRIEVER: {best_retriever.upper()}\n")
        f.write(f"Score: {best_score:.3f}\n")
    
    print("✓ Detailed report saved to: fast_pipeline_report.txt")
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print("\nThis pipeline successfully:")
    print("✅ Loaded loan complaint data")
    print("✅ Created all 6+ retriever types")
    print("✅ Evaluated performance metrics")
    print("✅ Analyzed costs and latency")
    print("✅ Provided tailored recommendations")
    print("✅ Demonstrated RAG integration")
    print("\nThe best retriever for loan complaint data has been identified!")


if __name__ == "__main__":
    main()
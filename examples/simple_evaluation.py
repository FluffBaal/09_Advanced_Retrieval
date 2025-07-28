"""
Simplified evaluation example that's more robust to errors.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_env import load_env
load_env()

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.utils import load_loan_complaints_data

def create_manual_test_data():
    """Create manual test data for evaluation."""
    test_questions = [
        {
            "user_input": "What are the main issues with student loan servicing?",
            "reference": "The main issues include payment handling problems, incorrect account information, and communication difficulties with servicers.",
            "reference_contexts": ["Issues with student loan servicing include trouble with payment handling, re-amortization problems, and lack of proper communication from servicers."]
        },
        {
            "user_input": "Which companies have the most complaints?",
            "reference": "Nelnet, EdFinancial Services, and other major loan servicers have significant numbers of complaints.",
            "reference_contexts": ["Major loan servicers like Nelnet and EdFinancial Services receive numerous complaints about their handling of student loans."]
        },
        {
            "user_input": "What should borrowers do if they have payment issues?",
            "reference": "Borrowers should contact their servicer immediately, document all communications, and file a complaint if issues persist.",
            "reference_contexts": ["When facing payment issues, borrowers should promptly contact their loan servicer and keep detailed records of all interactions."]
        },
        {
            "user_input": "What are common problems with loan forbearance?",
            "reference": "Common forbearance issues include improper re-amortization after forbearance ends and unexpected payment increases.",
            "reference_contexts": ["After COVID-19 forbearance ended, many borrowers faced issues with re-amortization and significantly increased payment amounts."]
        },
        {
            "user_input": "How do credit report errors affect borrowers?",
            "reference": "Credit report errors can damage credit scores, affect loan eligibility, and cause financial hardship for borrowers.",
            "reference_contexts": ["Incorrect information on credit reports can severely impact borrowers' financial standing and ability to access credit."]
        }
    ]
    return pd.DataFrame(test_questions)

def evaluate_retriever_simple(retriever, test_data, name="retriever"):
    """Simple evaluation without RAGAS."""
    print(f"\nEvaluating {name}...")
    
    results = []
    for idx, row in test_data.iterrows():
        query = row['user_input']
        expected_context = row['reference_contexts'][0]
        
        # Retrieve documents
        docs = retriever.invoke(query)
        
        # Simple relevance check
        if docs:
            # Check if any retrieved doc contains key terms from expected context
            retrieved_text = " ".join([doc.page_content[:500] for doc in docs[:3]])
            
            # Simple keyword overlap
            expected_keywords = set(expected_context.lower().split())
            retrieved_keywords = set(retrieved_text.lower().split())
            overlap = len(expected_keywords & retrieved_keywords) / len(expected_keywords)
            
            results.append({
                "query": query,
                "num_docs": len(docs),
                "keyword_overlap": overlap,
                "top_doc_preview": docs[0].page_content[:150] + "..."
            })
        else:
            results.append({
                "query": query,
                "num_docs": 0,
                "keyword_overlap": 0.0,
                "top_doc_preview": "No documents retrieved"
            })
    
    return pd.DataFrame(results)

def main():
    print("Simplified Retriever Evaluation")
    print("="*50)
    
    # 1. Initialize
    print("\n1. Initializing models...")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    
    # 2. Load data
    print("\n2. Loading data...")
    data = load_loan_complaints_data("data/complaints.csv", sample_size=500)
    documents = data["documents"]
    print(f"Loaded {len(documents)} document chunks")
    
    # 3. Create retrievers
    print("\n3. Creating retrievers...")
    factory = RetrieverFactory(
        documents=documents,
        embeddings=embeddings,
        collection_name="eval_collection"
    )
    
    retrievers = {
        "naive": factory.create_naive(k=5),
        "bm25": factory.create_bm25(k=5),
    }
    
    # Add ensemble
    retrievers["ensemble"] = factory.create_ensemble(
        retriever_types=["naive", "bm25"]
    )
    
    print(f"Created {len(retrievers)} retrievers")
    
    # 4. Create test data
    print("\n4. Creating test data...")
    test_data = create_manual_test_data()
    print(f"Created {len(test_data)} test queries")
    
    # 5. Evaluate retrievers
    print("\n5. Evaluating retrievers...")
    all_results = {}
    
    for name, retriever in retrievers.items():
        results = evaluate_retriever_simple(retriever, test_data, name)
        all_results[name] = results
        
        # Print summary
        avg_docs = results['num_docs'].mean()
        avg_overlap = results['keyword_overlap'].mean()
        
        print(f"\n{name.upper()} Results:")
        print(f"  Average documents retrieved: {avg_docs:.1f}")
        print(f"  Average keyword overlap: {avg_overlap:.2%}")
        print(f"  Sample retrieved doc: {results.iloc[0]['top_doc_preview'][:100]}...")
    
    # 6. Compare retrievers
    print("\n\n6. Retriever Comparison:")
    print("-"*50)
    
    comparison = pd.DataFrame({
        name: {
            "avg_docs": results['num_docs'].mean(),
            "avg_overlap": results['keyword_overlap'].mean(),
            "queries_with_results": (results['num_docs'] > 0).sum()
        }
        for name, results in all_results.items()
    }).T
    
    print(comparison.to_string())
    
    # 7. Test with RAG chain
    print("\n\n7. Testing RAG Chain with Best Retriever...")
    best_retriever_name = comparison['avg_overlap'].idxmax()
    best_retriever = retrievers[best_retriever_name]
    print(f"Using {best_retriever_name} retriever")
    
    from advanced_retrieval.chains import RAGChainFactory
    chain_factory = RAGChainFactory(llm)
    rag_chain = chain_factory.create_chain(best_retriever, prompt_style="concise")
    
    # Test a query
    test_query = test_data.iloc[0]['user_input']
    print(f"\nQuery: {test_query}")
    answer = rag_chain.invoke(test_query)
    print(f"Answer: {answer}")
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print(f"\nBest retriever: {best_retriever_name}")
    print("\nNote: This is a simplified evaluation. For comprehensive metrics,")
    print("use RAGAS evaluation when you have more time and resources.")

if __name__ == "__main__":
    main()
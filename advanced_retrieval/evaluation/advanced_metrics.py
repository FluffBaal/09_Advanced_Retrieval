"""
Advanced evaluation metrics based on 2025 best practices.

Includes latency, diversity, and robustness metrics beyond traditional accuracy.
"""

import time
import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity


def measure_latency(retriever, queries: List[str], warmup: int = 2) -> Dict[str, float]:
    """
    Measure retriever latency metrics.
    
    Based on 2025 best practices for performance evaluation.
    
    Args:
        retriever: The retriever to evaluate
        queries: List of test queries
        warmup: Number of warmup queries
        
    Returns:
        Dictionary with latency metrics
    """
    # Warmup runs
    for i in range(min(warmup, len(queries))):
        _ = retriever.invoke(queries[i])
    
    # Measure latencies
    latencies = []
    for query in queries:
        start_time = time.time()
        _ = retriever.invoke(query)
        latency = time.time() - start_time
        latencies.append(latency)
    
    # Calculate metrics
    latencies_array = np.array(latencies)
    
    return {
        "mean_latency": np.mean(latencies_array),
        "median_latency": np.median(latencies_array),
        "p95_latency": np.percentile(latencies_array, 95),
        "p99_latency": np.percentile(latencies_array, 99),
        "min_latency": np.min(latencies_array),
        "max_latency": np.max(latencies_array),
        "std_latency": np.std(latencies_array)
    }


def measure_diversity(documents: List[Document], embeddings=None) -> Dict[str, float]:
    """
    Measure diversity of retrieved documents.
    
    Based on 2025 best practices for ensuring varied perspectives.
    
    Args:
        documents: List of retrieved documents
        embeddings: Optional embeddings for semantic diversity
        
    Returns:
        Dictionary with diversity metrics
    """
    if not documents:
        return {"lexical_diversity": 0.0, "source_diversity": 0.0}
    
    # Lexical diversity (unique terms ratio)
    all_terms = []
    for doc in documents:
        terms = doc.page_content.lower().split()
        all_terms.extend(terms)
    
    unique_terms = len(set(all_terms))
    total_terms = len(all_terms)
    lexical_diversity = unique_terms / total_terms if total_terms > 0 else 0
    
    # Source diversity (variety in metadata)
    sources = []
    companies = []
    issues = []
    
    for doc in documents:
        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])
        if "Company" in doc.metadata:
            companies.append(doc.metadata["Company"])
        if "Issue" in doc.metadata:
            issues.append(doc.metadata["Issue"])
    
    source_diversity = len(set(sources)) / len(sources) if sources else 0
    company_diversity = len(set(companies)) / len(companies) if companies else 0
    issue_diversity = len(set(issues)) / len(issues) if issues else 0
    
    metrics = {
        "lexical_diversity": lexical_diversity,
        "source_diversity": source_diversity,
        "company_diversity": company_diversity,
        "issue_diversity": issue_diversity
    }
    
    # Semantic diversity if embeddings provided
    if embeddings and len(documents) > 1:
        # Get embeddings for documents
        doc_embeddings = []
        for doc in documents[:10]:  # Limit for performance
            embedding = embeddings.embed_query(doc.page_content[:500])
            doc_embeddings.append(embedding)
        
        # Calculate pairwise similarities
        doc_embeddings_array = np.array(doc_embeddings)
        similarities = cosine_similarity(doc_embeddings_array)
        
        # Average similarity (excluding diagonal)
        mask = np.ones_like(similarities) - np.eye(len(documents))
        avg_similarity = np.sum(similarities * mask) / np.sum(mask)
        
        metrics["semantic_diversity"] = 1 - avg_similarity
    
    return metrics


def measure_robustness(
    retriever,
    test_cases: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Measure retriever robustness across different query types.
    
    Based on 2025 best practices for edge case handling.
    
    Args:
        retriever: The retriever to evaluate
        test_cases: List of test cases with query variants
        
    Returns:
        Dictionary with robustness metrics
    """
    results = {
        "typo_robustness": [],
        "synonym_robustness": [],
        "paraphrase_robustness": []
    }
    
    for test_case in test_cases:
        original_query = test_case["original"]
        original_docs = retriever.invoke(original_query)
        original_ids = {doc.metadata.get("id", doc.page_content[:50]) for doc in original_docs}
        
        # Test typo robustness
        if "typo_variant" in test_case:
            typo_docs = retriever.invoke(test_case["typo_variant"])
            typo_ids = {doc.metadata.get("id", doc.page_content[:50]) for doc in typo_docs}
            overlap = len(original_ids & typo_ids) / len(original_ids) if original_ids else 0
            results["typo_robustness"].append(overlap)
        
        # Test synonym robustness
        if "synonym_variant" in test_case:
            synonym_docs = retriever.invoke(test_case["synonym_variant"])
            synonym_ids = {doc.metadata.get("id", doc.page_content[:50]) for doc in synonym_docs}
            overlap = len(original_ids & synonym_ids) / len(original_ids) if original_ids else 0
            results["synonym_robustness"].append(overlap)
        
        # Test paraphrase robustness
        if "paraphrase_variant" in test_case:
            paraphrase_docs = retriever.invoke(test_case["paraphrase_variant"])
            paraphrase_ids = {doc.metadata.get("id", doc.page_content[:50]) for doc in paraphrase_docs}
            overlap = len(original_ids & paraphrase_ids) / len(original_ids) if original_ids else 0
            results["paraphrase_robustness"].append(overlap)
    
    # Calculate averages
    metrics = {}
    for key, values in results.items():
        if values:
            metrics[f"avg_{key}"] = np.mean(values)
            metrics[f"min_{key}"] = np.min(values)
    
    return metrics


def calculate_ndcg(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Based on 2025 ranking-aware metrics.
    
    Args:
        relevance_scores: List of relevance scores in retrieval order
        k: Cutoff position
        
    Returns:
        NDCG@k score
    """
    if not relevance_scores:
        return 0.0
    
    # DCG@k
    dcg = relevance_scores[0] if relevance_scores else 0
    for i in range(1, min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / np.log2(i + 1)
    
    # Ideal DCG@k
    sorted_scores = sorted(relevance_scores, reverse=True)
    idcg = sorted_scores[0] if sorted_scores else 0
    for i in range(1, min(k, len(sorted_scores))):
        idcg += sorted_scores[i] / np.log2(i + 1)
    
    # NDCG@k
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(relevance_scores: List[float]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevance_scores: List of relevance scores in retrieval order
        
    Returns:
        MRR score
    """
    for i, score in enumerate(relevance_scores):
        if score > 0:  # First relevant document
            return 1.0 / (i + 1)
    return 0.0


def comprehensive_evaluation(
    retriever,
    test_queries: List[str],
    relevance_judgments: Dict[str, List[float]],
    embeddings=None
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation based on 2025 best practices.
    
    Args:
        retriever: The retriever to evaluate
        test_queries: List of test queries
        relevance_judgments: Relevance scores for each query
        embeddings: Optional embeddings for diversity calculation
        
    Returns:
        Comprehensive evaluation results
    """
    results = {
        "performance_metrics": {},
        "latency_metrics": {},
        "diversity_metrics": {},
        "ranking_metrics": {}
    }
    
    # Latency metrics
    results["latency_metrics"] = measure_latency(retriever, test_queries)
    
    # Performance and ranking metrics
    ndcg_scores = []
    mrr_scores = []
    all_retrieved_docs = []
    
    for query in test_queries:
        docs = retriever.invoke(query)
        all_retrieved_docs.extend(docs)
        
        if query in relevance_judgments:
            scores = relevance_judgments[query][:len(docs)]
            ndcg_scores.append(calculate_ndcg(scores))
            mrr_scores.append(calculate_mrr(scores))
    
    results["ranking_metrics"]["avg_ndcg@10"] = np.mean(ndcg_scores) if ndcg_scores else 0
    results["ranking_metrics"]["avg_mrr"] = np.mean(mrr_scores) if mrr_scores else 0
    
    # Diversity metrics
    results["diversity_metrics"] = measure_diversity(all_retrieved_docs, embeddings)
    
    return results
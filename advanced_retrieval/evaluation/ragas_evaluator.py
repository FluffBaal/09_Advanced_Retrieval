"""
RAGAS-based evaluation for retrieval methods.
"""

import time
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset


class RagasEvaluator:
    """Evaluator for retrieval methods using RAGAS metrics."""
    
    def __init__(
        self,
        llm: BaseLLM,
        embeddings: Embeddings,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            llm: Language model for evaluation
            embeddings: Embedding model for evaluation
            metrics: List of metric names to use (uses all if None)
        """
        self.llm_wrapper = LangchainLLMWrapper(llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)
        
        # Initialize metrics
        self.metric_instances = self._initialize_metrics(metrics)
    
    def _initialize_metrics(self, metric_names: Optional[List[str]] = None) -> List[Any]:
        """Initialize RAGAS metric instances."""
        all_metrics = {
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
            "context_relevance": ContextRelevance(),
            "answer_relevancy": AnswerRelevancy(),
            "faithfulness": Faithfulness(),
        }
        
        if metric_names is None:
            metric_names = list(all_metrics.keys())
        
        metrics = []
        for name in metric_names:
            if name in all_metrics:
                metric = all_metrics[name]
                # Configure metrics with LLM and embeddings
                if hasattr(metric, 'llm'):
                    metric.llm = self.llm_wrapper
                if hasattr(metric, 'embeddings'):
                    metric.embeddings = self.embeddings_wrapper
                metrics.append(metric)
        
        return metrics
    
    def evaluate_retriever(
        self,
        retriever: BaseRetriever,
        test_data: pd.DataFrame,
        name: str = "retriever"
    ) -> Dict[str, Any]:
        """
        Evaluate a single retriever using RAGAS metrics.
        
        Args:
            retriever: The retriever to evaluate
            test_data: Test dataset with columns: user_input, reference, reference_contexts
            name: Name of the retriever
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating {name}...")
        
        # Track timing
        start_time = time.time()
        
        # Prepare evaluation data
        eval_data = []
        for idx, row in test_data.iterrows():
            query = row['user_input']
            
            # Retrieve documents
            retrieved_docs = retriever.invoke(query)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            eval_data.append({
                'user_input': query,
                'reference': row['reference'],
                'reference_contexts': row['reference_contexts'],
                'retrieved_contexts': contexts,
                'response': row['reference']  # Using reference as response for evaluation
            })
        
        # Convert to dataset format expected by RAGAS
        eval_df = pd.DataFrame(eval_data)
        dataset = Dataset.from_pandas(eval_df)
        
        # Run evaluation
        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.metric_instances,
                llm=self.llm_wrapper,
                embeddings=self.embeddings_wrapper,
                column_map={
                    "question": "user_input",
                    "ground_truth": "reference",
                    "answer": "response",
                    "contexts": "retrieved_contexts"
                }
            )
            
            # Calculate timing
            end_time = time.time()
            latency = end_time - start_time
            
            # Prepare results
            scores = results.scores()
            scores['latency'] = latency
            scores['retriever_name'] = name
            
            return scores
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            return {
                'retriever_name': name,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def evaluate_multiple(
        self,
        retrievers: Dict[str, BaseRetriever],
        test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate multiple retrievers and return comparative results.
        
        Args:
            retrievers: Dictionary mapping names to retriever instances
            test_data: Test dataset
            
        Returns:
            DataFrame with comparative results
        """
        results = []
        
        for name, retriever in retrievers.items():
            result = self.evaluate_retriever(retriever, test_data, name)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_cost_estimates(
        self,
        retrievers: Dict[str, BaseRetriever],
        test_data: pd.DataFrame,
        cost_per_1k_tokens: float = 0.01
    ) -> Dict[str, float]:
        """
        Estimate costs for different retrievers.
        
        Args:
            retrievers: Dictionary of retrievers
            test_data: Test dataset
            cost_per_1k_tokens: Cost per 1000 tokens
            
        Returns:
            Dictionary mapping retriever names to estimated costs
        """
        costs = {}
        
        for name, retriever in retrievers.items():
            total_tokens = 0
            
            for idx, row in test_data.iterrows():
                query = row['user_input']
                docs = retriever.invoke(query)
                
                # Estimate tokens (rough approximation)
                for doc in docs:
                    total_tokens += len(doc.page_content.split()) * 1.3
            
            costs[name] = (total_tokens / 1000) * cost_per_1k_tokens
        
        return costs


def evaluate_retrievers(
    retrievers: Dict[str, BaseRetriever],
    test_data: pd.DataFrame,
    llm: BaseLLM,
    embeddings: Embeddings,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate multiple retrievers.
    
    Args:
        retrievers: Dictionary mapping names to retriever instances
        test_data: Test dataset with required columns
        llm: Language model for evaluation
        embeddings: Embedding model for evaluation
        metrics: List of metrics to use
        
    Returns:
        Dictionary with evaluation results and analysis
    """
    evaluator = RagasEvaluator(llm, embeddings, metrics)
    
    # Run evaluations
    results_df = evaluator.evaluate_multiple(retrievers, test_data)
    
    # Calculate costs
    costs = evaluator.calculate_cost_estimates(retrievers, test_data)
    
    # Prepare summary
    summary = {
        'results': results_df,
        'costs': costs,
        'best_by_metric': {},
        'recommendations': []
    }
    
    # Find best performer for each metric
    numeric_columns = results_df.select_dtypes(include=['float64', 'int64']).columns
    for metric in numeric_columns:
        if metric != 'latency':  # Higher is better for most metrics
            best_idx = results_df[metric].idxmax()
        else:  # Lower is better for latency
            best_idx = results_df[metric].idxmin()
        
        summary['best_by_metric'][metric] = results_df.loc[best_idx, 'retriever_name']
    
    # Generate recommendations
    if 'context_precision' in numeric_columns and 'latency' in numeric_columns:
        # Balance performance and speed
        results_df['efficiency_score'] = (
            results_df['context_precision'] / results_df['latency']
        )
        most_efficient = results_df.loc[
            results_df['efficiency_score'].idxmax(), 
            'retriever_name'
        ]
        summary['recommendations'].append(
            f"{most_efficient} offers the best balance of performance and speed"
        )
    
    return summary
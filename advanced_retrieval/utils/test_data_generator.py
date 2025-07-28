"""
Test data generation using RAGAS synthetic data generation.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def generate_test_data(
    documents: List[Document],
    llm: BaseLLM,
    embeddings: Embeddings,
    num_samples: int = 20,
    distributions: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Generate synthetic test data using RAGAS.
    
    Args:
        documents: List of documents to generate questions from
        llm: Language model for generation
        embeddings: Embedding model
        num_samples: Number of test samples to generate
        distributions: Distribution of question types
        
    Returns:
        DataFrame with test data (user_input, reference, reference_contexts)
    """
    # Wrap models for RAGAS
    llm_wrapper = LangchainLLMWrapper(llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)
    
    # Set default distributions
    if distributions is None:
        distributions = None  # Let Ragas use defaults
    
    # Create test generator
    generator = TestsetGenerator(
        llm=llm_wrapper,
        embedding_model=embeddings_wrapper
    )
    
    # Generate test set
    try:
        # Generate test set
        if distributions:
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=num_samples,
                distributions=distributions,
                with_debugging_logs=False
            )
        else:
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=num_samples,
                with_debugging_logs=False
            )
        
        # Convert to DataFrame format expected by evaluator
        test_df = testset.to_pandas()
        
        # Ensure required columns exist
        if 'question' in test_df.columns:
            test_df['user_input'] = test_df['question']
        if 'ground_truth' in test_df.columns:
            test_df['reference'] = test_df['ground_truth']
        if 'contexts' in test_df.columns:
            test_df['reference_contexts'] = test_df['contexts']
        
        # Select only required columns
        required_columns = ['user_input', 'reference', 'reference_contexts']
        test_df = test_df[required_columns]
        
        return test_df
        
    except Exception as e:
        print(f"Error generating test data: {str(e)}")
        # Fallback to manual generation
        return generate_manual_test_data(documents, num_samples)


def generate_manual_test_data(
    documents: List[Document],
    num_samples: int = 20
) -> pd.DataFrame:
    """
    Manually generate test data as a fallback.
    
    Args:
        documents: List of documents
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with test data
    """
    import random
    random.seed(42)
    
    test_data = []
    sample_docs = random.sample(documents, min(num_samples * 2, len(documents)))
    
    for i, doc in enumerate(sample_docs[:num_samples]):
        content = doc.page_content
        metadata = doc.metadata
        
        # Generate different types of questions
        if i % 3 == 0:
            # Product-related question
            if "Product:" in content:
                product = content.split("Product:")[1].split("\n")[0].strip()
                test_data.append({
                    "user_input": f"What complaints are there about {product}?",
                    "reference": content,
                    "reference_contexts": [content]
                })
        elif i % 3 == 1:
            # Issue-related question
            if "Issue:" in content:
                issue = content.split("Issue:")[1].split("\n")[0].strip()
                test_data.append({
                    "user_input": f"What are the main concerns regarding {issue}?",
                    "reference": content,
                    "reference_contexts": [content]
                })
        else:
            # Company-related question
            if "Company:" in content:
                company = content.split("Company:")[1].split("\n")[0].strip()
                test_data.append({
                    "user_input": f"What issues have been reported about {company}?",
                    "reference": content,
                    "reference_contexts": [content]
                })
    
    # Ensure we have enough samples
    while len(test_data) < num_samples:
        doc = random.choice(documents)
        test_data.append({
            "user_input": "What are the main complaint categories in the dataset?",
            "reference": doc.page_content,
            "reference_contexts": [doc.page_content]
        })
    
    return pd.DataFrame(test_data[:num_samples])


def generate_diverse_test_questions(
    documents: List[Document],
    llm: BaseLLM,
    num_questions: int = 50
) -> List[Dict[str, Any]]:
    """
    Generate diverse test questions for comprehensive evaluation.
    
    Args:
        documents: Source documents
        llm: Language model for generation
        num_questions: Number of questions to generate
        
    Returns:
        List of question dictionaries
    """
    question_types = [
        "factual",
        "analytical", 
        "comparative",
        "summary",
        "specific_detail"
    ]
    
    questions = []
    for i in range(num_questions):
        q_type = question_types[i % len(question_types)]
        doc = documents[i % len(documents)]
        
        question = {
            "type": q_type,
            "document": doc,
            "metadata": doc.metadata
        }
        
        # Generate type-specific questions
        if q_type == "factual":
            question["user_input"] = f"What product is mentioned in complaint {doc.metadata.get('complaint_id', i)}?"
        elif q_type == "analytical":
            question["user_input"] = "What patterns can be identified in the complaints?"
        elif q_type == "comparative":
            question["user_input"] = "How do complaints differ between products?"
        elif q_type == "summary":
            question["user_input"] = "Summarize the main issues reported"
        else:
            question["user_input"] = f"What specific issue was reported by the consumer?"
        
        questions.append(question)
    
    return questions
"""
Data loading utilities for retrieval evaluation.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(
    file_path: str,
    text_column: str = "text",
    metadata_columns: Optional[List[str]] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Load documents from a CSV file and split into chunks.
    
    Args:
        file_path: Path to the CSV file
        text_column: Column containing the main text
        metadata_columns: Columns to include as metadata
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Create documents
    documents = []
    for idx, row in df.iterrows():
        # Get text content
        content = str(row[text_column])
        
        # Build metadata
        metadata = {"row_index": idx}
        if metadata_columns:
            for col in metadata_columns:
                if col in row:
                    metadata[col] = str(row[col])
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Split documents if needed
    if chunk_size > 0:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        documents = splitter.split_documents(documents)
    
    return documents


def load_loan_complaints_data(
    file_path: str = "data/loan_complaints_processed.csv",
    sample_size: Optional[int] = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Load loan complaints data specifically for the assignment.
    
    Args:
        file_path: Path to the loan complaints CSV
        sample_size: Number of samples to load (None for all)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary with documents and metadata
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Create documents with relevant metadata
    documents = []
    for idx, row in df.iterrows():
        # Combine complaint text
        complaint_text = f"Issue: {row.get('Issue', 'N/A')}\n"
        complaint_text += f"Product: {row.get('Product', 'N/A')}\n"
        complaint_text += f"Company: {row.get('Company', 'N/A')}\n"
        complaint_text += f"Complaint: {row.get('Consumer complaint narrative', 'N/A')}"
        
        metadata = {
            "complaint_id": row.get('Complaint ID', idx),
            "product": row.get('Product', 'N/A'),
            "issue": row.get('Issue', 'N/A'),
            "company": row.get('Company', 'N/A'),
            "state": row.get('State', 'N/A'),
            "date_received": row.get('Date received', 'N/A')
        }
        
        documents.append(Document(page_content=complaint_text, metadata=metadata))
    
    # Split documents
    if chunk_size > 0:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_documents = splitter.split_documents(documents)
    else:
        split_documents = documents
    
    return {
        "documents": split_documents,
        "original_documents": documents,
        "dataframe": df,
        "metadata": {
            "total_complaints": len(df),
            "unique_products": df['Product'].nunique() if 'Product' in df else 0,
            "unique_companies": df['Company'].nunique() if 'Company' in df else 0,
            "total_chunks": len(split_documents)
        }
    }


def prepare_evaluation_data(
    documents: List[Document],
    num_samples: int = 20
) -> pd.DataFrame:
    """
    Prepare data for evaluation by selecting representative samples.
    
    Args:
        documents: List of documents
        num_samples: Number of samples to prepare
        
    Returns:
        DataFrame with evaluation samples
    """
    # Select diverse samples
    import random
    random.seed(42)
    
    sample_docs = random.sample(documents, min(num_samples, len(documents)))
    
    # Create evaluation data
    eval_data = []
    for doc in sample_docs:
        # Extract meaningful questions from the document
        content = doc.page_content
        metadata = doc.metadata
        
        # Create sample questions based on content
        if "Issue:" in content:
            issue = content.split("Issue:")[1].split("\n")[0].strip()
            eval_data.append({
                "user_input": f"What issues are related to {metadata.get('product', 'this product')}?",
                "reference": issue,
                "reference_contexts": [content]
            })
        
        if "Company:" in content:
            company = content.split("Company:")[1].split("\n")[0].strip()
            eval_data.append({
                "user_input": f"Which companies have complaints about {metadata.get('issue', 'this issue')}?",
                "reference": company,
                "reference_contexts": [content]
            })
    
    return pd.DataFrame(eval_data)
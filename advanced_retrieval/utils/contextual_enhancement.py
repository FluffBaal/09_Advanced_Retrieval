"""
Contextual enhancement utilities based on 2025 best practices.

Implements Contextual Chunk Headers (CCH) and other enhancement techniques.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import re


def add_contextual_chunk_headers(
    documents: List[Document],
    include_section: bool = True,
    include_document_summary: bool = True
) -> List[Document]:
    """
    Add Contextual Chunk Headers (CCH) to documents.
    
    Based on 2025 best practices, this prepends document and section context
    to each chunk for improved retrieval accuracy.
    
    Args:
        documents: List of documents to enhance
        include_section: Whether to include section headers
        include_document_summary: Whether to include document summary
        
    Returns:
        List of documents with contextual headers
    """
    enhanced_docs = []
    
    for doc in documents:
        # Extract metadata
        metadata = doc.metadata
        
        # Build contextual header
        header_parts = []
        
        # Add document-level context
        if include_document_summary:
            if "source" in metadata:
                header_parts.append(f"[Source: {metadata['source']}]")
            if "Product" in metadata:
                header_parts.append(f"[Product: {metadata['Product']}]")
            if "Company" in metadata:
                header_parts.append(f"[Company: {metadata['Company']}]")
        
        # Add section context
        if include_section and "Issue" in metadata:
            header_parts.append(f"[Issue: {metadata['Issue']}]")
        
        # Create header
        if header_parts:
            header = " ".join(header_parts) + "\n\n"
            enhanced_content = header + doc.page_content
        else:
            enhanced_content = doc.page_content
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata={**metadata, "has_contextual_header": True}
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs


def extract_key_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract key entities from text for contextual enhancement.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of entity types and their values
    """
    entities = {
        "companies": [],
        "products": [],
        "issues": [],
        "dates": [],
        "amounts": []
    }
    
    # Simple pattern matching for loan complaint context
    # Company names (capitalized multi-word phrases)
    company_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,?\s+(?:Inc|LLC|Corp|Services|Federal)\.?)?)\b"
    entities["companies"] = list(set(re.findall(company_pattern, text)))
    
    # Product types
    product_keywords = ["student loan", "federal loan", "private loan", "mortgage", "credit"]
    for keyword in product_keywords:
        if keyword.lower() in text.lower():
            entities["products"].append(keyword)
    
    # Issue types
    issue_keywords = ["payment", "servicer", "forbearance", "credit report", "discharge", "forgiveness"]
    for keyword in issue_keywords:
        if keyword.lower() in text.lower():
            entities["issues"].append(keyword)
    
    # Dates (simple pattern)
    date_pattern = r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    entities["dates"] = re.findall(date_pattern, text)
    
    # Amounts
    amount_pattern = r"\$[\d,]+(?:\.\d{2})?"
    entities["amounts"] = re.findall(amount_pattern, text)
    
    return entities


def create_semantic_summary(
    document: Document,
    max_length: int = 100
) -> str:
    """
    Create a semantic summary of a document for contextual enhancement.
    
    Args:
        document: Input document
        max_length: Maximum summary length
        
    Returns:
        Semantic summary string
    """
    content = document.page_content
    metadata = document.metadata
    
    # Extract key information
    entities = extract_key_entities(content)
    
    # Build summary
    summary_parts = []
    
    # Add product/issue context
    if metadata.get("Product"):
        summary_parts.append(f"{metadata['Product']} complaint")
    
    if metadata.get("Issue"):
        summary_parts.append(f"regarding {metadata['Issue']}")
    
    # Add company if present
    if metadata.get("Company"):
        summary_parts.append(f"with {metadata['Company']}")
    elif entities["companies"]:
        summary_parts.append(f"with {entities['companies'][0]}")
    
    # Add key entities found
    if entities["amounts"]:
        summary_parts.append(f"involving {entities['amounts'][0]}")
    
    summary = " ".join(summary_parts)
    
    # Truncate if needed
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return summary


def enhance_documents_with_summaries(
    documents: List[Document]
) -> List[Document]:
    """
    Enhance documents with semantic summaries.
    
    Args:
        documents: List of documents to enhance
        
    Returns:
        List of enhanced documents
    """
    enhanced_docs = []
    
    for doc in documents:
        summary = create_semantic_summary(doc)
        
        # Add summary to metadata
        enhanced_metadata = {
            **doc.metadata,
            "semantic_summary": summary
        }
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=enhanced_metadata
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs


def apply_relevant_segment_extraction(
    documents: List[Document],
    query: str,
    window_size: int = 2
) -> List[Document]:
    """
    Apply Relevant Segment Extraction (RSE) to documents.
    
    Dynamically constructs multi-chunk segments based on query relevance.
    
    Args:
        documents: Retrieved documents
        query: User query
        window_size: Number of adjacent chunks to include
        
    Returns:
        List of documents with expanded context
    """
    # This is a simplified version - in production, you'd want
    # to track chunk relationships and expand context intelligently
    
    query_terms = set(query.lower().split())
    enhanced_docs = []
    
    for doc in documents:
        # Calculate relevance score
        content_terms = set(doc.page_content.lower().split())
        relevance = len(query_terms & content_terms) / len(query_terms)
        
        # Add relevance to metadata
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "query_relevance": relevance,
                "extraction_method": "RSE"
            }
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs
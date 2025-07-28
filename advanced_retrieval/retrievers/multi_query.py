"""
Multi-query retriever that generates multiple query variations.
"""

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM


def create_multi_query_retriever(
    base_retriever: BaseRetriever,
    llm: BaseLLM,
    parser_key: str = "lines",
    include_original: bool = True
) -> MultiQueryRetriever:
    """
    Create a multi-query retriever that generates query variations.
    
    This retriever uses an LLM to generate multiple variations of the user's
    query, then retrieves documents for each variation and combines the results.
    
    Args:
        base_retriever: The base retriever to use for each query
        llm: Language model to generate query variations
        parser_key: Key for parsing LLM output (default: "lines")
        include_original: Whether to include the original query (default: True)
        
    Returns:
        A MultiQueryRetriever instance
    """
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        parser_key=parser_key,
        include_original=include_original
    )
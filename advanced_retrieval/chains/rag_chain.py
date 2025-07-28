"""
RAG chain implementations for different retrieval methods.
"""

from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
# Remove unused import


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseLLM,
    prompt_template: Optional[str] = None,
    chain_type: str = "stuff"
) -> Any:
    """
    Create a RAG chain with the specified retriever and LLM.
    
    Args:
        retriever: The retriever to use
        llm: The language model
        prompt_template: Optional custom prompt template
        chain_type: Type of chain ("stuff", "map_reduce", "refine", "map_rerank")
        
    Returns:
        A runnable RAG chain
    """
    if prompt_template is None:
        prompt_template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """
    
    if chain_type == "stuff":
        # Create prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    else:
        # Use RetrievalQA for other chain types
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )


class RAGChainFactory:
    """Factory for creating RAG chains with different configurations."""
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize the RAG chain factory.
        
        Args:
            llm: The language model to use
        """
        self.llm = llm
        self.prompt_templates = {
            "default": """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """,
            
            "detailed": """You are a helpful assistant answering questions about loan complaints.
            
Based on the following context, provide a detailed and accurate answer to the question.
If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {question}

Detailed Answer: """,
            
            "concise": """Based on the context below, provide a brief answer to the question.

Context: {context}

Question: {question}

Brief Answer: """,
            
            "analytical": """Analyze the following information and provide insights related to the question.

Context Information:
{context}

Question to Analyze: {question}

Analysis: """
        }
    
    def create_chain(
        self,
        retriever: BaseRetriever,
        prompt_style: str = "default",
        chain_type: str = "stuff",
        custom_prompt: Optional[str] = None
    ) -> Any:
        """
        Create a RAG chain with specified configuration.
        
        Args:
            retriever: The retriever to use
            prompt_style: Style of prompt ("default", "detailed", "concise", "analytical")
            chain_type: Type of chain to create
            custom_prompt: Optional custom prompt template
            
        Returns:
            A configured RAG chain
        """
        # Select prompt template
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = self.prompt_templates.get(prompt_style, self.prompt_templates["default"])
        
        return create_rag_chain(
            retriever=retriever,
            llm=self.llm,
            prompt_template=prompt_template,
            chain_type=chain_type
        )
    
    def create_multi_retriever_chain(
        self,
        retrievers: Dict[str, BaseRetriever],
        router_llm: Optional[BaseLLM] = None
    ) -> Any:
        """
        Create a chain that routes to different retrievers based on query type.
        
        Args:
            retrievers: Dictionary mapping retriever names to instances
            router_llm: Optional separate LLM for routing (uses main LLM if None)
            
        Returns:
            A multi-retriever chain
        """
        if router_llm is None:
            router_llm = self.llm
        
        # Create routing prompt
        routing_prompt = ChatPromptTemplate.from_template("""
Analyze the following question and determine which retriever would be most appropriate:

Available retrievers:
- naive: Best for simple, direct questions
- bm25: Best for keyword-based searches
- compression: Best for finding highly relevant information
- multi_query: Best for complex questions that may have multiple aspects
- parent_document: Best for questions requiring broader context
- ensemble: Best for comprehensive searches

Question: {question}

Return only the retriever name (lowercase, no extra text):
""")
        
        # Create router chain
        router = routing_prompt | router_llm | StrOutputParser()
        
        # Create individual chains for each retriever
        chains = {
            name: self.create_chain(retriever)
            for name, retriever in retrievers.items()
        }
        
        # Create routing function
        def route_question(question: str) -> Any:
            retriever_name = router.invoke({"question": question}).strip().lower()
            
            # Fallback to ensemble if invalid selection
            if retriever_name not in chains:
                retriever_name = "ensemble" if "ensemble" in chains else "naive"
            
            return chains[retriever_name].invoke(question)
        
        return route_question
    
    def create_hybrid_chain(
        self,
        retrievers: List[BaseRetriever],
        aggregation_method: str = "concatenate"
    ) -> Any:
        """
        Create a hybrid chain that uses multiple retrievers in parallel.
        
        Args:
            retrievers: List of retrievers to use
            aggregation_method: How to combine results ("concatenate", "rerank", "vote")
            
        Returns:
            A hybrid RAG chain
        """
        from langchain.retrievers import MergerRetriever
        
        # Create merger retriever
        if aggregation_method == "concatenate":
            merger = MergerRetriever(retrievers=retrievers)
        else:
            # For other methods, we'd implement custom logic
            merger = MergerRetriever(retrievers=retrievers)
        
        return self.create_chain(merger, prompt_style="detailed")
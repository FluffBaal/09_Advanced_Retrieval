# Advanced Retrieval Library

A comprehensive Python library for implementing and evaluating advanced retrieval methods for Retrieval-Augmented Generation (RAG) systems.

## Features

- **Multiple Retriever Types**:
  - Naive (cosine similarity)
  - BM25 (keyword-based)
  - Contextual Compression (with reranking)
  - Multi-Query (query expansion)
  - Parent Document (small-to-big retrieval)
  - Ensemble (combining multiple methods)

- **RAGAS-based Evaluation**:
  - Context Precision
  - Context Recall
  - Context Relevance
  - Answer Relevancy
  - Faithfulness

- **RAG Chain Support**:
  - Multiple prompt styles
  - Chain routing
  - Hybrid approaches

## Installation

```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv add langchain langchain-community langchain-openai ragas pandas numpy
```

## Quick Start

```python
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.utils import load_loan_complaints_data
from langchain_openai import OpenAIEmbeddings

# Load data
data = load_loan_complaints_data("data/loan_complaints.csv")
documents = data["documents"]

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create retriever factory
factory = RetrieverFactory(documents, embeddings)

# Create retrievers
naive_retriever = factory.create_naive(k=5)
bm25_retriever = factory.create_bm25(k=5)
ensemble_retriever = factory.create_ensemble(["naive", "bm25"])

# Use retriever
results = naive_retriever.get_relevant_documents("student loan issues")
```

## Library Structure

```
advanced_retrieval/
├── __init__.py              # Main package exports
├── retrievers/              # Retriever implementations
│   ├── __init__.py
│   ├── naive.py            # Naive cosine similarity
│   ├── bm25.py             # BM25 keyword retrieval
│   ├── compression.py      # Contextual compression
│   ├── multi_query.py      # Multi-query generation
│   ├── parent_document.py  # Parent document retrieval
│   ├── ensemble.py         # Ensemble methods
│   └── factory.py          # Factory for creating retrievers
├── evaluation/              # Evaluation tools
│   ├── __init__.py
│   └── ragas_evaluator.py  # RAGAS-based evaluation
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── test_data_generator.py  # Synthetic test data
└── chains/                  # RAG chain implementations
    ├── __init__.py
    └── rag_chain.py        # RAG chain factory
```

## Examples

### 1. Basic Retriever Usage

```python
from advanced_retrieval import RetrieverFactory
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize
embeddings = HuggingFaceEmbeddings()
factory = RetrieverFactory(documents, embeddings)

# Create and use retriever
retriever = factory.create_bm25(k=10)
docs = retriever.get_relevant_documents("loan servicing issues")
```

### 2. Evaluating Retrievers

```python
from advanced_retrieval.evaluation import evaluate_retrievers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize models
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Create retrievers
retrievers = factory.create_all_retrievers(llm)

# Evaluate
results = evaluate_retrievers(
    retrievers=retrievers,
    test_data=test_df,
    llm=llm,
    embeddings=embeddings
)

print(results['results'])  # Performance metrics
print(results['costs'])    # Cost estimates
```

### 3. RAG Chain Creation

```python
from advanced_retrieval.chains import RAGChainFactory

# Create chain factory
chain_factory = RAGChainFactory(llm)

# Create chain with specific style
chain = chain_factory.create_chain(
    retriever=ensemble_retriever,
    prompt_style="analytical"
)

# Use chain
response = chain.invoke("What are the main complaint categories?")
```

## Advanced Usage

### Custom Retriever Configuration

```python
# Create parent document retriever with custom chunk size
parent_retriever = factory.create_parent_document(
    chunk_size=500,
    chunk_overlap=50,
    k=15
)

# Create ensemble with custom weights
ensemble = factory.create_ensemble(
    retriever_types=["naive", "bm25", "compression"],
    weights=[0.3, 0.3, 0.4]
)
```

### Multi-Retriever Routing

```python
# Create routing chain
all_retrievers = factory.create_all_retrievers(llm)
routing_chain = chain_factory.create_multi_retriever_chain(all_retrievers)

# Automatically routes to best retriever
response = routing_chain("specific complaint about Wells Fargo")
```

## Evaluation Metrics

The library uses RAGAS (Retrieval Augmented Generation Assessment) metrics:

- **Context Precision**: How relevant are the retrieved contexts?
- **Context Recall**: How much of the required information is retrieved?
- **Context Relevance**: How relevant is each retrieved context?
- **Answer Relevancy**: How relevant is the generated answer?
- **Faithfulness**: How faithful is the answer to the retrieved contexts?

## Cost Estimation

The library includes cost estimation for different retrieval methods:

```python
costs = evaluator.calculate_cost_estimates(
    retrievers=retrievers,
    test_data=test_data,
    cost_per_1k_tokens=0.01
)
```

## Best Practices

1. **Choose the Right Retriever**:
   - Use BM25 for keyword-heavy queries
   - Use Multi-Query for complex, ambiguous questions
   - Use Ensemble for best overall performance

2. **Optimize Performance**:
   - Adjust chunk sizes based on your content
   - Use appropriate k values (typically 5-20)
   - Consider latency vs accuracy trade-offs

3. **Evaluation**:
   - Always evaluate on representative test data
   - Consider multiple metrics, not just accuracy
   - Factor in cost and latency requirements

## Requirements

- Python 3.8+
- LangChain
- RAGAS
- OpenAI API key (for OpenAI models)
- HuggingFace models (for local embeddings)

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
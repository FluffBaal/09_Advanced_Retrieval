# Advanced Retrieval Library - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Set Up API Keys

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 2. Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or pip:
```bash
pip install -r requirements.txt
```

### 3. Test the Installation

```bash
uv run python test_simple_workflow.py
```

You should see:
```
✓ All basic tests passed!
The library is working correctly.
```

## 📚 Basic Usage

### Simple Retrieval Example

```python
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.utils import load_loan_complaints_data
from langchain_openai import OpenAIEmbeddings

# Load data
data = load_loan_complaints_data("data/complaints.csv")
documents = data["documents"]

# Create retriever
embeddings = OpenAIEmbeddings()
factory = RetrieverFactory(documents, embeddings)
retriever = factory.create_bm25(k=5)

# Search
results = retriever.invoke("student loan payment issues")
for doc in results:
    print(doc.page_content[:200])
```

### RAG Chain Example

```python
from advanced_retrieval.chains import RAGChainFactory
from langchain_openai import ChatOpenAI

# Create RAG chain
llm = ChatOpenAI()
chain_factory = RAGChainFactory(llm)
rag_chain = chain_factory.create_chain(retriever)

# Ask questions
answer = rag_chain.invoke("What are the main issues with student loans?")
print(answer)
```

## 🔍 Available Retrievers

1. **Naive** - Basic cosine similarity search
   ```python
   retriever = factory.create_naive(k=5)
   ```

2. **BM25** - Keyword-based retrieval
   ```python
   retriever = factory.create_bm25(k=5)
   ```

3. **Contextual Compression** - Reranking with Cohere
   ```python
   retriever = factory.create_compression()  # Requires COHERE_API_KEY
   ```

4. **Multi-Query** - Generates query variations
   ```python
   retriever = factory.create_multi_query(llm)
   ```

5. **Parent Document** - Small-to-big retrieval
   ```python
   retriever = factory.create_parent_document()
   ```

6. **Ensemble** - Combines multiple retrievers
   ```python
   retriever = factory.create_ensemble(["naive", "bm25"])
   ```

## 📊 Evaluation

### Simple Evaluation
```bash
uv run python examples/simple_evaluation.py
```

### Full RAGAS Evaluation
```bash
uv run python examples/evaluation_example.py
```

## 🛠️ Troubleshooting

### "API key not found"
- Check `.env` file exists and contains your key
- Run `python load_env.py` to verify

### "Module not found"
- Ensure you're using `uv run` or activated the virtual environment
- Try `uv sync` to reinstall dependencies

### "Out of memory"
- Reduce sample size in data loading
- Use smaller chunks: `chunk_size=500`

## 📖 Examples

- `examples/basic_usage.py` - Simple retriever usage
- `examples/simple_evaluation.py` - Quick evaluation without RAGAS
- `examples/rag_chain_example.py` - Different RAG chain styles
- `examples/full_workflow.py` - Complete pipeline with RAGAS

## 🎯 Best Practices

1. **Start Simple**: Begin with BM25 or Naive retriever
2. **Test Small**: Use `sample_size=100` for quick tests
3. **Monitor Costs**: OpenAI API calls can add up
4. **Cache Results**: Reuse embeddings when possible

## 📚 Next Steps

1. Read the full documentation: `README_library.md`
2. Explore different retriever combinations
3. Customize prompt templates for your use case
4. Integrate into your own applications

## 🤝 Getting Help

- Check `API_KEY_SETUP.md` for API configuration
- Review `LIBRARY_SUMMARY.md` for architecture details
- See `explained.md` for understanding the original notebook

Happy retrieving! 🎉
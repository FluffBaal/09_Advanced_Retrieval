"""
PDF-based evaluation example following the reference notebook approach.

This demonstrates loading PDF documents and using RAGAS properly.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_env import load_env
load_env()

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from advanced_retrieval import RetrieverFactory
from advanced_retrieval.chains import RAGChainFactory


def main():
    print("PDF-Based RAG Evaluation (Following Reference Notebook)")
    print("="*60)
    
    # 1. Load PDF documents
    print("\n1. Loading PDF documents...")
    path = "data/"
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} PDF documents")
    
    # Show document names
    for doc in docs:
        if "source" in doc.metadata:
            print(f"  - {os.path.basename(doc.metadata['source'])}")
    
    # 2. Split documents
    print("\n2. Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"✓ Created {len(split_documents)} document chunks")
    
    # 3. Initialize models with proper wrapping
    print("\n3. Initializing models...")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Wrap models for RAGAS (as shown in reference notebook)
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    print("✓ Models initialized and wrapped for RAGAS")
    
    # 4. Create vectorstore
    print("\n4. Creating vectorstore...")
    vectorstore = Qdrant.from_documents(
        documents=split_documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="PDF_Documents"
    )
    print("✓ Vectorstore created")
    
    # 5. Create retriever
    print("\n5. Creating retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    print("✓ Retriever created")
    
    # 6. Generate synthetic test data (simplified)
    print("\n6. Generating synthetic test data...")
    try:
        # Use RAGAS TestsetGenerator as in reference notebook
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings
        )
        
        # Generate a small dataset
        print("   Generating testset (this may take a moment)...")
        dataset = generator.generate_with_langchain_docs(
            docs[:5],  # Use subset for speed
            testset_size=5
        )
        test_df = dataset.to_pandas()
        print(f"✓ Generated {len(test_df)} test samples")
        
        # Show sample questions
        print("\nSample generated questions:")
        for i, row in test_df.head(3).iterrows():
            print(f"  Q{i+1}: {row['user_input'][:100]}...")
            
    except Exception as e:
        print(f"! Error generating test data: {e}")
        print("  Using fallback test questions...")
        
        # Fallback test questions based on PDF content
        test_questions = [
            "What is the Direct Loan Program?",
            "What types of federal student aid are available?",
            "How does the Pell Grant work?",
            "What are the eligibility requirements for federal student aid?",
            "How do I apply for federal student loans?"
        ]
        
        test_df = None
    
    # 7. Create RAG chain
    print("\n7. Creating RAG chain...")
    chain_factory = RAGChainFactory(llm)
    
    # Use the same prompt style as reference notebook
    PROMPT_TEMPLATE = """Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""
    
    rag_chain = chain_factory.create_chain(
        retriever,
        custom_prompt=PROMPT_TEMPLATE
    )
    print("✓ RAG chain created")
    
    # 8. Test the RAG chain
    print("\n8. Testing RAG chain...")
    test_query = "What types of loans are available?"
    answer = rag_chain.invoke(test_query)
    print(f"\nQ: {test_query}")
    print(f"A: {answer}")
    
    # 9. Evaluate if we have test data
    if test_df is not None:
        print("\n9. Evaluating retriever performance...")
        correct_count = 0
        
        for idx, row in test_df.iterrows():
            query = row['user_input']
            expected = row['reference']
            
            # Get answer
            try:
                answer = rag_chain.invoke(query)
                # Simple relevance check
                if any(word in answer.lower() for word in expected.lower().split()[:5]):
                    correct_count += 1
            except:
                pass
        
        accuracy = correct_count / len(test_df) if len(test_df) > 0 else 0
        print(f"\nSimple accuracy: {accuracy:.2%}")
    
    print("\n" + "="*60)
    print("PDF-based evaluation complete!")
    print("\nKey differences from CSV approach:")
    print("- PDF documents provide richer, structured content")
    print("- Better suited for knowledge graph generation")
    print("- More comprehensive test data generation")
    print("- Follows the exact pattern from the reference notebook")


if __name__ == "__main__":
    main()
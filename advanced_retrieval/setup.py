"""
Setup script for the Advanced Retrieval library.
"""

from setuptools import setup, find_packages

with open("README_library.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced-retrieval",
    version="0.1.0",
    author="Advanced Retrieval Contributors",
    description="A comprehensive library for advanced retrieval methods in RAG systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-retrieval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "langchain-cohere>=0.1.0",
        "ragas>=0.3.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
        "sentence-transformers>=2.2.0",
        "qdrant-client>=1.7.0",
        "rank-bm25>=0.2.2",
        "pillow>=11.3.0",
        "rapidfuzz>=3.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
)
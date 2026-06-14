from setuptools import setup, find_packages

setup(
    name="rag-refresh-pipeline",
    version="0.1.0",
    description="RAG refresh pipeline with Qdrant, hybrid search, reranking, eval, and monitoring",
    author="Your Name",
    packages=find_packages(include=["dags", "dags.*", "streamlit_app", "streamlit_app.*"]),
    python_requires=">=3.11",
    install_requires=[],
)
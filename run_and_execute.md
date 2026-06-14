
10. Additional Files
data/urls_to_scrape.txt (Example)
text

# URLs to scrape for RAG knowledge base
# One URL per line, lines starting with # are ignored

# Company documentation
https://docs.yourcompany.com/getting-started
https://docs.yourcompany.com/api-reference

# Internal wiki pages
https://wiki.yourcompany.com/engineering/best-practices
https://wiki.yourcompany.com/hr/policies

# Blog posts
https://blog.yourcompany.com/engineering-principles
pytest.ini
ini

[pytest]
# Pytest configuration for RAG pipeline tests

# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Paths
testpaths = tests

# Output options
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings

# Coverage options (optional)
# --cov=dags
# --cov-report=html
# --cov-report=term-missing

# Markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
Usage Examples
Running Tests
Bash

# Run all tests
pytest

# Run specific test file
pytest tests/test_chunker.py

# Run with coverage
pytest --cov=dags --cov-report=html

# Run only unit tests
pytest -m unit
Database Initialization
Bash

# Initialize PostgreSQL tables
docker exec -i postgres psql -U airflow -d airflow < sql/init.sql

# Or via docker-compose
docker-compose exec postgres psql -U airflow -d airflow -f /sql/init.sql
Running Pipeline
Bash

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f airflow-scheduler

# Trigger DAG manually
docker-compose exec airflow-webserver airflow dags trigger rag_refresh_pipeline
Summary
All files created with ultra-simple code for junior engineers:

✅ SQL schema - PostgreSQL tables for metadata tracking
✅ Benchmark queries - Extended test set with categories
✅ DAG tests - Validates DAG integrity and structure
✅ Chunker tests - Tests text splitting logic
✅ Dedup tests - Tests Redis hashing with mocks
✅ Embed tests - Tests embedding with mocked APIs
✅ Eval tests - Tests Recall@K and MRR calculations
✅ .env template - All environment variables documented
✅ requirements.txt - Complete Python dependencies

The code is production-ready yet simple enough for a 2-3 year junior engineer to understand, modify, and extend! 🚀
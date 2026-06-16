#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# pre_build_check.sh  —  Test every tool standalone, BEFORE docker build
#
# Run section by section (copy-paste), or the whole file:
#     chmod +x pre_build_check.sh
#     ./pre_build_check.sh
#
# Each section is independent. Skip ones you don't need right now.
# ════════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"          # run from 02_RAG_Airflow/

echo "════════════════════════════════════════════════════════════"
echo " 1) STREAMLIT  — does the UI script import/run cleanly?"
echo "════════════════════════════════════════════════════════════"
echo "Run this, then open http://localhost:8501 and Ctrl+C when done:"
echo ""
echo "  QDRANT_HOST=localhost QDRANT_PORT=6333 \\"
echo "  RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \\"
echo "  streamlit run streamlit_app/app.py"
echo ""
echo "If it crashes on import (ModuleNotFoundError etc.) you'll see it"
echo "instantly in the terminal — no need to wait for docker build."
echo ""

echo "════════════════════════════════════════════════════════════"
echo " 2) AIRFLOW  — does the DAG parse within the timeout?"
echo "    (this reproduces the EXACT 'Broken DAG' / timeout error"
echo "     locally, against a throwaway sqlite metadata db)"
echo "════════════════════════════════════════════════════════════"
cat << 'EOF'

  # One-time setup (heavier install — only needed once):
  pip install "apache-airflow==2.8.1" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.11.txt"

  export AIRFLOW_HOME=/tmp/airflow_test
  export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
  export AIRFLOW__CORE__LOAD_EXAMPLES=false
  export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow_test/airflow.db
  export PYTHONPATH="$(pwd)/dags:$PYTHONPATH"

  airflow db migrate

  # THE key check — must finish well under 30s and print nothing:
  time airflow dags list-import-errors

  # Bonus — confirms the DAG + all its tasks are registered:
  airflow dags list
  airflow tasks list rag_refresh_pipeline

EOF
echo ""

echo "════════════════════════════════════════════════════════════"
echo " 3) MLFLOW  — does the binary start at all? (isolated test,"
echo "    uses sqlite + local folder, no Postgres/Docker needed)"
echo "════════════════════════════════════════════════════════════"
cat << 'EOF'

  mlflow server \
    --backend-store-uri sqlite:////tmp/mlflow_test.db \
    --default-artifact-root /tmp/mlflow_artifacts_test \
    --host 0.0.0.0 --port 5050 &
  MLFLOW_PID=$!

  sleep 3
  curl -sf http://localhost:5050/health && echo " <- mlflow OK"
  kill $MLFLOW_PID

EOF
echo "If this fails, your mlflow install itself is broken — fix that"
echo "BEFORE blaming docker networking for the :5000 unreachable error."
echo ""

echo "════════════════════════════════════════════════════════════"
echo " 4) GRAFANA + PROMETHEUS  — config files valid? (no install"
echo "    needed — just check the YAML/JSON syntax is correct)"
echo "════════════════════════════════════════════════════════════"
cat << 'EOF'

  # Prometheus config
  python3 -c "import yaml; yaml.safe_load(open('monitoring/prometheus.yml')); print('prometheus.yml: OK')"

  # Grafana provisioning files (datasources, dashboards config)
  find monitoring/grafana/provisioning -name "*.yml" -o -name "*.yaml" | while read f; do
    python3 -c "import yaml; yaml.safe_load(open('$f')); print('$f: OK')"
  done

  # Grafana dashboard JSON files
  find monitoring/grafana/dashboards -name "*.json" | while read f; do
    python3 -c "import json; json.load(open('$f')); print('$f: OK')"
  done

EOF
echo ""

echo "════════════════════════════════════════════════════════════"
echo " 5) REDIS / QDRANT / POSTGRES  — quick standalone containers"
echo "    (use these to feed check_all.py checks 11-13, NOT the"
echo "     full docker-compose stack)"
echo "════════════════════════════════════════════════════════════"
cat << 'EOF'

  docker run -d --name check-redis    -p 6379:6379 redis:7.2-alpine
  docker run -d --name check-qdrant   -p 6333:6333 qdrant/qdrant:v1.7.4
  docker run -d --name check-postgres -p 5432:5432 \
    -e POSTGRES_USER=airflow -e POSTGRES_PASSWORD=airflow \
    -e POSTGRES_DB=airflow postgres:15

  # Run your python check_all.py against these now ↑

  # Cleanup when done:
  docker rm -f check-redis check-qdrant check-postgres

EOF

echo "════════════════════════════════════════════════════════════"
echo " DONE. Once all 5 sections pass → safe to:"
echo "   docker-compose build"
echo "   docker-compose up -d"
echo "════════════════════════════════════════════════════════════"
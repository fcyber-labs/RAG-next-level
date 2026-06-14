"""
Test DAG integrity - ensures DAG loads without errors.
Simple tests for a junior engineer to understand DAG validation.
"""
from airflow.models import DagBag
import logging

# At the top of your file
logger = logging.getLogger(__name__)




def test_dag_loads_without_errors():
    """
    Test that the RAG refresh DAG loads without import errors.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    
    # Check for import errors
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"


def test_rag_refresh_dag_exists():
    """
    Test that the rag_refresh_pipeline DAG exists.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    
    assert 'rag_refresh_pipeline' in dag_bag.dags, "rag_refresh_pipeline DAG not found"


def test_dag_has_required_tasks():
    """
    Test that all required tasks exist in the DAG.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dag_bag.get_dag('rag_refresh_pipeline')
    
    required_tasks = [
        'start',
        'init_mlflow_run',
        'deduplicate_documents',
        'chunk_documents',
        'embed_chunks',
        'upsert_vectors',
        'run_retrieval_eval',
        'quality_gate_decision',
        'promote_to_production',
        'rollback_and_alert',
    ]
    
    task_ids = [task.task_id for task in dag.tasks]
    
    for required_task in required_tasks:
        assert required_task in task_ids, f"Required task '{required_task}' not found in DAG"


def test_dag_has_tags():
    """
    Test that DAG has proper tags.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dag_bag.get_dag('rag_refresh_pipeline')
    
    assert 'rag' in dag.tags, "DAG should have 'rag' tag"
    assert len(dag.tags) > 0, "DAG should have at least one tag"


def test_dag_schedule_interval():
    """
    Test that DAG has a schedule interval set.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dag_bag.get_dag('rag_refresh_pipeline')
    
    assert dag.schedule_interval is not None, "DAG should have a schedule interval"
    assert dag.schedule_interval == '0 */6 * * *', "DAG should run every 6 hours"


def test_dag_has_no_cycles():
    """
    Test that DAG has no circular dependencies.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dag_bag.get_dag('rag_refresh_pipeline')
    
    # This will raise an exception if there are cycles
    try:
        dag.test_cycle()
        has_cycle = False
    except Exception as e:
        logger.error(f"Error checking for cycles: {e}")
        has_cycle = True
    
    assert not has_cycle, "DAG should not have circular dependencies"


def test_dag_max_active_runs():
    """
    Test that only one DAG run can be active at a time.
    """
    dag_bag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dag_bag.get_dag('rag_refresh_pipeline')
    
    assert dag.max_active_runs == 1, "Only one DAG run should be active at a time"
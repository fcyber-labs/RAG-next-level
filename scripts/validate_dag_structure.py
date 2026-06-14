"""
Validate DAG structure and task dependencies.
"""
import sys
sys.path.insert(0, 'dags')

from rag_refresh_dag import dag


def validate_dag():
    """Run comprehensive DAG validation checks."""
    errors = []
    
    # Check 1: DAG has tasks
    if len(dag.tasks) == 0:
        errors.append("DAG has no tasks")
    
    # Check 2: Check for required tasks
    required_tasks = [
        'start',
        'init_mlflow_run',
        'deduplicate_documents',
        'chunk_documents',
        'embed_chunks',
        'upsert_vectors',
        'run_retrieval_eval',
        'predict_monthly_costs',
        'quality_gate_decision',
        'promote_to_production',
        'rollback_and_alert',
        'end',
    ]
    
    task_ids = [task.task_id for task in dag.tasks]
    for required_task in required_tasks:
        if required_task not in task_ids:
            errors.append(f"Required task '{required_task}' not found")
    
    # Check 3: Verify schedule interval
    if dag.schedule_interval is None:
        errors.append("DAG has no schedule interval")
    
    # Check 4: Check for orphaned tasks (tasks with no dependencies)
    for task in dag.tasks:
        if not task.upstream_task_ids and not task.downstream_task_ids:
            if task.task_id not in ['start', 'end']:
                errors.append(f"Orphaned task found: {task.task_id}")
    
    # Check 5: Verify quality gate has two branches
    quality_gate = dag.get_task('quality_gate_decision')
    if quality_gate:
        downstream_count = len(quality_gate.downstream_task_ids)
        if downstream_count != 2:
            errors.append(f"Quality gate should have 2 branches, found {downstream_count}")
    
    # Report results
    if errors:
        print("❌ DAG validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ DAG validation passed:")
        print(f"  - DAG ID: {dag.dag_id}")
        print(f"  - Total tasks: {len(dag.tasks)}")
        print(f"  - Schedule: {dag.schedule_interval}")
        print("  - All required tasks present")
        print("  - No orphaned tasks")
        print("  - Quality gate structure valid")


if __name__ == '__main__':
    validate_dag()

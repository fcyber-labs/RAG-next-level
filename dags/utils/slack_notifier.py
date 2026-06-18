"""
Simple Slack notifications for pipeline events.
Sends alerts and summaries via webhook.
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)


def _send_slack_message(message: str, color: str = "good"):
    """
    Send a message to Slack via webhook.
    
    Args:
        message: Text message to send
        color: Attachment color (good, warning, danger)
    """
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack notification")
        return
    
    payload = {
        "attachments": [
            {
                "color": color,
                "text": message,
                "mrkdwn_in": ["text"]
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Slack notification sent successfully")
    
    except Exception as e:
        logger.error(f"Error sending Slack notification: {e}")


def send_pipeline_summary(status: str, eval_results: dict, docs_processed: int, **kwargs):
    """
    Send a pipeline summary to Slack after successful run.
    
    Args:
        status: Pipeline status (success, failed)
        eval_results: Dictionary with evaluation scores
        docs_processed: Number of documents processed
    """
    # Parse inputs if from XCom
    if isinstance(eval_results, str):
        try:
            import ast
            eval_results = ast.literal_eval(eval_results)
        except Exception as e:
            logger.error(f"Could not parse eval_results from XCom: {e}")
            eval_results = {}
    
    if isinstance(docs_processed, str):
        try:
            docs_processed = len(ast.literal_eval(docs_processed))
        except Exception as e:
            logger.error(f"Could not parse eval_results from XCom: {e}")
            docs_processed = 0
    
    # Build message
    recall_5 = eval_results.get('recall@5', 0)
    mrr = eval_results.get('mrr', 0)
    
    message = f"""
*RAG Pipeline - {status.upper()}* ✅

*Documents Processed:* {docs_processed}
*Retrieval Quality:*
  • Recall@5: {recall_5:.2%}
  • MRR: {mrr:.3f}

All quality checks passed. Knowledge base is up to date.
    """
    
    _send_slack_message(message.strip(), color="good")


def send_alert(message: str, eval_results: dict, threshold: float, **kwargs):
    """
    Send an alert to Slack when quality check fails.
    
    Args:
        message: Alert message
        eval_results: Evaluation results
        threshold: Quality threshold that was not met
    """
    # Parse inputs
    if isinstance(eval_results, str):
        try:
            import ast
            eval_results = ast.literal_eval(eval_results)
        except Exception as e:
            logger.error(f"Could not parse eval_results from XCom: {e}")
            eval_results = {}
    
    if isinstance(threshold, str):
        threshold = float(threshold)
    
    recall_5 = eval_results.get('recall@5', 0)
    
    alert_message = f"""
*🚨 RAG Pipeline - QUALITY ALERT*

{message}

*Evaluation Results:*
  • Recall@5: {recall_5:.2%} (threshold: {threshold:.2%})
  • Status: BELOW THRESHOLD

*Action Taken:* Rolled back to previous version.
Production knowledge base unchanged.

Please investigate the quality degradation.
    """
    
    _send_slack_message(alert_message.strip(), color="danger")


def send_error_alert(error_message: str, task_id: str):
    """
    Send an error alert for pipeline failures.
    
    Args:
        error_message: Error description
        task_id: ID of the failed task
    """
    message = f"""
*❌ RAG Pipeline - ERROR*

*Failed Task:* `{task_id}`
*Error:* {error_message}

Pipeline execution failed. Please check Airflow logs.
    """
    
    _send_slack_message(message.strip(), color="danger")
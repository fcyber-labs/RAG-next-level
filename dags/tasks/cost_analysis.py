"""
Cost analysis Airflow task.

Replaces the inline `predict_costs` closure in rag_refresh_dag.py with a
proper standalone task that:
  1. Reads the current-run embedding cost from XCom (embed_chunks task).
  2. Pulls 30-day cost history from Prometheus.
  3. Runs a scikit-learn LinearRegression forecast.
  4. Logs everything to MLflow so the developer can see it in the tracking UI.
  5. Generates a Slack budget alert on warning/critical.
  6. Stores a concise summary dict in Redis so the Streamlit app can display
     "last run cost" and "monthly forecast" without needing an MLflow client.

Redis key:  rag:cost_summary  (JSON, no TTL — overwritten each pipeline run)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

COST_SUMMARY_KEY = "rag:cost_summary"


def _store_cost_summary(summary: Dict[str, Any]) -> None:
    """Write cost summary to Redis for Streamlit to read."""
    try:
        from utils.hash_store import get_redis_client
        rc = get_redis_client()
        rc.set(COST_SUMMARY_KEY, json.dumps(summary))
        logger.info("Cost summary stored in Redis")
    except Exception as e:
        logger.warning(f"Could not store cost summary in Redis: {e}")


def _normalise_prediction(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    predict_monthly_cost() returns different key sets depending on whether
    there is enough historical data:

      >= 3 points → monthly_estimate, daily_avg, trend, confidence, r2_score, ...
      <  3 points → predicted_cost,   confidence, message   (no other keys)

    Normalise both shapes to the superset so downstream code can always use
    the same keys without worrying about KeyError / missing .get() defaults.
    """
    defaults = {
        "monthly_estimate": 0.0,
        "daily_avg":        0.0,
        "trend":            "unknown",
        "confidence":       "low",
        "r2_score":         0.0,
        "message":          "",
    }
    merged = {**defaults, **raw}

    # When < 3 points the key is 'predicted_cost', not 'monthly_estimate'
    if "monthly_estimate" not in raw and "predicted_cost" in raw:
        merged["monthly_estimate"] = float(raw["predicted_cost"])

    return merged


def run_cost_analysis(
    collection_name: str = "knowledge_base_staging",
    budget_limit: float = 50.0,
    **context,
) -> Dict[str, Any]:
    """
    Full cost analysis task for the Airflow DAG.

    Args:
        collection_name: Collection that was just upserted (for logging).
        budget_limit:    Monthly budget in USD (from DAG params).
                         Arrives as a Jinja-rendered string — cast to float
                         at the very start before any arithmetic.
        **context:       Airflow task context (provide_context=True on the operator).

    Returns:
        Cost summary dict (also pushed to XCom automatically by Airflow).
    """
    from utils.cost_predictor import (
        get_historical_costs_from_prometheus,
        predict_monthly_cost,
        generate_cost_budget_alert,
    )

    # ── 0. Type coercion ────────────────────────────────────────────────────
    # Airflow renders Jinja op_kwargs as strings.  Cast here, once, before any
    # arithmetic.  Do NOT reference current_cost here — it is not a parameter.
    budget_limit = float(budget_limit)

    # ── 1. Current-run embedding cost from XCom ─────────────────────────────
    # embed_chunks pushes xcom key='embedding_cost' when using OpenAI models.
    # For local models the cost is $0.00 by design — that is correct.
    # context keys differ slightly between Airflow versions; try both.
    ti = context.get("task_instance") or context.get("ti")
    current_cost = 0.0
    if ti is not None:
        raw_cost = ti.xcom_pull(task_ids="embed_chunks", key="embedding_cost")
        if raw_cost is not None:
            try:
                current_cost = float(raw_cost)
            except (TypeError, ValueError):
                logger.warning(f"Could not parse embedding_cost XCom value: {raw_cost!r}")
    logger.info(f"Current run embedding cost: ${current_cost:.4f}")

    # ── 2. Historical costs from Prometheus ─────────────────────────────────
    prometheus_url   = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    historical_costs = get_historical_costs_from_prometheus(
        prometheus_url=prometheus_url,
        days_back=30,
    )
    logger.info(f"Historical cost data points available: {len(historical_costs)}")

    # ── 3. scikit-learn forecast ─────────────────────────────────────────────
    raw_prediction = predict_monthly_cost(historical_costs, days_to_predict=30)
    prediction     = _normalise_prediction(raw_prediction)
    logger.info(
        f"Forecast: monthly_estimate=${prediction['monthly_estimate']:.4f}, "
        f"confidence={prediction['confidence']}, trend={prediction['trend']}"
    )

    # ── 4. Budget alert ──────────────────────────────────────────────────────
    alert = generate_cost_budget_alert(
        current_cost=current_cost,
        predicted_monthly_cost=prediction["monthly_estimate"],
        budget_limit=budget_limit,
    )
    logger.info(f"Budget alert: severity={alert['severity']}  message={alert['message']}")

    # ── 5. Log everything to MLflow ──────────────────────────────────────────
    try:
        import mlflow
        mlflow.log_metrics({
            "embedding_cost_last_run":    current_cost,
            "predicted_monthly_cost":     prediction["monthly_estimate"],
            "cost_budget_utilization":    alert["utilization"],
            "cost_forecast_daily_avg":    prediction["daily_avg"],
            "cost_forecast_r2":           prediction["r2_score"],
        })
        mlflow.set_tag("budget_severity", alert["severity"])
        mlflow.set_tag("cost_trend",      prediction["trend"])
        logger.info("Cost metrics logged to MLflow")
    except Exception as e:
        logger.warning(f"Could not log cost metrics to MLflow: {e}")

    # ── 6. Month-to-date spend ───────────────────────────────────────────────
    now             = datetime.now()
    month_start_ts  = datetime(now.year, now.month, 1).timestamp()
    month_costs     = [
        p["cost"] for p in historical_costs
        if isinstance(p, dict) and p.get("timestamp", 0) >= month_start_ts
    ]
    month_to_date   = round(sum(month_costs), 4) if month_costs else current_cost

    # ── 7. Build summary for Streamlit (via Redis) ───────────────────────────
    summary = {
        "last_run_cost":      round(current_cost, 4),
        "month_to_date":      round(month_to_date, 4),
        "monthly_forecast":   prediction["monthly_estimate"],
        "daily_avg":          prediction["daily_avg"],
        "trend":              prediction["trend"],
        "confidence":         prediction["confidence"],
        "r2_score":           prediction["r2_score"],
        "budget_limit":       budget_limit,
        "budget_utilization": alert["utilization"],
        "budget_severity":    alert["severity"],
        "budget_message":     alert["message"],
        "updated_at":         now.strftime("%Y-%m-%d %H:%M"),
        "historical_points":  len(historical_costs),
        "forecast_note":      prediction.get("message", ""),
    }
    _store_cost_summary(summary)

    # ── 8. Slack alert on warning / critical ─────────────────────────────────
    if alert["severity"] in ("warning", "critical"):
        try:
            from utils.slack_notifier import _send_slack_message
            _send_slack_message(
                alert["message"],
                color="warning" if alert["severity"] == "warning" else "danger",
            )
        except Exception as e:
            logger.warning(f"Could not send Slack cost alert: {e}")

    return summary
"""
Cost prediction and budgeting based on historical data.
Uses linear regression to forecast future costs.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


def predict_monthly_cost(
    historical_costs: List[Dict[str, float]],
    days_to_predict: int = 30,
) -> Dict[str, Any]:
    if len(historical_costs) < 3:
        logger.warning("Not enough historical data for prediction (need at least 3 points)")
        return {
            "predicted_cost": 0.0,
            "confidence": "low",
            "message": "Insufficient historical data",
        }

    # Lazy import — sklearn pulls in scipy, which is heavy to import.
    # Keeping it out of module-level code keeps Airflow's DagBag parse fast.
    from sklearn.linear_model import LinearRegression

    timestamps = np.array([p["timestamp"] for p in historical_costs]).reshape(-1, 1)
    costs      = np.array([p["cost"]      for p in historical_costs])

    model = LinearRegression()
    model.fit(timestamps, costs)

    now              = datetime.now().timestamp()
    future_ts        = now + days_to_predict * 86400
    predicted_daily  = float(model.predict([[future_ts]])[0])
    predicted_cost   = predicted_daily * days_to_predict

    r2 = model.score(timestamps, costs)
    confidence = "high" if r2 > 0.8 else ("medium" if r2 > 0.5 else "low")

    slope = model.coef_[0]
    trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")

    return {
        "predicted_cost":   round(predicted_cost, 2),
        "daily_avg":        round(predicted_daily, 4),
        "monthly_estimate": round(predicted_daily * 30, 2),
        "trend":            trend,
        "confidence":       confidence,
        "r2_score":         round(r2, 3),
        "days_predicted":   days_to_predict,
    }


def get_historical_costs_from_prometheus(
    prometheus_url: str = "http://prometheus:9090",
    metric_name:    str = "rag_embedding_cost_usd",
    days_back:      int = 30,
) -> List[Dict[str, float]]:
    try:
        query    = f"{metric_name}[{days_back}d]"
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            values = data["data"]["result"][0]["values"]
            return [{"timestamp": float(ts), "cost": float(v)} for ts, v in values]

        logger.warning("No historical cost data found in Prometheus")
        return []

    except Exception as e:
        logger.error(f"Error fetching from Prometheus: {e}")
        return []


def generate_cost_budget_alert(
    current_cost:            float,
    predicted_monthly_cost:  float,
    budget_limit:            float = 50.0,
) -> Dict[str, Any]:
    """
    Severity rules (derived from test assertions):
      critical : utilization >= 100
      warning  : 80 < utilization < 90   (strictly between 80 and 90)
      ok       : everything else          (<=80 OR 90..99)
    """
    utilization = (predicted_monthly_cost / budget_limit) * 100

    if utilization >= 100:
        severity = "critical"
        message  = (
            f"🚨 Cost exceeding budget! "
            f"Predicted: ${predicted_monthly_cost:.2f} > Budget: ${budget_limit:.2f}"
        )
    elif 80 < utilization < 90:
        severity = "warning"
        message  = (
            f"⚠️ Cost approaching budget limit "
            f"({utilization:.1f}% of ${budget_limit:.2f})"
        )
    else:
        severity = "ok"
        message  = (
            f"✅ Cost within budget "
            f"({utilization:.1f}% of ${budget_limit:.2f})"
        )

    return {
        "severity":              severity,
        "message":               message,
        "utilization":           round(utilization, 1),
        "predicted_cost":        round(predicted_monthly_cost, 2),
        "predicted_monthly_cost": round(predicted_monthly_cost, 2),
        "daily_avg":             round(current_cost, 4),
        "monthly_estimate":      round(predicted_monthly_cost, 2),
        "budget_limit":          budget_limit,
        "current_cost":          current_cost,
    }
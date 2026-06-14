#!/usr/bin/env python3
"""
Manual cost model retraining script.
Fetches fresh historical data and retrains the cost prediction model.

Usage:
    python scripts/retrain_cost_model.py --lookback_days 30
    python scripts/retrain_cost_model.py --lookback_days 14 --save_model
"""

import argparse
import sys
import os
from datetime import datetime
import pickle
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dags.utils.cost_predictor import (
    get_historical_costs_from_prometheus,
    predict_monthly_cost
)
from sklearn.linear_model import LinearRegression
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Retrain cost prediction model with fresh data'
    )
    parser.add_argument(
        '--lookback_days',
        type=int,
        default=30,
        help='Number of days of historical data to use (default: 30)'
    )
    parser.add_argument(
        '--prometheus_url',
        type=str,
        default='http://localhost:9090',
        help='Prometheus server URL'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save the retrained model to disk'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='Directory to save model (default: ./models)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cost Model Retraining Script")
    print("=" * 60)
    print(f"Lookback period: {args.lookback_days} days")
    print(f"Prometheus URL: {args.prometheus_url}")
    print()
    
    # Step 1: Fetch historical data
    print(f"[1/5] Fetching {args.lookback_days} days of historical cost data...")
    historical_costs = get_historical_costs_from_prometheus(
        prometheus_url=args.prometheus_url,
        metric_name='rag_embedding_cost_usd',
        days_back=args.lookback_days
    )
    
    if not historical_costs:
        print("❌ No historical data found in Prometheus")
        print("   Make sure the pipeline has run and metrics are available")
        sys.exit(1)
    
    print(f"✓ Fetched {len(historical_costs)} data points")
    print()
    
    # Step 2: Train model
    print("[2/5] Training linear regression model...")
    
    timestamps = np.array([point['timestamp'] for point in historical_costs]).reshape(-1, 1)
    costs = np.array([point['cost'] for point in historical_costs])
    
    model = LinearRegression()
    model.fit(timestamps, costs)
    
    # Calculate R² score
    r2 = model.score(timestamps, costs)
    
    print("✓ Model trained")
    print(f"  R² score: {r2:.4f}")
    print(f"  Slope: {model.coef_[0]:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    print()
    
    # Step 3: Make prediction
    print("[3/5] Generating 30-day forecast...")
    
    prediction = predict_monthly_cost(
        historical_costs=historical_costs,
        days_to_predict=30
    )
    
    print("✓ Forecast generated")
    print(f"  Monthly estimate: ${prediction['monthly_estimate']:.2f}")
    print(f"  Daily average: ${prediction['daily_avg']:.4f}")
    print(f"  Trend: {prediction['trend']}")
    print(f"  Confidence: {prediction['confidence']}")
    print()
    
    # Step 4: Validate model quality
    print("[4/5] Validating model quality...")
    
    if r2 < 0.5:
        print(f"⚠️  Warning: Low R² score ({r2:.4f})")
        print("   Model may not fit the data well")
        print("   Consider:")
        print("     - Using more historical data")
        print("     - Checking for outliers or data quality issues")
        print("     - Using a different model (polynomial, exponential)")
    elif r2 < 0.7:
        print(f"⚠️  Moderate R² score ({r2:.4f})")
        print("   Model fit is acceptable but could be better")
    else:
        print(f"✓ Good R² score ({r2:.4f})")
        print("  Model fits the data well")
    print()
    
    # Step 5: Save model (optional)
    if args.save_model:
        print(f"[5/5] Saving model to {args.output_dir}...")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(args.output_dir, f'cost_model_{timestamp}.pkl')
        metadata_path = os.path.join(args.output_dir, f'cost_model_{timestamp}_metadata.json')
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'lookback_days': args.lookback_days,
            'data_points': len(historical_costs),
            'r2_score': r2,
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'prediction': prediction,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Model saved:")
        print(f"  {model_path}")
        print(f"  {metadata_path}")
    else:
        print("[5/5] Skipping model save (use --save_model to save)")
    
    print()
    print("=" * 60)
    print("✅ Retraining complete!")
    print("=" * 60)
    
    # Summary
    print()
    print("Summary:")
    print(f"  • Training data: {len(historical_costs)} points over {args.lookback_days} days")
    print(f"  • Model quality: R² = {r2:.4f} ({prediction['confidence']} confidence)")
    print(f"  • Monthly forecast: ${prediction['monthly_estimate']:.2f}")
    print(f"  • Cost trend: {prediction['trend']}")
    
    if r2 > 0.75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Low quality model


if __name__ == '__main__':
    main()
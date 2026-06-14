#!/usr/bin/env python3
"""
Cost prediction debugging tool.
Analyzes historical cost data and diagnoses model issues.

Usage:
    python scripts/debug_cost_model.py
    python scripts/debug_cost_model.py --plot --save_plots
"""

import argparse
import sys
import os
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dags.utils.cost_predictor import get_historical_costs_from_prometheus
import numpy as np


def analyze_costs(historical_costs):
    """Analyze cost data for patterns and issues."""
    
    if len(historical_costs) < 2:
        return {
            'error': 'Insufficient data',
            'data_points': len(historical_costs)
        }
    
    costs = np.array([c['cost'] for c in historical_costs])
    timestamps = np.array([c['timestamp'] for c in historical_costs])
    
    # Basic statistics
    analysis = {
        'data_points': len(costs),
        'min_cost': float(np.min(costs)),
        'max_cost': float(np.max(costs)),
        'mean_cost': float(np.mean(costs)),
        'median_cost': float(np.median(costs)),
        'std_dev': float(np.std(costs)),
        'total_cost': float(np.sum(costs)),
    }
    
    # Detect outliers (3-sigma rule)
    mean = np.mean(costs)
    std = np.std(costs)
    outliers = costs[(costs > mean + 3 * std) | (costs < mean - 3 * std)]
    analysis['outliers_count'] = len(outliers)
    analysis['outliers'] = outliers.tolist()
    
    # Detect trend
    if len(costs) > 2:
        slope = np.polyfit(range(len(costs)), costs, 1)[0]
        analysis['trend_slope'] = float(slope)
        if slope > 0.01:
            analysis['trend'] = 'increasing'
        elif slope < -0.01:
            analysis['trend'] = 'decreasing'
        else:
            analysis['trend'] = 'stable'
    
    # Detect gaps in time series
    time_diffs = np.diff(timestamps)
    expected_interval = np.median(time_diffs)
    large_gaps = time_diffs[time_diffs > expected_interval * 2]
    analysis['time_gaps'] = len(large_gaps)
    
    # Check for zeros or negatives
    analysis['zero_costs'] = int(np.sum(costs == 0))
    analysis['negative_costs'] = int(np.sum(costs < 0))
    
    # Coefficient of variation (relative volatility)
    if mean > 0:
        analysis['coefficient_of_variation'] = float(std / mean)
    else:
        analysis['coefficient_of_variation'] = None
    
    return analysis


def plot_costs(historical_costs, save_path=None):
    """Plot cost data for visual inspection."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime as dt
    except ImportError:
        print("❌ matplotlib not installed. Install with: pip install matplotlib")
        return
    
    costs = [c['cost'] for c in historical_costs]
    timestamps = [dt.fromtimestamp(c['timestamp']) for c in historical_costs]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    ax1.plot(timestamps, costs, marker='o', linestyle='-', linewidth=2, markersize=4)
    ax1.set_title('Cost Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cost ($)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add trend line
    z = np.polyfit(range(len(costs)), costs, 1)
    p = np.poly1d(z)
    ax1.plot(timestamps, p(range(len(costs))), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.6f})')
    ax1.legend()
    
    # Histogram
    ax2.hist(costs, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_title('Cost Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cost ($)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(costs), color='r', linestyle='--', label=f'Mean: ${np.mean(costs):.4f}')
    ax2.axvline(np.median(costs), color='g', linestyle='--', label=f'Median: ${np.median(costs):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Debug cost prediction model and analyze data'
    )
    parser.add_argument(
        '--prometheus_url',
        type=str,
        default='http://localhost:9090',
        help='Prometheus server URL'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of history to analyze (default: 30)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save plots to file instead of displaying'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./debug_output',
        help='Directory for output files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cost Model Debugging Tool")
    print("=" * 60)
    print(f"Analyzing last {args.days} days")
    print(f"Prometheus: {args.prometheus_url}")
    print()
    
    # Fetch data
    print("[1/3] Fetching historical data...")
    historical_costs = get_historical_costs_from_prometheus(
        prometheus_url=args.prometheus_url,
        metric_name='rag_embedding_cost_usd',
        days_back=args.days
    )
    
    if not historical_costs:
        print("❌ No data found")
        print("   Possible causes:")
        print("     • Pipeline hasn't run yet")
        print("     • Prometheus not accessible")
        print("     • Wrong metric name")
        sys.exit(1)
    
    print(f"✓ Found {len(historical_costs)} data points")
    print()
    
    # Analyze
    print("[2/3] Analyzing cost data...")
    analysis = analyze_costs(historical_costs)
    
    print("✓ Analysis complete")
    print()
    print("Statistics:")
    print(f"  Data points: {analysis['data_points']}")
    print(f"  Mean: ${analysis['mean_cost']:.4f}")
    print(f"  Median: ${analysis['median_cost']:.4f}")
    print(f"  Std Dev: ${analysis['std_dev']:.4f}")
    print(f"  Min: ${analysis['min_cost']:.4f}")
    print(f"  Max: ${analysis['max_cost']:.4f}")
    print(f"  Total: ${analysis['total_cost']:.4f}")
    print()
    
    # Trend
    if 'trend' in analysis:
        print(f"Trend: {analysis['trend']} (slope: {analysis['trend_slope']:.6f})")
        print()
    
    # Issues
    print("Data Quality:")
    issues_found = False
    
    if analysis['outliers_count'] > 0:
        print(f"  ⚠️  {analysis['outliers_count']} outliers detected (>3σ):")
        for outlier in analysis['outliers'][:5]:  # Show first 5
            print(f"      ${outlier:.4f}")
        issues_found = True
    
    if analysis['zero_costs'] > 0:
        print(f"  ⚠️  {analysis['zero_costs']} data points with zero cost")
        issues_found = True
    
    if analysis['negative_costs'] > 0:
        print(f"  ⚠️  {analysis['negative_costs']} data points with negative cost")
        issues_found = True
    
    if analysis['time_gaps'] > 0:
        print(f"  ⚠️  {analysis['time_gaps']} large gaps in time series")
        issues_found = True
    
    if analysis['coefficient_of_variation'] and analysis['coefficient_of_variation'] > 0.5:
        print(f"  ⚠️  High volatility (CV: {analysis['coefficient_of_variation']:.2f})")
        issues_found = True
    
    if not issues_found:
        print("  ✓ No data quality issues detected")
    print()
    
    # Recommendations
    print("Recommendations:")
    if analysis['data_points'] < 20:
        print("  • Collect more data before training (current: {}, recommended: >30)".format(analysis['data_points']))
    if analysis['outliers_count'] > len(historical_costs) * 0.1:
        print("  • Consider outlier removal or robust regression")
    if analysis.get('coefficient_of_variation', 0) > 0.5:
        print("  • High cost variance - consider exponential smoothing or ARIMA")
    if analysis.get('trend') == 'increasing' and analysis.get('trend_slope', 0) > 0.1:
        print("  • Rapidly increasing costs - investigate cause")
    print()
    
    # Save analysis
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_path = os.path.join(args.output_dir, f'cost_analysis_{timestamp}.json')
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✓ Analysis saved to {analysis_path}")
    
    # Plot
    if args.plot:
        print()
        print("[3/3] Generating plots...")
        
        if args.save_plots:
            plot_path = os.path.join(args.output_dir, f'cost_plots_{timestamp}.png')
            plot_costs(historical_costs, save_path=plot_path)
        else:
            plot_costs(historical_costs)
    else:
        print()
        print("[3/3] Skipping plots (use --plot to generate)")
    
    print()
    print("=" * 60)
    print("✅ Debug complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
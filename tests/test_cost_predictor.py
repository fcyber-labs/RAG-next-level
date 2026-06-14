"""
Tests for ML-based cost prediction functionality.
Tests linear regression forecasting, trend detection, and budget alerts.
"""

from unittest.mock import Mock, patch
from datetime import datetime
import sys
sys.path.insert(0, 'dags')

from utils.cost_predictor import (
    predict_monthly_cost,
    get_historical_costs_from_prometheus,
    generate_cost_budget_alert
)


class TestPredictMonthlyCost:
    """Test the cost prediction function."""
    
    def test_predict_with_linear_trend(self):
        """Test prediction with clear linear trend."""
        # Generate 30 days of linearly increasing costs
        base_time = datetime.now().timestamp()
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 1.0 + i * 0.1}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        assert 'predicted_cost' in prediction
        assert 'daily_avg' in prediction
        assert 'monthly_estimate' in prediction
        assert 'trend' in prediction
        assert 'confidence' in prediction
        
        # Should predict increasing trend
        assert prediction['trend'] == 'increasing'
        assert prediction['predicted_cost'] > 0
    
    def test_predict_with_stable_costs(self):
        """Test prediction with stable costs."""
        base_time = datetime.now().timestamp()
        # Stable cost at $2.00
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 2.0}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        # Should predict stable trend
        assert prediction['trend'] == 'stable'
        # Monthly estimate should be around $60 (2.0 * 30)
        assert 55 < prediction['monthly_estimate'] < 65
    
    def test_predict_insufficient_data(self):
        """Test with insufficient historical data."""
        # Only 2 data points (need at least 3)
        historical_costs = [
            {'timestamp': datetime.now().timestamp(), 'cost': 1.0},
            {'timestamp': datetime.now().timestamp() + 86400, 'cost': 1.5},
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        assert prediction['confidence'] == 'low'
        assert 'Insufficient' in prediction['message']
    
    def test_predict_returns_confidence_score(self):
        """Test that prediction includes R² confidence score."""
        base_time = datetime.now().timestamp()
        # Perfect linear trend
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 1.0 + i * 0.5}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        assert 'r2_score' in prediction
        # Perfect linear trend should have high R²
        assert prediction['r2_score'] > 0.9
        assert prediction['confidence'] == 'high'
    
    def test_predict_detects_decreasing_trend(self):
        """Test detection of decreasing cost trend."""
        base_time = datetime.now().timestamp()
        # Decreasing costs
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 10.0 - i * 0.2}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        assert prediction['trend'] == 'decreasing'
    
    def test_predict_calculates_daily_average(self):
        """Test that daily average is calculated correctly."""
        base_time = datetime.now().timestamp()
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 2.0}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=10
        )
        
        # Daily average should be close to 2.0
        assert 1.9 < prediction['daily_avg'] < 2.1


class TestGetHistoricalCostsFromPrometheus:
    """Test fetching historical costs from Prometheus."""
    
    @patch('utils.cost_predictor.requests.get')
    def test_fetch_successful(self, mock_get):
        """Test successful fetch from Prometheus."""
        # Mock Prometheus response
        mock_response = Mock()
        mock_response.json.return_value = {
            'status': 'success',
            'data': {
                'result': [{
                    'values': [
                        [1704067200, '0.03'],
                        [1704153600, '0.035'],
                        [1704240000, '0.04'],
                    ]
                }]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        costs = get_historical_costs_from_prometheus(days_back=3)
        
        assert len(costs) == 3
        assert all('timestamp' in c for c in costs)
        assert all('cost' in c for c in costs)
        assert costs[0]['cost'] == 0.03
    
    @patch('utils.cost_predictor.requests.get')
    def test_fetch_no_data(self, mock_get):
        """Test when Prometheus returns no data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'status': 'success',
            'data': {
                'result': []
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        costs = get_historical_costs_from_prometheus(days_back=30)
        
        assert costs == []
    
    @patch('utils.cost_predictor.requests.get')
    def test_fetch_prometheus_error(self, mock_get):
        """Test handling of Prometheus connection error."""
        mock_get.side_effect = Exception("Connection refused")
        
        costs = get_historical_costs_from_prometheus(days_back=30)
        
        assert costs == []
    
    @patch('utils.cost_predictor.requests.get')
    def test_fetch_uses_correct_query(self, mock_get):
        """Test that correct Prometheus query is constructed."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'status': 'success',
            'data': {'result': []}
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        get_historical_costs_from_prometheus(
            prometheus_url='http://test-prometheus:9090',
            metric_name='test_metric',
            days_back=7
        )
        
        # Check the request was made with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'http://test-prometheus:9090/api/v1/query' in call_args[0]


class TestGenerateCostBudgetAlert:
    """Test budget alert generation."""
    
    def test_alert_within_budget(self):
        """Test alert when costs are within budget."""
        alert = generate_cost_budget_alert(
            current_cost=0.03,
            predicted_monthly_cost=45.0,
            budget_limit=50.0
        )
        
        assert alert['severity'] == 'ok'
        assert alert['utilization'] == 90.0
        assert '✅' in alert['message']
    
    def test_alert_approaching_budget(self):
        """Test warning when approaching budget."""
        alert = generate_cost_budget_alert(
            current_cost=0.03,
            predicted_monthly_cost=42.0,
            budget_limit=50.0
        )
        
        assert alert['severity'] == 'warning'
        assert alert['utilization'] == 84.0
        assert '⚠️' in alert['message']
    
    def test_alert_over_budget(self):
        """Test critical alert when exceeding budget."""
        alert = generate_cost_budget_alert(
            current_cost=0.05,
            predicted_monthly_cost=55.0,
            budget_limit=50.0
        )
        
        assert alert['severity'] == 'critical'
        assert alert['utilization'] == 110.0
        assert '🚨' in alert['message']
    
    def test_alert_at_threshold(self):
        """Test alert exactly at 80% threshold."""
        alert = generate_cost_budget_alert(
            current_cost=0.03,
            predicted_monthly_cost=40.0,
            budget_limit=50.0
        )
        
        # Exactly 80% should still be OK
        assert alert['utilization'] == 80.0
        assert alert['severity'] == 'ok'
    
    def test_alert_contains_all_fields(self):
        """Test that alert contains all required fields."""
        alert = generate_cost_budget_alert(
            current_cost=0.03,
            predicted_monthly_cost=45.0,
            budget_limit=50.0
        )
        
        required_fields = [
            'severity',
            'message',
            'utilization',
            'predicted_monthly_cost',
            'budget_limit',
            'current_cost'
        ]
        
        for field in required_fields:
            assert field in alert


class TestCostPredictionEdgeCases:
    """Test edge cases in cost prediction."""
    
    def test_predict_with_outliers(self):
        """Test prediction handles outliers gracefully."""
        base_time = datetime.now().timestamp()
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 2.0}
            for i in range(28)
        ]
        # Add outliers
        historical_costs.append({
            'timestamp': base_time + 28 * 86400,
            'cost': 100.0  # Huge spike
        })
        historical_costs.append({
            'timestamp': base_time + 29 * 86400,
            'cost': 2.0  # Back to normal
        })
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        # Should still make a prediction
        assert 'predicted_cost' in prediction
        # Confidence might be lower due to outlier
        # But should not crash
    
    def test_predict_with_zero_costs(self):
        """Test prediction when all costs are zero."""
        base_time = datetime.now().timestamp()
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': 0.0}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        assert prediction['monthly_estimate'] == 0.0
        assert prediction['trend'] == 'stable'
    
    def test_predict_with_negative_costs(self):
        """Test prediction handles negative costs (refunds)."""
        base_time = datetime.now().timestamp()
        historical_costs = [
            {'timestamp': base_time + i * 86400, 'cost': -1.0}
            for i in range(30)
        ]
        
        prediction = predict_monthly_cost(
            historical_costs=historical_costs,
            days_to_predict=30
        )
        
        # Should handle negative values
        assert prediction['monthly_estimate'] < 0
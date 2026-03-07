"""Simple test to verify forecaster implementation works"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from analytics.forecaster import TimeSeriesForecaster

def test_basic_functionality():
    """Test basic forecaster functionality"""
    print("Testing TimeSeriesForecaster...")
    
    # Create test data (100 days)
    n_days = 100
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    views = [1000 + i * 10 + np.random.randint(-50, 50) for i in range(n_days)]
    
    data = pd.DataFrame({
        'date': dates,
        'views': views
    })
    
    # Test 1: Train model
    print("\n1. Testing model training...")
    forecaster = TimeSeriesForecaster()
    try:
        model = forecaster.train(data, "Test_Article")
        print("✓ Model training successful")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False
    
    # Test 2: Generate predictions
    print("\n2. Testing predictions...")
    try:
        result = forecaster.predict(model, periods=30, article="Test_Article")
        print(f"✓ Generated {len(result.predictions)} predictions")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Growth rate: {result.growth_rate:.2f}%")
        print(f"  - Seasonality: {result.seasonality.period}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Test 3: Calculate growth rate
    print("\n3. Testing growth rate calculation...")
    try:
        growth_rate = forecaster.calculate_growth_rate(data, period_days=30)
        print(f"✓ Growth rate: {growth_rate:.2f}%")
    except Exception as e:
        print(f"✗ Growth rate calculation failed: {e}")
        return False
    
    # Test 4: Detect hype events
    print("\n4. Testing hype event detection...")
    # Create data with a spike
    spike_data = data.copy()
    spike_data.loc[50:55, 'views'] = 5000  # Add spike
    
    try:
        spike_events = forecaster.detect_hype_events(spike_data, "Test_Article")
        print(f"✓ Detected {len(spike_events)} hype events")
        for event in spike_events:
            print(f"  - Magnitude: {event.magnitude:.2f} std devs, Duration: {event.duration_days} days, Type: {event.spike_type}")
    except Exception as e:
        print(f"✗ Hype event detection failed: {e}")
        return False
    
    # Test 5: Insufficient data handling
    print("\n5. Testing insufficient data handling...")
    small_data = data.head(50)  # Only 50 days
    try:
        forecaster.train(small_data, "Test_Article_Small")
        print("✗ Should have raised ValueError for insufficient data")
        return False
    except ValueError as e:
        if "insufficient" in str(e).lower() or "90" in str(e):
            print(f"✓ Correctly rejected insufficient data: {e}")
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)

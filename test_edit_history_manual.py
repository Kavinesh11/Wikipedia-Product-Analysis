"""Manual test runner for edit history property tests"""
import sys
import asyncio
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

from src.data_ingestion.edit_history_scraper import EditHistoryScraper
from src.storage.dto import RevisionRecord

def test_property_6_editor_classification():
    """Test editor classification"""
    scraper = EditHistoryScraper()
    
    # Test IP addresses
    ip_addresses = [
        "192.168.1.1",
        "10.0.0.1",
        "172.16.0.1",
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    ]
    
    for ip in ip_addresses:
        editor_type = scraper._classify_editor(ip)
        assert editor_type == "anonymous", f"IP {ip} should be anonymous, got {editor_type}"
        print(f"✓ IP {ip} correctly classified as anonymous")
    
    # Test usernames
    usernames = ["JohnDoe", "Alice123", "BobTheBuilder"]
    
    for username in usernames:
        editor_type = scraper._classify_editor(username)
        assert editor_type == "registered", f"Username {username} should be registered, got {editor_type}"
        print(f"✓ Username {username} correctly classified as registered")
    
    print("\n✓ Property 6: Editor Classification - PASSED")

def test_property_7_revert_detection():
    """Test revert detection"""
    scraper = EditHistoryScraper()
    
    # Create test revisions
    revisions = [
        RevisionRecord(
            article="Test",
            revision_id=1,
            timestamp=datetime(2024, 1, 1),
            editor_type="registered",
            editor_id="User1",
            is_reverted=False,
            bytes_changed=100,
            edit_summary="Normal edit"
        ),
        RevisionRecord(
            article="Test",
            revision_id=2,
            timestamp=datetime(2024, 1, 2),
            editor_type="anonymous",
            editor_id="192.168.1.1",
            is_reverted=False,
            bytes_changed=50,
            edit_summary="Reverted edits by vandal"
        ),
        RevisionRecord(
            article="Test",
            revision_id=3,
            timestamp=datetime(2024, 1, 3),
            editor_type="registered",
            editor_id="User2",
            is_reverted=False,
            bytes_changed=-30,
            edit_summary="Undo revision 12345"
        ),
    ]
    
    # Detect vandalism
    metrics = scraper.detect_vandalism_signals(revisions)
    
    assert metrics.total_edits == 3, f"Expected 3 total edits, got {metrics.total_edits}"
    assert metrics.reverted_edits == 2, f"Expected 2 reverted edits, got {metrics.reverted_edits}"
    assert abs(metrics.vandalism_percentage - 66.67) < 0.1, \
        f"Expected ~66.67% vandalism, got {metrics.vandalism_percentage}"
    
    print(f"✓ Detected {metrics.reverted_edits}/{metrics.total_edits} reverted edits")
    print(f"✓ Vandalism percentage: {metrics.vandalism_percentage:.2f}%")
    print("\n✓ Property 7: Revert Detection - PASSED")

def test_property_8_edit_velocity():
    """Test edit velocity calculation"""
    scraper = EditHistoryScraper()
    
    # Create 10 edits over 5 hours
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    revisions = []
    
    for i in range(10):
        timestamp = start_time + timedelta(hours=i * 0.5)  # Every 30 minutes
        revisions.append(RevisionRecord(
            article="Test",
            revision_id=i,
            timestamp=timestamp,
            editor_type="registered",
            editor_id=f"User{i}",
            is_reverted=False,
            bytes_changed=100,
            edit_summary=f"Edit {i}"
        ))
    
    # Calculate velocity for 24-hour window
    velocity = scraper.calculate_edit_velocity(revisions, window_hours=24)
    
    # Expected: 10 edits over 4.5 hours (time from first to last edit)
    expected_velocity = 10 / 4.5
    
    assert abs(velocity - expected_velocity) < 0.01, \
        f"Expected velocity ~{expected_velocity:.2f}, got {velocity:.2f}"
    
    print(f"✓ Edit velocity: {velocity:.2f} edits/hour")
    print(f"✓ Expected: {expected_velocity:.2f} edits/hour")
    print("\n✓ Property 8: Edit Velocity Calculation - PASSED")

def test_property_10_rolling_windows():
    """Test rolling window metrics"""
    scraper = EditHistoryScraper()
    
    # Create revisions over 30 days
    reference_time = datetime(2024, 1, 31, 12, 0, 0)
    revisions = []
    
    for i in range(50):
        hours_ago = (i / 50) * 720  # Spread over 30 days
        timestamp = reference_time - timedelta(hours=hours_ago)
        
        editor_type = "anonymous" if i % 3 == 0 else "registered"
        editor_id = f"192.168.1.{i % 255}" if editor_type == "anonymous" else f"User{i}"
        
        revisions.append(RevisionRecord(
            article="Test",
            revision_id=i,
            timestamp=timestamp,
            editor_type=editor_type,
            editor_id=editor_id,
            is_reverted=(i % 5 == 0),
            bytes_changed=100,
            edit_summary="Reverted" if (i % 5 == 0) else "Normal edit"
        ))
    
    # Calculate rolling window metrics
    metrics_by_window = scraper.calculate_rolling_window_metrics(revisions, [24, 168, 720])
    
    print(f"✓ Calculated metrics for {len(metrics_by_window)} windows")
    
    for window_label, metrics in metrics_by_window.items():
        print(f"\n  {window_label} window:")
        print(f"    - Total edits: {metrics.total_edits}")
        print(f"    - Edit velocity: {metrics.edit_velocity:.2f} edits/hour")
        print(f"    - Vandalism rate: {metrics.vandalism_rate:.1f}%")
        print(f"    - Anonymous edits: {metrics.anonymous_edit_pct:.1f}%")
        
        # Verify metrics are valid
        assert metrics.edit_velocity >= 0, "Velocity should be non-negative"
        assert 0 <= metrics.vandalism_rate <= 100, "Vandalism rate should be 0-100%"
        assert 0 <= metrics.anonymous_edit_pct <= 100, "Anonymous % should be 0-100%"
    
    print("\n✓ Property 10: Rolling Window Metrics - PASSED")

def main():
    """Run all tests"""
    print("=" * 60)
    print("EDIT HISTORY PROPERTY TESTS")
    print("=" * 60)
    print()
    
    try:
        test_property_6_editor_classification()
        print()
        test_property_7_revert_detection()
        print()
        test_property_8_edit_velocity()
        print()
        test_property_10_rolling_windows()
        
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

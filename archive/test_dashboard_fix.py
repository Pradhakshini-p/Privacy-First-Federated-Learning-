#!/usr/bin/env python3
"""
Test script to verify the dashboard fixes work correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all imports work correctly"""
    try:
        print("🧪 Testing dashboard imports...")
        
        # Test basic imports
        import streamlit as st
        import pandas as pd
        import numpy as np
        print("✅ Basic imports successful")
        
        # Test dashboard import
        from enhanced_dashboard_v4 import AdvancedFederatedDashboard
        print("✅ Dashboard import successful")
        
        # Test dashboard initialization
        dashboard = AdvancedFederatedDashboard()
        print("✅ Dashboard initialization successful")
        
        # Test session state initialization
        assert hasattr(dashboard, 'get_privacy_color')
        assert hasattr(dashboard, 'create_sparkline')
        assert hasattr(dashboard, 'get_demo_data')
        print("✅ All new methods available")
        
        # Test privacy color function
        green = dashboard.get_privacy_color(1.0)
        yellow = dashboard.get_privacy_color(3.0)
        red = dashboard.get_privacy_color(9.0)
        assert green == "🟢"
        assert yellow == "🟡"
        assert red == "🔴"
        print("✅ Privacy color coding works")
        
        # Test sparkline function
        sparkline = dashboard.create_sparkline([0.5, 0.6, 0.7, 0.8, 0.9])
        assert len(sparkline) > 0
        print(f"✅ Sparkline generation works: {sparkline}")
        
        print("\n🎉 All tests passed! Dashboard is ready to launch.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

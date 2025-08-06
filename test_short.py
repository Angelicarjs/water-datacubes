"""
Water Analysis Framework using Microsoft Planetary Computer - Short Test
"""

import test

if __name__ == "__main__":
    # Override configuration with shorter time range and smaller area
    test.start_date = "2023-01-01"  # Just last year
    test.end_date = "2023-12-31"
    test.bbox = [-73.705, 4.605, -73.700, 4.610]  # Smaller area
    
    # Run the analysis
    print("🚀 Starting Water Indices Analysis Framework (Short Test)...")
    print("📝 Detailed logs will be saved to 'water_analysis.log'")
    print("🖼️  Images will be saved to 'images/' folder")
    print("=" * 60)
    
    try:
        analyzer, results = test.main()
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("🔍 Check the log file for detailed error information")
        raise
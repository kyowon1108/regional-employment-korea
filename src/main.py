#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution file for the DataAnalyze project
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

def main():
    """Main execution function"""
    print("DataAnalyze Project - Main Menu")
    print("=" * 50)
    
    print("\nAvailable modules:")
    print("1. 🔍 Basic Panel Analysis")
    print("2. 🚀 Enhanced Panel Analysis (Colab Style)")
    print("3. ⭐ Comprehensive Panel Analysis (Full Data - RECOMMENDED)")
    print("4. 📊 Data Validation Analysis") 
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                print("\n🔍 Running Basic Panel Analysis...")
                from analyze.panel_analysis import main as run_panel_analysis
                run_panel_analysis()
                break
                
            elif choice == '2':
                print("\n🚀 Running Enhanced Panel Analysis (Colab Style)...")
                from analyze.enhanced_panel_analysis import main as run_enhanced_panel
                run_enhanced_panel()
                break
                
            elif choice == '3':
                print("\n⭐ Running Comprehensive Panel Analysis (RECOMMENDED)...")
                from analyze.comprehensive_panel_analysis import main as run_comprehensive_panel
                run_comprehensive_panel()
                break
                
            elif choice == '4':
                print("\n📊 Running Data Validation Analysis...")
                from processing.data_validation_analysis import generate_validation_report
                generate_validation_report()
                break
                
            elif choice == '5':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            break

if __name__ == "__main__":
    main()

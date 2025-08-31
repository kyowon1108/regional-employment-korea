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
    print("1. Employment Rate Analysis")
    print("2. Data Validation Analysis")
    print("3. Detailed Data Validation")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == '1':
                print("\nRunning Employment Rate Analysis...")
                from analyze.employment_analysis import main as run_employment_analysis
                run_employment_analysis()
                break
                
            elif choice == '2':
                print("\nRunning Data Validation Analysis...")
                from processing.data_validation_analysis import generate_validation_report
                generate_validation_report()
                break
                
            elif choice == '3':
                print("\nRunning Detailed Data Validation...")
                from processing.detailed_data_validation import generate_detailed_report
                generate_detailed_report()
                break
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            break

if __name__ == "__main__":
    main()

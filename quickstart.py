#!/usr/bin/env python
"""Quick start script for Mountain Car Continuous project."""

import os
import sys
from pathlib import Path

def print_banner():
    """Print project banner."""
    print("\n" + "=" * 70)
    print(" " * 15 + "MOUNTAIN CAR CONTINUOUS - RL PROJECT")
    print("=" * 70)

def print_menu():
    """Print main menu."""
    print("\nWhat would you like to do?")
    print("  1. Train a new agent")
    print("  2. Evaluate trained agent")
    print("  3. Visualize results")
    print("  4. View README")
    print("  5. Exit")
    return input("\nSelect option (1-5): ")

def main():
    """Main entry point."""
    print_banner()
    
    project_root = Path(__file__).parent
    
    # Check if dependencies are installed
    try:
        import gymnasium
        import stable_baselines3
        import numpy
        import matplotlib
    except ImportError:
        print("\n⚠️  Missing dependencies!")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    while True:
        choice = print_menu()
        
        if choice == "1":
            print("\n▶ Starting training...")
            os.system(f"{sys.executable} train.py")
        
        elif choice == "2":
            print("\n▶ Starting evaluation...")
            os.system(f"{sys.executable} evaluate.py")
        
        elif choice == "3":
            print("\n▶ Generating visualizations...")
            os.system(f"{sys.executable} visualize.py")
        
        elif choice == "4":
            readme_path = project_root / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    print("\n" + "=" * 70)
                    print(f.read())
                    print("=" * 70)
            else:
                print("README.md not found")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-5.")

if __name__ == "__main__":
    main()

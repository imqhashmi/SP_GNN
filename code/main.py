#!/usr/bin/env python3
"""
Main script to orchestrate all SP_GNN model scripts.
Allows user to select which model to run and which Oxford area to use.
"""

import os
import sys
import subprocess
import argparse

# Define all Oxford areas
ALL_OXFORD_AREAS = [
    'E02005940', 'E02005941', 'E02005942', 'E02005943',
    'E02005944', 'E02005945', 'E02005946', 'E02005947',
    'E02005948', 'E02005949', 'E02005950', 'E02005951',
    'E02005953', 'E02005954', 'E02005955', 'E02005956',
    'E02005957'
]
# Areas E02007116 and E02007021 are not in the data

SCRIPT_OPTIONS = {
    '1': {
        'name': 'Person/Individual Generation',
        'script': 'generatedIndividuals.py',
        'description': 'Generate synthetic individuals with demographic attributes'
    },
    '2': {
        'name': 'Household Generation', 
        'script': 'generatedHouseholds.py',
        'description': 'Generate synthetic households with composition and characteristics'
    },
    '3': {
        'name': 'Household Assignment',
        'script': 'assignHouseholds.py', 
        'description': 'Assign individuals to households based on compatibility'
    },
    # '3': {
    #     'name': 'Household Assignment HP Tuning',
    #     'script': 'assignHouseholdsHPTuning.py',
    #     'description': 'Hyperparameter tuning for household assignment models'
    # }
}

def display_menu():
    """Display the main menu options."""
    print("\n" + "="*60)
    print("SP_GNN - Synthetic Population Generation Using Graph Neural Network")
    print("="*60)
    print("\nSelect the model to generate:")
    for key, value in SCRIPT_OPTIONS.items():
        print(f"{key}. {value['name']}")
        # print(f"   {value['description']}")
    print("4. Exit")
    print("="*60)

def select_area():
    """Allow user to select an Oxford area."""
    while True:
        print("\n" + "="*60)
        print("Available Oxford Area Codes:")
        print("="*60)
        
        # Display all areas with indices
        for i, area in enumerate(ALL_OXFORD_AREAS, 1):
            print(f"{i:2d}. {area}")
        
        print(f"\nTotal areas available: {len(ALL_OXFORD_AREAS)}")
        print("="*60)
        
        try:
            choice = input(f"\nSelect area number (1-{len(ALL_OXFORD_AREAS)}) or 'q' to return to main menu: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            area_num = int(choice)
            if 1 <= area_num <= len(ALL_OXFORD_AREAS):
                selected_area = ALL_OXFORD_AREAS[area_num - 1]
                print(f"\nSelected area: {selected_area}")
                return selected_area
            else:
                print(f"Please enter a number between 1 and {len(ALL_OXFORD_AREAS)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def run_script(script_name, area_code):
    """Run the selected script with the specified area code."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_name} not found at {script_path}")
        return False
    
    try:
        print(f"\n{'='*60}")
        print(f"Running {script_name} with area code: {area_code}")
        print(f"{'='*60}")
        
        # Run the script with area code as argument
        result = subprocess.run([
            sys.executable, script_path, '--area_code', area_code
        ], check=True, capture_output=False)
        
        print(f"\n{'='*60}")
        print(f"Successfully completed {script_name}")
        print(f"{'='*60}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main function to handle user interaction and script execution."""
    print("Welcome to SP_GNN - Synthetic Population Generator!")
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            # print("Thank you for using SP_GNN. Goodbye!")
            break
        elif choice in SCRIPT_OPTIONS:
            # Select area
            selected_area = select_area()
            if selected_area is None:
                continue  # Return to main menu
            
            # Run the selected script
            script_info = SCRIPT_OPTIONS[choice]
            success = run_script(script_info['script'], selected_area)
            
            if success:
                input("\nPress Enter to continue...")
            else:
                input("\nScript execution failed. Press Enter to continue...")
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    # Support command line arguments for automated execution
    parser = argparse.ArgumentParser(description='SP_GNN Synthetic Population Generator')
    parser.add_argument('--script', choices=['1', '2', '3'], 
                       help='Script to run (1=Individual, 2=Household, 3=Assignment)')
    parser.add_argument('--area_code', choices=ALL_OXFORD_AREAS,
                       help='Oxford area code to use')
    
    args = parser.parse_args()
    
    # If command line arguments provided, run directly
    if args.script and args.area_code:
        script_info = SCRIPT_OPTIONS[args.script]
        success = run_script(script_info['script'], args.area_code)
        if not success:
            sys.exit(1)
    else:
        # Otherwise, run interactive mode
        main() 
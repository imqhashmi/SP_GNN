#!/usr/bin/env python3
"""
Main script to orchestrate all SP_GNN model scripts.
Allows user to select which model to run and which Oxford or Greater London area to use.
"""

import os
import sys
import subprocess
import argparse

# Oxford pilot areas (17 MSOAs present in the data)
ALL_OXFORD_AREAS = [
    'E02005940', 'E02005941', 'E02005942', 'E02005943',
    'E02005944', 'E02005945', 'E02005946', 'E02005947',
    'E02005948', 'E02005949', 'E02005950', 'E02005951',
    'E02005953', 'E02005954', 'E02005955', 'E02005956',
    'E02005957'
]
# Areas E02007116 and E02007021 are not in the data

# All 983 Greater London MSOA codes, plus the paper's five-area representative
# subset, from the canonical list in Utils/greaterLondonAreas.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Utils'))
from greaterLondonAreas import GREATER_LONDON_MSOA_AREAS as ALL_LONDON_AREAS
from greaterLondonAreas import REPRESENTATIVE_SUBSET

ALL_VALID_AREAS = list(dict.fromkeys(ALL_OXFORD_AREAS + list(ALL_LONDON_AREAS)))

SCRIPT_OPTIONS = {
    '1': {
        'name': 'Person/Individual Generation',
        'script': 'generateIndividuals.py',
        # 'script': 'generatedIndividuals_V4.py',
        'description': 'Generate synthetic individuals with demographic attributes',
        'requires_area': True
    },
    '2': {
        'name': 'Household Generation', 
        'script': 'generateHouseholds.py',
        # 'script': 'generatedHouseholds_V3.py',
        'description': 'Generate synthetic households with composition and characteristics',
        'requires_area': True
    },
    '3': {
        'name': 'Household Assignment',
        'script': 'assignHouseholds.py',
        'description': 'Hyperparameter tuning for household assignment models',
        'requires_area': True
    },
    '4': {
        'name': 'Create Master Glossary',
        'script': 'Utils/createGlossary.py',
        'description': 'Combine all crosstable glossaries into a single master file',
        'requires_area': True
    },
    '5': {
        'name': 'Persons Convergence & Performance Plots',
        'script': 'Utils/plotConvergencePerformance.py',
        'description': 'Generate convergence and performance plots for individuals/persons only',
        'requires_area': False,
        'plot_type': 'individuals'
    },
    '6': {
        'name': 'Households Convergence & Performance Plots', 
        'script': 'Utils/plotConvergencePerformance.py',
        'description': 'Generate convergence and performance plots for households only',
        'requires_area': False,
        'plot_type': 'households'
    },
    '7': {
        'name': 'All Convergence & Performance Plots',
        'script': 'Utils/plotConvergencePerformance.py',
        'description': 'Generate convergence and performance plots for both persons and households',
        'requires_area': False,
        'plot_type': 'both'
    },
    '8': {
        'name': 'Run Multiple Areas (Batch)',
        'script': 'Utils/runMultipleAreas.py',
        'description': 'Run individuals and households generation for multiple predefined areas',
        'requires_area': False
    },
    '9': {
        'name': 'Run Assignment HP Tuning (All Areas)',
        'script': 'Utils/runAssignmentHPTuning.py',
        'description': 'Run household assignment hyperparameter tuning for all 17 areas',
        'requires_area': False
    },
    '10': {
        'name': 'Model Evaluation',
        'script': 'evaluation.py',
        'description': 'Load and evaluate existing model outputs with crosstable comparisons',
        'requires_area': False,
        'special_execution': True
    }
}

def display_menu():
    """Display the main menu options."""
    print("\n" + "="*60)
    print("SP_GNN - Synthetic Population Generation Using Graph Neural Network")
    print("="*60)
    print("\nSelect the operation to perform:")
    
    # Group options by type
    generation_options = ['1', '2', '3', '4']
    analysis_options = ['5', '6', '7', '8', '9', '10']
    
    print("\n📊 Generation & Processing:")
    for key in generation_options:
        if key in SCRIPT_OPTIONS:
            value = SCRIPT_OPTIONS[key]
            area_req = "🎯" if value.get('requires_area', True) else "🌐"
            print(f"  {key}. {area_req} {value['name']}")
    
    print("\n📈 Analysis & Utilities:")
    for key in analysis_options:
        if key in SCRIPT_OPTIONS:
            value = SCRIPT_OPTIONS[key]
            area_req = "🎯" if value.get('requires_area', True) else "🌐"
            print(f"  {key}. {area_req} {value['name']}")
    
    print("\n🔄 Other:")
    print("  11. Exit")
    
    print(f"\n{'='*60}")
    print("🎯 = Requires area code selection | 🌐 = Processes all areas")
    print("="*60)

def select_region():
    """Choose Oxford (pilot) or Greater London before picking an MSOA."""
    while True:
        print("\n" + "=" * 60)
        print("Select region")
        print("=" * 60)
        print(f"  1. Oxford ({len(ALL_OXFORD_AREAS)} MSOA areas)")
        print(f"  2. Greater London ({len(ALL_LONDON_AREAS)} MSOA areas)")
        choice = input("\nSelect 1-2, or 'q' to return: ").strip()
        if choice.lower() == 'q':
            return None
        if choice == '1':
            return 'oxford'
        if choice == '2':
            return 'london'
        print("Please enter 1, 2, or 'q'.")

def select_oxford_area():
    """Allow user to select an Oxford area, or type any known MSOA code."""
    while True:
        print("\n" + "=" * 60)
        print("Available Oxford Area Codes:")
        print("=" * 60)
        for i, area in enumerate(ALL_OXFORD_AREAS, 1):
            print(f"{i:2d}. {area}")
        print(f"\nTotal Oxford areas: {len(ALL_OXFORD_AREAS)}")
        print("Or type any MSOA code in full (e.g. E02005940).")
        print("=" * 60)

        choice = input(
            f"\nSelect 1-{len(ALL_OXFORD_AREAS)}, enter an area code, or 'q' to return: "
        ).strip()

        if choice.lower() == 'q':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(ALL_OXFORD_AREAS):
            selected_area = ALL_OXFORD_AREAS[int(choice) - 1]
            print(f"\nSelected area: {selected_area}")
            return selected_area
        code = choice.upper()
        if code in ALL_VALID_AREAS:
            print(f"\nSelected area: {code}")
            return code
        print(f"'{choice}' is not a recognised Oxford or Greater London MSOA code.")

def select_london_area():
    """Select a Greater London MSOA: type a code directly, or pick from the subset.

    With 983 areas a numbered list is unusable, so the paper's five-area
    representative subset is offered as a shortlist and any other code can be
    typed in full. Codes are validated against the canonical list.
    """
    subset_notes = ['smallest', 'lower quartile', 'median', 'upper quartile', 'largest']
    while True:
        print("\n" + "=" * 60)
        print(f"Greater London: {len(ALL_LONDON_AREAS)} MSOA areas available")
        print("=" * 60)
        print("\nRepresentative subset (the areas reported in the paper):")
        for i, (area, note) in enumerate(zip(REPRESENTATIVE_SUBSET, subset_notes), 1):
            print(f"  {i}. {area}  ({note})")
        print("\nOr type any Greater London MSOA code in full (e.g. E02000001).")
        choice = input(
            f"\nSelect 1-{len(REPRESENTATIVE_SUBSET)}, enter an area code, "
            f"or 'q' to return: "
        ).strip()

        if choice.lower() == 'q':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(REPRESENTATIVE_SUBSET):
            selected_area = REPRESENTATIVE_SUBSET[int(choice) - 1]
            print(f"\nSelected area: {selected_area}")
            return selected_area
        code = choice.upper()
        if code in ALL_LONDON_AREAS:
            print(f"\nSelected area: {code}")
            return code
        print(f"'{choice}' is not a Greater London MSOA code. "
              f"Codes look like E02000001; see code/Utils/greaterLondonAreas.py for the list.")

def select_area():
    """Select region, then an MSOA within that region."""
    region = select_region()
    if region is None:
        return None
    if region == 'oxford':
        return select_oxford_area()
    return select_london_area()

def run_script(script_name, area_code=None, plot_type=None):
    """Run the selected script with or without area code depending on requirements."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_name} not found at {script_path}")
        return False
    
    try:
        print(f"\n{'='*60}")
        if area_code:
            print(f"Running {script_name} with area code: {area_code}")
        elif plot_type:
            print(f"Running {script_name} with plot type: {plot_type}")
        else:
            print(f"Running {script_name}")
        print(f"{'='*60}")
        
        # Build command based on parameters
        cmd = [sys.executable, script_path]
        if area_code:
            cmd.extend(['--area_code', area_code])
        if plot_type:
            cmd.extend(['--plot_type', plot_type])
        
        # Run the script
        result = subprocess.run(cmd, check=True, capture_output=False)
        
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
        choice = input("\nEnter your choice (1-11): ").strip()
        
        if choice == '11':
            # print("Thank you for using SP_GNN. Goodbye!")
            break
        elif choice in SCRIPT_OPTIONS:
            script_info = SCRIPT_OPTIONS[choice]
            
            # Check if script requires area code
            if script_info.get('requires_area', True):
                # Select area
                selected_area = select_area()
                if selected_area is None:
                    continue  # Return to main menu
                
                # Run the selected script with area code
                success = run_script(script_info['script'], selected_area)
            else:
                # Run the script without area code
                print(f"\n{script_info['description']}")
                if choice in ['5', '6', '7']:
                    plot_type = script_info.get('plot_type', 'both')
                    entity_type = "individuals/persons" if plot_type == 'individuals' else "households" if plot_type == 'households' else "both individuals and households"
                    print("\nThis will:")
                    print(f"  • Search for existing output folders for {entity_type}")
                    print("  • Generate interactive convergence plots (loss & accuracy over epochs)")
                    print("  • Generate performance plots (training time vs population)")
                    print("  • Generate training progress plots (population vs time progression)")
                    print("  • Save plots as HTML files in the outputs/ directory")
                elif choice == '8':
                    print("\nThis will:")
                    print("  • Run individual generation for predefined area codes")
                    print("  • Run household generation for predefined area codes") 
                    print("  • Automatically generate convergence and performance plots")
                    print("  • Process multiple areas in sequence")
                elif choice == '9':
                    print("\nThis will:")
                    print("  • Run household assignment hyperparameter tuning for all 17 areas")
                    print("  • Check for required input files (person_nodes.pt, household_nodes.pt)")
                    print("  • Skip areas with missing input files")
                    print("  • Generate assignment performance plots and analysis")
                    print("  • Create detailed hyperparameter tuning results")
                elif choice == '10':
                    print("\nThis will:")
                    print("  • Load existing model outputs (person_nodes.pt or household_nodes.pt)")
                    print("  • Allow selection between individuals and households evaluation")
                    print("  • Select an Oxford or Greater London area")
                    print("  • Generate crosstable comparison plots (actual vs predicted)")
                    print("  • Calculate accuracy metrics (R² and RMSE)")
                    print("  • Save interactive HTML plots to the outputs directory")
                
                # Handle special execution for evaluation script
                if script_info.get('special_execution'):
                    try:
                        # Import and run evaluation script directly
                        import evaluation
                        evaluation.main()
                        success = True
                    except Exception as e:
                        print(f"Error running evaluation: {e}")
                        success = False
                else:
                    # Check if script has plot_type parameter
                    plot_type = script_info.get('plot_type')
                    success = run_script(script_info['script'], plot_type=plot_type)
            
            if success:
                input("\nPress Enter to continue...")
            else:
                input("\nScript execution failed. Press Enter to continue...")
        else:
            print("Invalid choice. Please enter a number between 1 and 11.")

if __name__ == "__main__":
    # Support command line arguments for automated execution
    parser = argparse.ArgumentParser(description='SP_GNN Synthetic Population Generator')
    parser.add_argument('--script', choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                       help='Script to run (1=Individual, 2=Household, 3=Assignment, 4=Create Master Glossary, 5=Persons Plots, 6=Households Plots, 7=All Plots, 8=Run Multiple Areas, 9=Assignment HP Tuning All Areas, 10=Model Evaluation)')
    parser.add_argument('--area_code', choices=ALL_VALID_AREAS, metavar='AREA_CODE',
                       help='Oxford or Greater London MSOA code, e.g. E02005940 or E02000001 '
                            '(not required for scripts 4, 5, 6, 7, 8, 9, 10)')
    
    args = parser.parse_args()
    
    # If command line arguments provided, run directly
    if args.script:
        script_info = SCRIPT_OPTIONS[args.script]
        
        # Check if script requires area code
        if script_info.get('requires_area', True):
            if not args.area_code:
                print(f"Error: Script '{script_info['name']}' requires an area code.")
                print(f"Oxford: {len(ALL_OXFORD_AREAS)} areas; Greater London: {len(ALL_LONDON_AREAS)} "
                      f"(see code/Utils/greaterLondonAreas.py)")
                sys.exit(1)
            success = run_script(script_info['script'], args.area_code)
        else:
            # Script doesn't require area code
            if script_info.get('special_execution'):
                # Handle special execution for evaluation script
                try:
                    import evaluation
                    evaluation.main()
                    success = True
                except Exception as e:
                    print(f"Error running evaluation: {e}")
                    success = False
            else:
                plot_type = script_info.get('plot_type')
                success = run_script(script_info['script'], plot_type=plot_type)
        
        if not success:
            sys.exit(1)
    else:
        # Otherwise, run interactive mode
        main()

#!/usr/bin/env python3
"""
Script to run multiple area codes for both individuals and households generation,
then automatically generate convergence and performance plots.
"""

import os
import sys
import subprocess
import time

# Define the area codes to process
AREA_CODES = [
    'E02005940', 'E02005941', 'E02005942', 'E02005943',
    'E02005944', 'E02005945', 'E02005946', 'E02005947',
    'E02005948', 'E02005949', 'E02005950', 'E02005951',
    'E02005953', 'E02005954', 'E02005955', 'E02005956',
    'E02005957'
]

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Completed successfully in {duration:.1f} seconds")
            return True
        else:
            print(f"\n❌ Failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Main function to run multiple areas"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("RUNNING MULTIPLE AREA CODES FOR GNN POPULATION SYNTHESIS")
    print("="*80)
    print(f"Processing area codes: {', '.join(AREA_CODES)}")
    print(f"Current directory: {current_dir}")
    
    # Track results
    individuals_results = {}
    households_results = {}
    
    # Process each area code
    for area_code in AREA_CODES:
        print(f"\n{'='*80}")
        print(f"PROCESSING AREA CODE: {area_code}")
        print(f"{'='*80}")
        
        # Run individuals generation
        individuals_cmd = [sys.executable, 'generatedIndividuals.py', '--area_code', area_code]
        individuals_success = run_command(
            individuals_cmd, 
            f"Individuals Generation for {area_code}"
        )
        individuals_results[area_code] = individuals_success
        
        # Run households generation
        households_cmd = [sys.executable, 'generatedHouseholds.py', '--area_code', area_code]
        households_success = run_command(
            households_cmd, 
            f"Households Generation for {area_code}"
        )
        households_results[area_code] = households_success
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING CONVERGENCE AND PERFORMANCE PLOTS")
    print(f"{'='*80}")
    
    plot_cmd = [sys.executable, 'plotConvergencePerformance.py']
    plot_success = run_command(plot_cmd, "Convergence and Performance Plotting")
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    print("\nIndividuals Generation Results:")
    for area_code, success in individuals_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {area_code}: {status}")
    
    print("\nHouseholds Generation Results:")
    for area_code, success in households_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {area_code}: {status}")
    
    plot_status = "✅ SUCCESS" if plot_success else "❌ FAILED"
    print(f"\nPlotting: {plot_status}")
    
    # Overall status
    all_individuals_success = all(individuals_results.values())
    all_households_success = all(households_results.values())
    overall_success = all_individuals_success and all_households_success and plot_success
    
    print(f"\n{'='*80}")
    if overall_success:
        print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("\nGenerated files can be found in:")
        print("  - code/outputs/individuals_{area_code}/")
        print("  - code/outputs/households_{area_code}/")
        print("  - code/outputs/convergence_plots.html")
        print("  - code/outputs/performance_plots.html")
    else:
        print("⚠️  SOME TASKS FAILED - CHECK LOGS ABOVE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
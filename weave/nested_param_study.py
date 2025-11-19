#!/usr/bin/env python3
"""
Full factorial parameter sweep over physical parameters (A, a, L, M, gamma).

For each combination, runs a complete (α, ε) parameter study via run_alpha_epsilon_study.py.

This creates a nested directory structure:
    base_dir/
        A4.0_a4.0_L1.0_M1.0_gamma1.0/
            alpha_0.5000_eps_0.0500/
                stats.json
            alpha_0.5000_eps_0.0794/
                stats.json
            ...
            study_summary.json
            mobility_phase_diagram.png
        A4.0_a4.0_L1.0_M1.0_gamma2.0/
            ...
        ...
"""

import subprocess
import numpy as np
import json
import itertools
from pathlib import Path
import time


# =============================================================================
# CONFIGURATION - Edit these lists to define your parameter sweep
# =============================================================================

# Physical parameters to sweep
A_VALUES = [1.0]                               # Barrier amplitude
a_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]          # Shape factor (barrier sharpness)
L_VALUES = [0.25, 1.0, 4.0, 10.0]              # Length scale in x-y direction
M_VALUES = [1.0]                               # Length scale in x+y direction
gamma_VALUES = [0.1, 0.25, 1.0, 4.0, 10.0]     # Damping coefficient

# Fixed parameters for all studies
MASS = 1.0

# Alpha-epsilon study parameters (passed to each inner study)
ALPHA_MIN = 0.25
ALPHA_MAX = 20.0
N_ALPHA = 20

EPSILON_MIN = 0.25
EPSILON_MAX = 2.5
N_EPSILON = N_ALPHA

# Simulation parameters
DT = 0.001
NSTEPS = 20000
NTRAJS = 100
NCORES = 2         
OUTER_NCORES = 24  

# Paths
BASE_DIR = "data/alpha-eps_parameter-sweep"
ALPHA_EPSILON_SCRIPT = "./alpha_eps_study.py"
WEAVE_SCRIPT = "./weave_parallel.py"

# Control flags
SKIP_EXISTING = True  # Skip studies that already have study_summary.json
DRY_RUN = False       # If True, print commands without executing


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_study_dirname(A, a, L, M, gamma):
    """
    Generate directory name for a specific parameter combination.
    
    Format: A{A}_a{a}_L{L}_M{M}_gamma{gamma}
    """
    return f"A{A:.1f}_a{a:.1f}_L{L:.1f}_M{M:.1f}_gamma{gamma:.1f}"


def run_alpha_epsilon_study(A, a, L, M, gamma, study_dir):
    """
    Run a single (α, ε) parameter study for given physical parameters.
    
    Returns:
    --------
    success : bool
        True if study completed successfully
    """
    # Build command
    cmd = [
        'python', ALPHA_EPSILON_SCRIPT,
        '--alpha_min', str(ALPHA_MIN),
        '--alpha_max', str(ALPHA_MAX),
        '--n_alpha', str(N_ALPHA),
        '--eps_min', str(EPSILON_MIN),
        '--eps_max', str(EPSILON_MAX),
        '--n_eps', str(N_EPSILON),
        '--A', str(A),
        '--a', str(a),
        '--L', str(L),
        '--M', str(M),
        '--gamma', str(gamma),
        '--m', str(MASS),
        '--dt', str(DT),
        '--nsteps', str(NSTEPS),
        '--ntrajs', str(NTRAJS),
        '--study_dir', str(study_dir),
        '--weave_script', WEAVE_SCRIPT,
    ]
    
    if NCORES is not None:
        cmd.extend(['--ncores', str(NCORES)])
    
    if OUTER_NCORES is not None:
        cmd.extend(['--outer_ncores', str(OUTER_NCORES)])
    
    if SKIP_EXISTING:
        cmd.append('--skip_existing')
    
    # Print command
    print(f"\nRunning: {' '.join(cmd)}")
    
    if DRY_RUN:
        print("  (dry run - not executing)")
        return True
    
    # Run study
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed with error:\n{e.stderr}")
        return False


def save_sweep_summary(base_dir, param_combinations, results):
    """
    Save summary of entire parameter sweep.
    """
    summary = {
        'parameter_ranges': {
            'A': A_VALUES,
            'a': a_VALUES,
            'L': L_VALUES,
            'M': M_VALUES,
            'gamma': gamma_VALUES
        },
        'alpha_epsilon_study_params': {
            'alpha_min': ALPHA_MIN,
            'alpha_max': ALPHA_MAX,
            'n_alpha': N_ALPHA,
            'epsilon_min': EPSILON_MIN,
            'epsilon_max': EPSILON_MAX,
            'n_epsilon': N_EPSILON
        },
        'simulation_params': {
            'm': MASS,
            'dt': DT,
            'nsteps': NSTEPS,
            'ntrajs': NTRAJS
        },
        'results': results
    }
    
    summary_file = Path(base_dir) / 'sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved sweep summary: {summary_file}")


def extract_optimal_points(base_dir, param_combinations):
    """
    Extract optimal (α*, ε*) and max mobility from each study.
    
    Returns:
    --------
    optimal_data : list of dicts
        Each dict contains parameters and optimal point info
    """
    optimal_data = []
    
    for params in param_combinations:
        A, a, L, M, gamma = params
        study_dir = Path(base_dir) / get_study_dirname(A, a, L, M, gamma)
        summary_file = study_dir / 'study_summary.json'
        
        if not summary_file.exists():
            print(f"Warning: No summary for {study_dir.name}")
            continue
        
        # Load study summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        optimal = summary.get('optimal', None)
        
        if optimal:
            optimal_data.append({
                'A': A,
                'a': a,
                'L': L,
                'M': M,
                'gamma': gamma,
                'zeta': gamma**2 / (4 * MASS * A / L**2),  # Damping parameter
                'alpha_opt': optimal.get('alpha_opt'),
                'epsilon_opt': optimal.get('epsilon_opt'),
                'mu_max': optimal.get('mu_max')
            })
    
    return optimal_data


def print_optimal_points_table(optimal_data):
    """
    Print a nice table of optimal points.
    """
    print("\n" + "="*100)
    print("OPTIMAL POINTS SUMMARY")
    print("="*100)
    print(f"{'A':>6} {'a':>6} {'L':>6} {'M':>6} {'gamma':>6} {'ζ':>8} {'α*':>8} {'ε*':>8} {'μ̃_max':>10}")
    print("-"*100)
    
    for data in optimal_data:
        print(f"{data['A']:6.1f} {data['a']:6.1f} {data['L']:6.1f} {data['M']:6.1f} "
              f"{data['gamma']:6.1f} {data['zeta']:8.3f} {data['alpha_opt']:8.2f} "
              f"{data['epsilon_opt']:8.2f} {data['mu_max']:10.4f}")
    
    print("="*100 + "\n")


def analyze_trends(optimal_data):
    """
    Analyze how optimal point shifts with physical parameters.
    """
    import matplotlib.pyplot as plt
    
    # Convert to arrays for analysis
    A_vals = np.array([d['A'] for d in optimal_data])
    a_vals = np.array([d['a'] for d in optimal_data])
    gamma_vals = np.array([d['gamma'] for d in optimal_data])
    zeta_vals = np.array([d['zeta'] for d in optimal_data])
    alpha_opt_vals = np.array([d['alpha_opt'] for d in optimal_data])
    epsilon_opt_vals = np.array([d['epsilon_opt'] for d in optimal_data])
    mu_max_vals = np.array([d['mu_max'] for d in optimal_data])
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # α* vs. shape factor a
    axes[0, 0].scatter(a_vals, alpha_opt_vals, c=gamma_vals, s=100, 
                      cmap='viridis', edgecolors='black', linewidths=1)
    axes[0, 0].set_xlabel('Shape factor a', fontsize=12)
    axes[0, 0].set_ylabel('Optimal α*', fontsize=12)
    axes[0, 0].set_title('Optimal Temperature vs. Barrier Sharpness', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ε* vs. shape factor a
    axes[0, 1].scatter(a_vals, epsilon_opt_vals, c=gamma_vals, s=100,
                      cmap='viridis', edgecolors='black', linewidths=1)
    axes[0, 1].set_xlabel('Shape factor a', fontsize=12)
    axes[0, 1].set_ylabel('Optimal ε*', fontsize=12)
    axes[0, 1].set_title('Optimal Forcing vs. Barrier Sharpness', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # μ̃_max vs. shape factor a
    im = axes[0, 2].scatter(a_vals, mu_max_vals, c=gamma_vals, s=100,
                           cmap='viridis', edgecolors='black', linewidths=1)
    axes[0, 2].set_xlabel('Shape factor a', fontsize=12)
    axes[0, 2].set_ylabel('Max mobility μ̃_max', fontsize=12)
    axes[0, 2].set_title('Maximum Mobility vs. Barrier Sharpness', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=axes[0, 2], label='γ (damping)')
    
    # α* vs. damping (ζ)
    axes[1, 0].scatter(zeta_vals, alpha_opt_vals, c=a_vals, s=100,
                      cmap='plasma', edgecolors='black', linewidths=1)
    axes[1, 0].set_xlabel('ζ = γ²/(4mA/L²) (damping parameter)', fontsize=12)
    axes[1, 0].set_ylabel('Optimal α*', fontsize=12)
    axes[1, 0].set_title('Optimal Temperature vs. Damping', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ε* vs. damping (ζ)
    axes[1, 1].scatter(zeta_vals, epsilon_opt_vals, c=a_vals, s=100,
                      cmap='plasma', edgecolors='black', linewidths=1)
    axes[1, 1].set_xlabel('ζ (damping parameter)', fontsize=12)
    axes[1, 1].set_ylabel('Optimal ε*', fontsize=12)
    axes[1, 1].set_title('Optimal Forcing vs. Damping', fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # μ̃_max vs. damping (ζ)
    im2 = axes[1, 2].scatter(zeta_vals, mu_max_vals, c=a_vals, s=100,
                            cmap='plasma', edgecolors='black', linewidths=1)
    axes[1, 2].set_xlabel('ζ (damping parameter)', fontsize=12)
    axes[1, 2].set_ylabel('Max mobility μ̃_max', fontsize=12)
    axes[1, 2].set_title('Maximum Mobility vs. Damping', fontweight='bold')
    axes[1, 2].set_xscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=axes[1, 2], label='a (shape factor)')
    
    plt.suptitle('Trends in Optimal Operating Points', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(BASE_DIR) / 'optimal_points_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved analysis plot: {output_path}")
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution: nested parameter sweep.
    """
    print("="*100)
    print("FULL FACTORIAL PARAMETER SWEEP")
    print("="*100)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        A_VALUES, a_VALUES, L_VALUES, M_VALUES, gamma_VALUES
    ))
    
    n_total = len(param_combinations)
    n_per_study = N_ALPHA * N_EPSILON
    
    print(f"\nParameter combinations: {n_total}")
    print(f"Simulations per study: {n_per_study}")
    print(f"Total simulations: {n_total * n_per_study}")
    print(f"\nBase directory: {BASE_DIR}")
    print(f"\nParameter ranges:")
    print(f"  A:     {A_VALUES}")
    print(f"  a:     {a_VALUES}")
    print(f"  L:     {L_VALUES}")
    print(f"  M:     {M_VALUES}")
    print(f"  gamma: {gamma_VALUES}")
    print(f"\nAlpha-epsilon grid:")
    print(f"  α: [{ALPHA_MIN}, {ALPHA_MAX}] ({N_ALPHA} points)")
    print(f"  ε: [{EPSILON_MIN}, {EPSILON_MAX}] ({N_EPSILON} points)")
    
    if DRY_RUN:
        print("\n*** DRY RUN MODE - Commands will be printed but not executed ***")
    
    # Create base directory
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run studies
    print(f"\n{'='*100}")
    print("RUNNING NESTED STUDIES")
    print(f"{'='*100}\n")
    
    results = []
    completed = 0
    skipped = 0
    failed = 0
    
    start_time = time.time()
    
    for i, params in enumerate(param_combinations):
        A, a, L, M, gamma = params
        
        print(f"\n{'='*100}")
        print(f"[{i+1}/{n_total}] A={A}, a={a}, L={L}, M={M}, gamma={gamma}")
        print(f"{'='*100}")
        
        # Generate study directory
        study_dirname = get_study_dirname(A, a, L, M, gamma)
        study_dir = Path(BASE_DIR) / study_dirname
        
        # Check if already completed
        summary_file = study_dir / 'study_summary.json'
        if SKIP_EXISTING and summary_file.exists():
            print(f"Skipping (study_summary.json exists): {study_dirname}")
            skipped += 1
            results.append({
                'params': {'A': A, 'a': a, 'L': L, 'M': M, 'gamma': gamma},
                'study_dir': study_dirname,
                'status': 'skipped'
            })
            continue
        
        # Run study
        success = run_alpha_epsilon_study(A, a, L, M, gamma, study_dir)
        
        if success:
            completed += 1
            status = 'completed'
        else:
            failed += 1
            status = 'failed'
        
        results.append({
            'params': {'A': A, 'a': a, 'L': L, 'M': M, 'gamma': gamma},
            'study_dir': study_dirname,
            'status': status
        })
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*100}")
    print("SWEEP SUMMARY")
    print(f"{'='*100}")
    print(f"Total studies:    {n_total}")
    print(f"Completed:        {completed}")
    print(f"Skipped:          {skipped}")
    print(f"Failed:           {failed}")
    print(f"Total time:       {elapsed/3600:.2f} hours")
    print(f"{'='*100}\n")
    
    # Save sweep summary
    save_sweep_summary(BASE_DIR, param_combinations, results)
    
    # Extract and analyze optimal points
    if not DRY_RUN:
        print("\nExtracting optimal points from all studies...")
        optimal_data = extract_optimal_points(BASE_DIR, param_combinations)
        
        if optimal_data:
            # Print table
            print_optimal_points_table(optimal_data)
            
            # Save optimal points
            optimal_file = Path(BASE_DIR) / 'optimal_points.json'
            with open(optimal_file, 'w') as f:
                json.dump(optimal_data, f, indent=2)
            print(f"Saved optimal points: {optimal_file}")
            
            # Analyze trends
            print("\nAnalyzing trends...")
            analyze_trends(optimal_data)
        else:
            print("No optimal points found (no completed studies?)")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

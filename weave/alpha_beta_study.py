#!/usr/bin/env python3
"""
Driver script for α-β parameter study of weave potential.

This script:
1. Generates a grid of (α, β) parameters
2. Runs weave.py for each parameter set via command line
3. Stores results in organized directories
4. Analyzes and plots results from stats.json files
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os
import argparse
from pathlib import Path
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run α-β parameter study for weave simulations'
    )
    
    # Study parameters
    parser.add_argument('--alpha_min', type=float, default=0.1,
                       help='Minimum α = A/kBT')
    parser.add_argument('--alpha_max', type=float, default=30.0,
                       help='Maximum α = A/kBT')
    parser.add_argument('--n_alpha', type=int, default=15,
                       help='Number of α values')
    
    parser.add_argument('--beta_min', type=float, default=0.1,
                       help='Minimum β = F·L/kBT')
    parser.add_argument('--beta_max', type=float, default=30.0,
                       help='Maximum β = F·L/kBT')
    parser.add_argument('--n_beta', type=int, default=15,
                       help='Number of β values')
    
    # Fixed physical parameters
    parser.add_argument('--A', type=float, default=4.0,
                       help='Barrier amplitude')
    parser.add_argument('--L', type=float, default=1.0,
                       help='Length scale in x-y direction')
    parser.add_argument('--M', type=float, default=1.0,
                       help='Length scale in x+y direction')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Damping coefficient')
    parser.add_argument('--m', type=float, default=1.0,
                       help='Particle mass')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    parser.add_argument('--nsteps', type=int, default=2000,
                       help='Number of steps per trajectory')
    parser.add_argument('--ntrajs', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--ncores', type=int, default=None,
                       help='Number of cores for parallelization')
    
    # Directory management
    parser.add_argument('--study_dir', type=str, default='alpha_beta_study',
                       help='Base directory for study results')
    parser.add_argument('--weave_script', type=str, default='./weave.py',
                       help='Path to weave.py script')
    
    # Control
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip simulations that already have stats.json')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without running')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from existing data')
    
    return parser.parse_args()


def generate_parameter_grid(args):
    """
    Generate (α, β) parameter grid.
    
    Returns:
    --------
    param_list : list of dicts
        Each dict contains simulation parameters
    """
    # Generate logarithmic grids
    alpha_vals = np.logspace(np.log10(args.alpha_min), 
                             np.log10(args.alpha_max), 
                             args.n_alpha)
    beta_vals = np.logspace(np.log10(args.beta_min), 
                           np.log10(args.beta_max), 
                           args.n_beta)
    
    param_list = []
    
    for alpha in alpha_vals:
        for beta in beta_vals:
            # Convert dimensionless to physical parameters
            # α = A/kBT  →  kBT = A/α
            kT = args.A / alpha
            
            # β = F·L/kBT  →  F = β·kBT/L
            gradp = beta * kT / args.L
            
            # Create parameter dict
            params = {
                'alpha': alpha,
                'beta': beta,
                'kT': kT,
                'Fpx': gradp,
                'A': args.A,
                'a': alpha,  # Note: 'a' in weave.py is the shape parameter
                'L': args.L,
                'M': args.M,
                'gamma': args.gamma,
                'm': args.m,
                'dt': args.dt,
                'nsteps': args.nsteps,
                'ntrajs': args.ntrajs,
                'ncores': args.ncores if args.ncores else '',
            }
            
            param_list.append(params)
    
    return param_list, alpha_vals, beta_vals


def get_output_dir(study_dir, alpha, beta):
    """
    Generate output directory name for given parameters.
    
    Format: study_dir/alpha_{alpha:.4f}_beta_{beta:.4f}
    """
    dirname = f"alpha_{alpha:.4f}_beta_{beta:.4f}"
    return Path(study_dir) / dirname


def run_simulation(params, outdir, weave_script, dry_run=False):
    """
    Run a single weave.py simulation via command line.
    
    Returns:
    --------
    success : bool
        True if simulation completed successfully
    """
    # Build command
    cmd = [
        'python', weave_script,
        '--m', str(params['m']),
        '--gamma', str(params['gamma']),
        '--kT', str(params['kT']),
        '--Fpx', str(params['Fpx']),
        '--A', str(params['A']),
        '--a', str(params['a']),
        '--L', str(params['L']),
        '--M', str(params['M']),
        '--dt', str(params['dt']),
        '--nsteps', str(params['nsteps']),
        '--ntrajs', str(params['ntrajs']),
        '--outdir', str(outdir),
    ]
    
    if params['ncores']:
        cmd.extend(['--ncores', str(params['ncores'])])
    
    # Print command
    cmd_str = ' '.join(cmd)
    print(f"\nRunning: {cmd_str}")
    
    if dry_run:
        print("  (dry run - not executing)")
        return True
    
    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed with error:\n{e.stderr}")
        return False


def load_results(study_dir, alpha_vals, beta_vals):
    """
    Load all stats.json files from study directory.
    
    Returns:
    --------
    results : dict
        Nested dict: results[alpha][beta] = stats_dict
    """
    results = {}
    
    for alpha in alpha_vals:
        results[alpha] = {}
        for beta in beta_vals:
            outdir = get_output_dir(study_dir, alpha, beta)
            stats_file = outdir / 'stats.json'
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                results[alpha][beta] = stats
            else:
                results[alpha][beta] = None
                print(f"Warning: No results for α={alpha:.4f}, β={beta:.4f}")
    
    return results


def extract_mobility_grid(results, alpha_vals, beta_vals, L, gamma, A):
    """
    Extract dimensionless mobility from results.
    
    Returns:
    --------
    mu_xx_grid, mu_yx_grid : 2D arrays
        Dimensionless mobility grids
    """
    n_alpha = len(alpha_vals)
    n_beta = len(beta_vals)
    
    mu_xx_grid = np.full((n_alpha, n_beta), np.nan)
    mu_xy_grid = np.full((n_alpha, n_beta), np.nan)
    D_xx_grid = np.full((n_alpha, n_beta), np.nan)
    D_xy_grid = np.full((n_alpha, n_beta), np.nan)
    D_yy_grid = np.full((n_alpha, n_beta), np.nan)
    
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            if results[alpha][beta] is not None:
                stats = results[alpha][beta]
                kT = A / alpha
                Fpx = beta * kT / L
                
                # Extract mobility (assume stored in stats)
                if 'D_xx' in stats and 'D_xy' in stats:
                    D_xx_grid[i, j] = stats['D_xx'] * gamma / kT
                    D_xy_grid[i, j] = stats['D_xy'] * gamma / kT
                    D_yy_grid[i, j] = stats['D_yy'] * gamma / kT

                if 'xf' in stats and 'yf' in stats and 'tf' in stats:
                    mu_xx_grid[i, j] = stats['xf'] / (stats['tf'] * Fpx) * gamma
                    mu_xy_grid[i, j] = stats['yf'] / (stats['tf'] * Fpx) * gamma
    
    return mu_xx_grid, mu_yx_grid, D_xx_grid, D_xy_grid, D_yy_grid


def plot_mobility_phase_diagram(alpha_vals, beta_vals, mu_xx_grid, study_dir):
    """
    Create main phase diagram plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    Beta, Alpha = np.meshgrid(beta_vals, alpha_vals)
    
    # Plot heatmap
    levels = 20
    contour = ax.contourf(Beta, Alpha, mu_xx_grid, 
                          levels=levels, cmap='viridis')
    
    # Contour lines
    contour_lines = ax.contour(Beta, Alpha, mu_xx_grid, 
                               levels=10, colors='white', 
                               linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Find and mark maximum
    if not np.all(np.isnan(mu_xx_grid)):
        max_idx = np.unravel_index(np.nanargmax(mu_xx_grid), mu_xx_grid.shape)
        alpha_opt = alpha_vals[max_idx[0]]
        beta_opt = beta_vals[max_idx[1]]
        mu_max = mu_xx_grid[max_idx]
        
        ax.plot(beta_opt, alpha_opt, 'r*', markersize=30, 
                markeredgecolor='white', markeredgewidth=2,
                label=f'Max: α={alpha_opt:.2f}, β={beta_opt:.2f}, μ̃={mu_max:.3f}')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$β = F_x L / (k_B T)$ [driving strength]', fontsize=14)
    ax.set_ylabel('$α = A / (k_B T)$ [inverse temperature]', fontsize=14)
    ax.set_title('Dimensionless Mobility $\\tilde{μ}_{xx}(α, β)$', 
                 fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='$\\tilde{μ}_{xx} = μ_{xx} · γ$')
    
    # Legend
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(study_dir) / 'mobility_phase_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return fig


def plot_mobility_slices(alpha_vals, beta_vals, mu_xx_grid, study_dir):
    """
    Plot 1D slices through parameter space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: μ vs β for fixed α values
    alpha_samples = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    for alpha_sample in alpha_samples:
        idx = np.argmin(np.abs(alpha_vals - alpha_sample))
        alpha_actual = alpha_vals[idx]
        axes[0].plot(beta_vals, mu_xx_grid[idx, :], 
                    marker='o', label=f'α = {alpha_actual:.2f}')
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('$β$ (driving strength)', fontsize=12)
    axes[0].set_ylabel('$\\tilde{μ}_{xx}$', fontsize=12)
    axes[0].set_title('Mobility vs. Driving\n(fixed temperature)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: μ vs α for fixed β values
    beta_samples = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    for beta_sample in beta_samples:
        idx = np.argmin(np.abs(beta_vals - beta_sample))
        beta_actual = beta_vals[idx]
        axes[1].plot(alpha_vals, mu_xx_grid[:, idx], 
                    marker='o', label=f'β = {beta_actual:.2f}')
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$α$ (inverse temperature)', fontsize=12)
    axes[1].set_ylabel('$\\tilde{μ}_{xx}$', fontsize=12)
    axes[1].set_title('Mobility vs. Temperature\n(fixed driving)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(study_dir) / 'mobility_slices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def analyze_optimal_point(alpha_vals, beta_vals, mu_xx_grid):
    """
    Find and characterize the optimal point.
    """
    if np.all(np.isnan(mu_xx_grid)):
        print("\nNo valid data to analyze!")
        return None
    
    # Find maximum
    max_idx = np.unravel_index(np.nanargmax(mu_xx_grid), mu_xx_grid.shape)
    alpha_opt = alpha_vals[max_idx[0]]
    beta_opt = beta_vals[max_idx[1]]
    mu_max = mu_xx_grid[max_idx]
    
    print("\n" + "="*60)
    print("OPTIMAL POINT ANALYSIS")
    print("="*60)
    print(f"\nOptimal parameters:")
    print(f"  α* = {alpha_opt:.3f}  (A/kBT)")
    print(f"  β* = {beta_opt:.3f}  (F·L/kBT)")
    print(f"  μ̃_max = {mu_max:.4f}")
    print(f"\nPhysical interpretation:")
    print(f"  Optimal kBT/A = {1/alpha_opt:.3f}")
    print(f"  Optimal F·L/kBT = {beta_opt:.3f}")
    print("="*60 + "\n")
    
    return {'alpha_opt': alpha_opt, 'beta_opt': beta_opt, 'mu_max': mu_max}


def main():
    """Main execution."""
    args = parse_arguments()
    
    print("="*60)
    print("α-β PARAMETER STUDY")
    print("="*60)
    print(f"\nStudy directory: {args.study_dir}")
    print(f"Parameter ranges:")
    print(f"  α: [{args.alpha_min}, {args.alpha_max}] ({args.n_alpha} points)")
    print(f"  β: [{args.beta_min}, {args.beta_max}] ({args.n_beta} points)")
    print(f"Total simulations: {args.n_alpha * args.n_beta}")
    
    # Generate parameter grid
    param_list, alpha_vals, beta_vals = generate_parameter_grid(args)
    
    if not args.plot_only:
        # Run simulations
        print(f"\n{'='*60}")
        print("RUNNING SIMULATIONS")
        print(f"{'='*60}\n")
        
        completed = 0
        skipped = 0
        failed = 0
        
        start_time = time.time()
        
        for i, params in enumerate(param_list):
            alpha = params['alpha']
            beta = params['beta']
            outdir = get_output_dir(args.study_dir, alpha, beta)
            stats_file = outdir / 'stats.json'
            
            print(f"\n[{i+1}/{len(param_list)}] α={alpha:.4f}, β={beta:.4f}")
            
            # Check if already exists
            if args.skip_existing and stats_file.exists():
                print(f"  Skipping (stats.json exists)")
                skipped += 1
                continue
            
            # Run simulation
            success = run_simulation(params, outdir, args.weave_script, args.dry_run)
            
            if success:
                completed += 1
            else:
                failed += 1
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("SIMULATION SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {completed}")
        print(f"Skipped:   {skipped}")
        print(f"Failed:    {failed}")
        print(f"Time:      {elapsed/60:.1f} minutes")
        print(f"{'='*60}\n")
    
    # Load and plot results
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS")
    print(f"{'='*60}\n")
    
    results = load_results(args.study_dir, alpha_vals, beta_vals)
    mu_xx_grid, mu_yx_grid = extract_mobility_grid(results, alpha_vals, beta_vals, 
                                                   args.L, args.gamma, args.A)
    
    # Generate plots
    plot_mobility_phase_diagram(alpha_vals, beta_vals, mu_xx_grid, args.study_dir)
    plot_mobility_slices(alpha_vals, beta_vals, mu_xx_grid, args.study_dir)
    
    # Analyze optimal point
    optimal = analyze_optimal_point(alpha_vals, beta_vals, mu_xx_grid)
    
    # Save summary
    summary = {
        'alpha_vals': alpha_vals.tolist(),
        'beta_vals': beta_vals.tolist(),
        'mu_xx_grid': mu_xx_grid.tolist(),
        'mu_yx_grid': mu_yx_grid.tolist(),
        'optimal': optimal,
        'parameters': vars(args)
    }
    
    summary_file = Path(args.study_dir) / 'study_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved study summary: {summary_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

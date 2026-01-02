#!/usr/bin/env python3
"""
Driver script for γ-ε_y parameter study of weave potential.

This script:
1. Generates a grid of (γ, ε_y | ε_x) parameters
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
import copy
import pprint
from multiprocessing import Pool, cpu_count

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run γ-ε_y parameter study for weave simulations', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Study parameters
    parser.add_argument('--gamma_min', type=float, default=0.2,
                       help='Minimum γ (damping coefficient)')
    parser.add_argument('--gamma_max', type=float, default=10.0,
                       help='Maximum γ (damping coefficient)')
    parser.add_argument('--n_gamma', type=int, default=15,
                       help='Number of γ values')
    
    parser.add_argument('--epsy_min', type=float, default=1e-2,
                       help='Minimum ε_y = F_y·M/(A·a)')
    parser.add_argument('--epsy_max', type=float, default=2.0,
                       help='Maximum ε_y = F_y·M/(A·a)')
    parser.add_argument('--n_epsy', type=int, default=15,
                       help='Number of ε_y values')
    
    # Fixed physical parameters
    parser.add_argument('--epsx', type=float, default=1.0,
                       help='ε_x = F_x·L/(A·a)')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='α = A/kBT (inverse temperature)')
    parser.add_argument('--A', type=float, default=1.0,
                       help='Barrier amplitude')
    parser.add_argument('--a', type=float, default=1.0,
                       help='Shape factor')
    parser.add_argument('--L', type=float, default=1.0,
                       help='Length scale in x-y direction')
    parser.add_argument('--M', type=float, default=1.0,
                       help='Length scale in x+y direction')
    parser.add_argument('--m', type=float, default=1.0,
                       help='Particle mass')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.001,
                       help='Time step')
    parser.add_argument('--nsteps', type=int, default=20000,
                       help='Number of steps per trajectory')
    parser.add_argument('--ntrajs', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--outfreq', type=int, default=1,
                       help='Number of iterations per sample')
    parser.add_argument('--ncores', type=int, default=None,
                       help='Number of cores for parallelization')
    parser.add_argument('--outer_ncores', type=int, default=None,
                       help='Number of cores for outer loop parallelization')
    
    # Directory management
    parser.add_argument('--study_dir', type=str, default='gamma_epsy_study',
                       help='Base directory for study results')
    parser.add_argument('--weave_script', type=str, default='./weave_parallel.py',
                       help='Path to weave.py script')
    
    # Control
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip simulations that already have stats.json')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without running')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from existing data')
    parser.add_argument('--do_subplots', default=False, 
                        action="store_true", help='create plots from individual simulations')
    
    return parser.parse_args()

def generate_parameter_grid(args):
    """
    Generate (γ, ε_y) parameter grid with fixed temperature.
    """
    # Fixed temperature determines kBT
    kT = args.A / args.alpha
    
    # Fixed x-forcing
    Fpx = args.epsx * args.A * args.a / args.L
    
    gamma_vals = np.logspace(np.log10(args.gamma_min), 
                             np.log10(args.gamma_max), 
                             args.n_gamma)
    eps_vals = np.logspace(np.log10(args.epsy_min), 
                           np.log10(args.epsy_max), 
                           args.n_epsy)
    
    param_list = []
    for gamma in gamma_vals:
        # Compute dimensionless damping parameter ζ
        zeta = gamma**2 * args.L**2 / (4 * args.m * args.A)
        
        # Add reference simulation (no cross-forcing)
        param_list.append({
            'gamma': gamma,
            'zeta': zeta,
            'eps': 0,
            'kT': kT,
            'Fpx': Fpx,
            'Fpy': 0,
            'A': args.A,
            'a': args.a,
            'L': args.L,
            'M': args.M,
            'alpha': args.alpha,
            'm': args.m,
            'dt': args.dt,
            'nsteps': args.nsteps,
            'ntrajs': args.ntrajs,
            'outfreq': args.outfreq,
            'ncores': args.ncores if args.ncores else '',
            'do_subplots': args.do_subplots
        })

        for eps in eps_vals:
            # ε = F·M/(A·a)  →  F = ε·A·a/M
            Fpy = eps * args.A * args.a / args.M
            
            params = {
                'gamma': gamma,
                'zeta': zeta,
                'eps': eps,
                'kT': kT,
                'Fpx': Fpx,
                'Fpy': Fpy,
                'A': args.A,
                'a': args.a,
                'L': args.L,
                'M': args.M,
                'alpha': args.alpha,
                'm': args.m,
                'dt': args.dt,
                'nsteps': args.nsteps,
                'ntrajs': args.ntrajs,
                'outfreq': args.outfreq,
                'ncores': args.ncores if args.ncores else '',
                'do_subplots': args.do_subplots
            }
            
            param_list.append(params)
    
    return param_list, gamma_vals, eps_vals

def get_output_dir(study_dir, gamma, eps):
    """
    Generate output directory name for given parameters.
    
    Format: study_dir/gamma_{gamma:.4f}_eps_{eps:.4f}
    """
    dirname = f"gamma_{gamma:.4f}_eps_{eps:.4f}"
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
        '--Fpy', str(params['Fpy']),
        '--A', str(params['A']),
        '--a', str(params['a']),
        '--L', str(params['L']),
        '--M', str(params['M']),
        '--dt', str(params['dt']),
        '--nsteps', str(params['nsteps']),
        '--ntrajs', str(params['ntrajs']),
        '--outfreq', str(params['outfreq']),
        '--outdir', str(outdir)
    ]

    if params['do_subplots']:
        cmd.append('--do_plots')
    
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


def load_results(study_dir, gamma_vals, eps_vals):
    """
    Load all stats.json files from study directory.
    
    Returns:
    --------
    results : dict
        Nested dict: results[gamma][eps] = stats_dict
    """
    results = {}
    
    for gamma in gamma_vals:
        results[gamma] = {}
        for eps in (eps_vals.tolist() + [0]):
            outdir = get_output_dir(study_dir, gamma, eps)
            stats_file = outdir / 'stats.json'
            
            if stats_file.exists():
                print(f'    loading gamma = {gamma:.4f}, eps = {eps:.4f}')
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                results[gamma][eps] = stats
            else:
                results[gamma][eps] = None
                print(f"Warning: No results for γ={gamma:.4f}, ε={eps:.4f}")
    
    return results


def extract_mobility_grid(results, gamma_vals, eps_vals, epsx, L, M, A, alpha):
    """
    Extract dimensionless mobility from results.
    
    Returns:
    --------
    mu_xx_grid, mu_xy_grid : 2D arrays
        Dimensionless mobility grids
    """
    n_gamma = len(gamma_vals)
    n_eps = len(eps_vals)
    
    mu_xx_grid = np.full((n_gamma, n_eps), np.nan)
    mu_xy_grid = np.full((n_gamma, n_eps), np.nan)
    D_xx_grid = np.full((n_gamma, n_eps), np.nan)
    D_xy_grid = np.full((n_gamma, n_eps), np.nan)
    D_yy_grid = np.full((n_gamma, n_eps), np.nan)

    for i, gamma in enumerate(gamma_vals):
        ref_stats = results[gamma][0.0]
        
        if ref_stats is None:
            print(f"Warning: No reference simulation for γ={gamma:.4f}")
            continue
            
        for j, eps in enumerate(eps_vals):
            if results[gamma][eps] is not None:
                stats = results[gamma][eps]
                kT = A / alpha
                Fpx = epsx * kT / L
                Fpy = eps * kT / M
                
                # Extract diffusion coefficients
                if 'D_xx' in stats and 'D_xy' in stats:
                    D_xx_grid[i, j] = stats['D_xx'] * gamma / kT
                    D_xy_grid[i, j] = stats['D_xy'] * gamma / kT
                    D_yy_grid[i, j] = stats['D_yy'] * gamma / kT

                # Extract mobility from final positions
                if 'xf' in stats and 'yf' in stats and 'tf' in stats:
                    mu_xx_grid[i, j] = (stats['xf'] - ref_stats['xf']) / (stats['tf'] * Fpx) * gamma
                    mu_xy_grid[i, j] = (stats['yf'] - ref_stats['yf']) / (stats['tf'] * Fpx) * gamma
    
    return mu_xx_grid, mu_xy_grid, D_xx_grid, D_xy_grid, D_yy_grid


def plot_mobility_phase_diagram(gamma_vals, eps_vals, mu_xy_grid, study_dir, alpha, A, L, m):
    """
    Create main phase diagram plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    Eps, Gamma = np.meshgrid(eps_vals, gamma_vals)
    
    # Compute ζ for labeling
    Zeta = Gamma**2 * L**2 / (4 * m * A)
    
    # Plot heatmap
    levels = 20
    contour = ax.contourf(Eps, Gamma, mu_xy_grid, 
                          levels=levels, cmap='RdBu', center=0)
    
    # Contour lines
    contour_lines = ax.contour(Eps, Gamma, mu_xy_grid, 
                               levels=10, colors='black', 
                               linewidths=0.5, alpha=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Find and mark maximum
    if not np.all(np.isnan(mu_xy_grid)):
        max_idx = np.unravel_index(np.nanargmax(mu_xy_grid), mu_xy_grid.shape)
        gamma_opt = gamma_vals[max_idx[0]]
        eps_opt = eps_vals[max_idx[1]]
        mu_max = mu_xy_grid[max_idx]
        zeta_opt = gamma_opt**2 * L**2 / (4 * m * A)
        
        ax.plot(eps_opt, gamma_opt, 'r*', markersize=30, 
                markeredgecolor='white', markeredgewidth=2,
                label=f'Max: γ={gamma_opt:.2f} (ζ={zeta_opt:.2f}), ε={eps_opt:.2f}, μ̃={mu_max:.3f}')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$ε_y = F_y M / (A a)$ [cross-forcing strength]', fontsize=14)
    ax.set_ylabel('$γ$ [damping coefficient]', fontsize=14)
    ax.set_title(f'Cross-Mobility $\\tilde{{μ}}_{{xy}}(γ, ε_y)$ at α={alpha:.1f}', 
                 fontsize=16, fontweight='bold')
    
    # Add second y-axis for ζ
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    zeta_min = gamma_vals.min()**2 * L**2 / (4 * m * A)
    zeta_max = gamma_vals.max()**2 * L**2 / (4 * m * A)
    ax2.set_ylim(zeta_min, zeta_max)
    ax2.set_ylabel('$ζ = γ^2L^2/(4mA)$ [damping parameter]', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='$\\tilde{μ}_{xy}$')
    
    # Legend
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'mobility_phase_diagram.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return fig

def plot_diffusion_phase_diagrams(gamma_vals, eps_vals, D_xx_grid, D_xy_grid, D_yy_grid, study_dir, alpha, A, L, m):
    """
    Create diffusion coefficient phase diagrams.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # Create meshgrid
    Eps, Gamma = np.meshgrid(eps_vals, gamma_vals)

    for ax, grid, label, name in zip(axes, 
                                      [D_xx_grid, D_xy_grid, D_yy_grid],
                                      ['D_{xx}', 'D_{xy}', 'D_{yy}'],
                                      ['Dxx', 'Dxy', 'Dyy']):
        
        # Plot heatmap
        levels = 20
        cmap = 'RdBu' if 'xy' in name else 'viridis'
        center = 0 if 'xy' in name else None
        
        contour = ax.contourf(Eps, Gamma, grid, 
                              levels=levels, cmap=cmap)
        
        # Contour lines
        contour_lines = ax.contour(Eps, Gamma, grid, 
                                   levels=10, colors='black', 
                                   linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Find and mark extremum
        if not np.all(np.isnan(grid)):
            if 'xy' in name:
                # For cross-term, find maximum absolute value
                max_idx = np.unravel_index(np.nanargmax(np.abs(grid)), grid.shape)
            else:
                max_idx = np.unravel_index(np.nanargmax(grid), grid.shape)
                
            gamma_opt = gamma_vals[max_idx[0]]
            eps_opt = eps_vals[max_idx[1]]
            val_max = grid[max_idx]
            zeta_opt = gamma_opt**2 * L**2 / (4 * m * A)
            
            ax.plot(eps_opt, gamma_opt, 'r*', markersize=20, 
                    markeredgecolor='white', markeredgewidth=2,
                    label=f'γ={gamma_opt:.2f}, ε={eps_opt:.2f}, $\\tilde{{{label}}}$={val_max:.3f}')
        
        # Formatting
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$ε_y$ [cross-forcing]', fontsize=12)
        ax.set_ylabel('$γ$ [damping]', fontsize=12)
        ax.set_title(f'$\\tilde{{{label}}}(γ, ε_y)$', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, label=f'$\\tilde{{{label}}}$')
        
        # Legend
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle(f'Diffusion Coefficients at α={alpha:.1f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'diffusion_phase_diagrams.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return fig
    
def plot_mobility_slices(gamma_vals, eps_vals, mu_xy_grid, study_dir, n_slices=5):
    """
    Plot 1D slices through parameter space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: μ vs ε_y for fixed γ values
    gamma_samples = [10**x for x in np.linspace(np.log10(min(gamma_vals)), np.log10(max(gamma_vals)), n_slices)]
    for gamma_sample in gamma_samples:
        idx = np.argmin(np.abs(gamma_vals - gamma_sample))
        gamma_actual = gamma_vals[idx]
        axes[0].plot(eps_vals, mu_xy_grid[idx, :], 
                    marker='o', label=f'γ = {gamma_actual:.2f}')
    
    axes[0].set_xscale('log')
    axes[0].set_xlabel('$ε_y$ (cross-forcing strength)', fontsize=12)
    axes[0].set_ylabel('$\\tilde{μ}_{xy}$', fontsize=12)
    axes[0].set_title('Cross-Mobility vs. Cross-Forcing\n(fixed damping)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Right: μ vs γ for fixed ε_y values
    eps_samples = [10**x for x in np.linspace(np.log10(min(eps_vals)), np.log10(max(eps_vals)), n_slices)]
    for eps_sample in eps_samples:
        idx = np.argmin(np.abs(eps_vals - eps_sample))
        eps_actual = eps_vals[idx]
        axes[1].plot(gamma_vals, mu_xy_grid[:, idx], 
                    marker='o', label=f'ε_y = {eps_actual:.2f}')
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$γ$ (damping coefficient)', fontsize=12)
    axes[1].set_ylabel('$\\tilde{μ}_{xy}$', fontsize=12)
    axes[1].set_title('Cross-Mobility vs. Damping\n(fixed cross-forcing)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'mobility_slices.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def analyze_optimal_point(gamma_vals, eps_vals, mu_xy_grid, A, L, m):
    """
    Find and characterize the optimal point.
    """
    if np.all(np.isnan(mu_xy_grid)):
        print("\nNo valid data to analyze!")
        return None
    
    # Find maximum absolute value (cross-mobility can be negative)
    max_idx = np.unravel_index(np.nanargmax(np.abs(mu_xy_grid)), mu_xy_grid.shape)
    gamma_opt = gamma_vals[max_idx[0]]
    eps_opt = eps_vals[max_idx[1]]
    mu_max = mu_xy_grid[max_idx]
    zeta_opt = gamma_opt**2 * L**2 / (4 * m * A)
    
    print("\n" + "="*60)
    print("OPTIMAL POINT ANALYSIS")
    print("="*60)
    print(f"\nOptimal parameters:")
    print(f"  γ* = {gamma_opt:.3f}  (damping coefficient)")
    print(f"  ζ* = {zeta_opt:.3f}  (dimensionless damping)")
    print(f"  ε_y* = {eps_opt:.3f}  (F_y·M/(A·a))")
    print(f"  μ̃_xy_max = {mu_max:.4f}")
    print(f"\nPhysical interpretation:")
    if zeta_opt < 0.5:
        regime = "underdamped (inertial)"
    elif zeta_opt < 2:
        regime = "critically damped"
    else:
        regime = "overdamped (dissipative)"
    print(f"  Damping regime: {regime}")
    print(f"  Optimal cross-forcing: F_y·M/(A·a) = {eps_opt:.3f}")
    print("="*60 + "\n")
    
    return {'gamma_opt': gamma_opt, 'zeta_opt': zeta_opt, 'eps_opt': eps_opt, 'mu_max': mu_max}

def run_params(local_args):
    """Run single parameter set."""
    i, params = local_args[0]
    n = local_args[1]
    args = local_args[2]
    gamma = params['gamma']
    eps = params['eps']
    outdir = get_output_dir(args.study_dir, gamma, eps)
    stats_file = outdir / 'stats.json'
    
    print(f"\n[{i+1}/{n}] γ={gamma:.4f}, ε={eps:.4f}")
    
    # Check if already exists
    if args.skip_existing and stats_file.exists():
        print(f"  Skipping (stats.json exists)")
        return np.array([0, 1, 0])
            
    # Run simulation
    success = run_simulation(params, outdir, args.weave_script, args.dry_run)
    
    if success:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 0, 1])

def main():
    """Main execution."""
    args = parse_arguments()
    
    print("="*60)
    print("γ-ε_y PARAMETER STUDY")
    print("="*60)
    print(f"\nStudy directory: {args.study_dir}")
    print(f"Parameter ranges:")
    print(f"  γ: [{args.gamma_min}, {args.gamma_max}] ({args.n_gamma} points)")
    print(f"  ε_y: [{args.epsy_min}, {args.epsy_max}] ({args.n_epsy} points)")
    print(f"\nFixed parameters:")
    print(f"  α = {args.alpha:.2f} (temperature)")
    print(f"  ε_x = {args.epsx:.2f} (x-forcing)")
    print(f"Total simulations: {args.n_gamma * (args.n_epsy + 1)}")

    # Generate parameter grid
    param_list, gamma_vals, eps_vals = generate_parameter_grid(args)
    
    if not args.plot_only:
        # Run simulations
        print(f"\n{'='*60}")
        print("RUNNING SIMULATIONS")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        with Pool(args.outer_ncores) as pool:
            results = pool.map(run_params, zip(enumerate(param_list), 
                                               [len(param_list)]*len(param_list),
                                               [args]*len(param_list)))
        
        elapsed = time.time() - start_time

        completed, skipped, failed = sum(results)
        
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
    
    results = load_results(args.study_dir, gamma_vals, eps_vals)
    mu_xx_grid, mu_xy_grid, D_xx_grid, D_xy_grid, D_yy_grid = extract_mobility_grid(
        results, gamma_vals, eps_vals, args.epsx, args.L, args.M, args.A, args.alpha)
    
    # Generate plots
    plot_mobility_phase_diagram(gamma_vals, eps_vals, mu_xy_grid, args.study_dir, 
                               args.alpha, args.A, args.L, args.m)
    plot_diffusion_phase_diagrams(gamma_vals, eps_vals, D_xx_grid, D_xy_grid, D_yy_grid, 
                                  args.study_dir, args.alpha, args.A, args.L, args.m)
    plot_mobility_slices(gamma_vals, eps_vals, mu_xy_grid, args.study_dir)
    
    # Analyze optimal point
    optimal = analyze_optimal_point(gamma_vals, eps_vals, mu_xy_grid, args.A, args.L, args.m)
    
    # Save summary
    summary = {
        'gamma_vals': gamma_vals.tolist(),
        'epsy_vals': eps_vals.tolist(),
        'zeta_vals': (gamma_vals**2 * args.L**2 / (4 * args.m * args.A)).tolist(),
        'mu_xx_grid': mu_xx_grid.tolist(),
        'mu_xy_grid': mu_xy_grid.tolist(),
        'D_xx_grid': D_xx_grid.tolist(),
        'D_xy_grid': D_xy_grid.tolist(),
        'D_yy_grid': D_yy_grid.tolist(),
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

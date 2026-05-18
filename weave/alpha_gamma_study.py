#!/usr/bin/env python3
"""
Driver script for α-γ parameter study of weave potential.

Tests Kramers theory predictions across damping regimes:
- Low γ (underdamped): mobility asymptotes (γ-independent prefactor)
- Moderate γ: transition regime
- High γ (overdamped): mobility ~ 1/γ

The optimal temperature α* should be approximately constant across γ
(if Kramers theory is correct).
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os
import argparse
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
from helpers import T_star_approx

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run α-γ parameter study for weave simulations', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Study parameters
    parser.add_argument('--alpha_min', type=float, default=0.1,
                       help='Minimum α = A/kBT')
    parser.add_argument('--alpha_max', type=float, default=10.0,
                       help='Maximum α = A/kBT')
    parser.add_argument('--n_alpha', type=int, default=15,
                       help='Number of α values')
    
    parser.add_argument('--gamma_min', type=float, default=0.05,
                       help='Minimum γ (damping coefficient)')
    parser.add_argument('--gamma_max', type=float, default=50.0,
                       help='Maximum γ (damping coefficient)')
    parser.add_argument('--n_gamma', type=int, default=15,
                       help='Number of γ values')
    
    # Fixed parameters
    parser.add_argument('--epsx', type=float, default=0.5,
                       help='ε_x = F_x·L/(A·a) (fixed forcing)')
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
    parser.add_argument('--nsteps', type=int, default=200000,
                       help='Number of steps per trajectory')
    parser.add_argument('--ntrajs', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--outfreq', type=int, default=1,
                       help='Number of iterations per sample')
    parser.add_argument('--ncores', type=int, default=None,
                       help='Number of cores for parallelization')
    parser.add_argument('--outer_ncores', type=int, default=None,
                       help='Number of cores for outer parallelization')
    
    # Directory management
    parser.add_argument('--study_dir', type=str, default='alpha_gamma_study',
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
    parser.add_argument('--log_contours', default=False, action='store_true',
                       help='Use log scale for contour levels')
    
    return parser.parse_args()

def generate_parameter_grid(args):
    """
    Generate (α, γ) parameter grid.
    """
    Fpx = args.epsx * args.A / args.L
    
    alpha_vals = np.logspace(np.log10(args.alpha_min), 
                             np.log10(args.alpha_max), 
                             args.n_alpha)
    gamma_vals = np.logspace(np.log10(args.gamma_min), 
                             np.log10(args.gamma_max), 
                             args.n_gamma)
    
    param_list = []
    for alpha in alpha_vals:
        kT = args.A / alpha
        for gamma in gamma_vals:
            zeta = gamma**2 * args.L**2 / (4 * args.m * args.A)
            
            params = {
                'alpha': alpha,
                'gamma': gamma,
                'zeta': zeta,
                'kT': kT,
                'Fpx': Fpx,
                'Fpy': 0.0,
                'A': args.A,
                'a': args.a,
                'L': args.L,
                'M': args.M,
                'm': args.m,
                'dt': args.dt,
                'nsteps': args.nsteps,
                'ntrajs': args.ntrajs,
                'outfreq': args.outfreq,
                'ncores': args.ncores if args.ncores else '',
                'do_subplots': args.do_subplots
            }
            
            param_list.append(params)
    
    return param_list, alpha_vals, gamma_vals


def get_output_dir(study_dir, alpha, gamma):
    """Generate output directory name."""
    dirname = f"alpha_{alpha:.4f}_gamma_{gamma:.4f}"
    return Path(study_dir) / dirname


def run_simulation(params, outdir, weave_script, dry_run=False):
    """Run a single weave.py simulation via command line."""
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
    
    cmd_str = ' '.join(cmd)
    print(f"\nRunning: {cmd_str}")
    
    if dry_run:
        print("  (dry run - not executing)")
        return True
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed with error:\n{e.stderr}")
        return False


def load_results(study_dir, alpha_vals, gamma_vals):
    """Load all stats.json files."""
    results = {}
    
    for alpha in alpha_vals:
        results[alpha] = {}
        for gamma in gamma_vals:
            outdir = get_output_dir(study_dir, alpha, gamma)
            stats_file = outdir / 'stats.json'
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                results[alpha][gamma] = stats
            else:
                results[alpha][gamma] = None
                print(f"Warning: No results for α={alpha:.4f}, γ={gamma:.4f}")
    
    return results


def extract_mobility_grid(results, alpha_vals, gamma_vals, epsx, L, A, a):
    """Extract dimensionless mobility from results."""
    n_alpha = len(alpha_vals)
    n_gamma = len(gamma_vals)
    
    mu_xx_grid = np.full((n_alpha, n_gamma), np.nan)
    D_xx_grid = np.full((n_alpha, n_gamma), np.nan)

    for i, alpha in enumerate(alpha_vals):
        for j, gamma in enumerate(gamma_vals):
            if results[alpha][gamma] is not None:
                stats = results[alpha][gamma]
                kT = A / alpha
                Fpx = epsx * A * a / L
                
                if 'D_xx' in stats:
                    D_xx_grid[i, j] = stats['D_xx'] * gamma / kT

                if 'xf' in stats and 'tf' in stats:
                    mu_xx_grid[i, j] = stats['xf'] / (stats['tf'] * Fpx) * gamma
    
    return mu_xx_grid, D_xx_grid


def plot_phase_diagram(alpha_vals, gamma_vals, mu_xx_grid, study_dir, epsx, 
                       log_contours=False, alpha_star_approx=1):
    """Create main phase diagram plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    Gamma, Alpha = np.meshgrid(gamma_vals, alpha_vals)
    
    # Choose contour levels
    if log_contours:
        # Use log levels - filter out non-positive values
        valid_data = mu_xx_grid[mu_xx_grid > 0]
        if len(valid_data) > 0:
            vmin = np.nanmin(valid_data)
            vmax = np.nanmax(valid_data)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
            from matplotlib.colors import LogNorm
            contour = ax.contourf(Gamma, Alpha, mu_xx_grid, 
                                  levels=levels, cmap='viridis', norm=LogNorm())
            contour_lines = ax.contour(Gamma, Alpha, mu_xx_grid, 
                                       levels=levels[::2], colors='white', 
                                       linewidths=0.5, alpha=0.5)
        else:
            contour = ax.contourf(Gamma, Alpha, mu_xx_grid, levels=20, cmap='viridis')
    else:
        levels = 20
        contour = ax.contourf(Gamma, Alpha, mu_xx_grid, 
                              levels=levels, cmap='viridis')
        contour_lines = ax.contour(Gamma, Alpha, mu_xx_grid, 
                                   levels=10, colors='white', 
                                   linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Find optimal α for each γ
    optimal_alphas = []
    for j in range(len(gamma_vals)):
        col = mu_xx_grid[:, j]
        if not np.all(np.isnan(col)):
            opt_idx = np.nanargmax(col)
            optimal_alphas.append(alpha_vals[opt_idx])
        else:
            optimal_alphas.append(np.nan)
    
    # Plot ridge of optimal α
    ax.plot(gamma_vals, optimal_alphas, 'r-', linewidth=2, 
            label='Optimal α(γ)', alpha=0.8)
    ax.plot(gamma_vals, optimal_alphas, 'r*', markersize=10, 
            markeredgecolor='white', markeredgewidth=1)
    
    # Theoretical prediction (if simple form known)
    # For weak forcing: α* = 1
    ax.axhline(y=alpha_star_approx, color='cyan', linestyle='--', linewidth=2, 
              label='Kramers prediction', alpha=0.8)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\gamma$ [damping coefficient]', fontsize=14)
    ax.set_ylabel('$\\alpha = A / (k_B T)$ [inverse temperature]', fontsize=14)
    ax.set_title(f'Mobility $\\tilde{{\\mu}}_{{xx}}(\\alpha, \\gamma)$\n' +
                 f'Fixed: $\\varepsilon_x = {epsx:.2f}$', 
                 fontsize=16, fontweight='bold')
    
    cbar_label = 'log scale' if log_contours else 'linear'
    cbar = plt.colorbar(contour, ax=ax, label=f'$\\tilde{{\\mu}}_{{xx}}$ ({cbar_label})')
    
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    suffix = '_log' if log_contours else ''
    output_path = study_path / f'mobility_phase_diagram{suffix}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    return fig


def plot_mobility_vs_gamma(alpha_vals, gamma_vals, mu_xx_grid, study_dir, n_slices=5):
    """
    Plot μ vs γ for fixed α values.
    Should show 3 regimes: underdamped plateau, transition, overdamped 1/γ decay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Linear plot
    alpha_samples = [10**x for x in np.linspace(np.log10(min(alpha_vals)), 
                                                 np.log10(max(alpha_vals)), n_slices)]
    
    for alpha_sample in alpha_samples:
        idx = np.argmin(np.abs(alpha_vals - alpha_sample))
        alpha_actual = alpha_vals[idx]
        axes[0].plot(gamma_vals, mu_xx_grid[idx, :], 
                    marker='o', label=f'α = {alpha_actual:.2f}')
    
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('$\\gamma$ (damping)', fontsize=12)
    axes[0].set_ylabel('$\\tilde{\\mu}_{xx}$', fontsize=12)
    axes[0].set_title('Mobility vs. Damping\n(should asymptote at low γ, decay as 1/γ at high γ)', 
                     fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Add reference 1/γ line
    gamma_ref = np.array([gamma_vals[0], gamma_vals[-1]])
    mu_ref = 1.0 / gamma_ref  # Arbitrary normalization
    # Find a good normalization
    high_gamma_data = mu_xx_grid[:, -3:]  # Last 3 columns
    if not np.all(np.isnan(high_gamma_data)):
        norm_factor = np.nanmean(high_gamma_data * gamma_vals[-3:])
        axes[0].plot(gamma_ref, norm_factor / gamma_ref, 'k--', 
                    linewidth=2, alpha=0.5, label='∝ 1/γ')
    
    # Right: Mobility scaled by γ (should asymptote at high γ if Kramers is right)
    for alpha_sample in alpha_samples:
        idx = np.argmin(np.abs(alpha_vals - alpha_sample))
        alpha_actual = alpha_vals[idx]
        axes[1].plot(gamma_vals, mu_xx_grid[idx, :] * gamma_vals, 
                    marker='o', label=f'α = {alpha_actual:.2f}')
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$\\gamma$ (damping)', fontsize=12)
    axes[1].set_ylabel('$\\tilde{\\mu}_{xx} \\cdot \\gamma$', fontsize=12)
    axes[1].set_title('Scaled Mobility μ̃·γ\n(asymptotes at large γ → overdamped Kramers)', 
                     fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'mobility_vs_gamma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def plot_mobility_vs_alpha(alpha_vals, gamma_vals, mu_xx_grid, study_dir, n_slices=5):
    """
    Plot μ vs α for fixed γ values.
    All curves should peak at approximately the same α* (if Kramers is right).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    gamma_samples = [10**x for x in np.linspace(np.log10(min(gamma_vals)), 
                                                 np.log10(max(gamma_vals)), n_slices)]
    
    for gamma_sample in gamma_samples:
        idx = np.argmin(np.abs(gamma_vals - gamma_sample))
        gamma_actual = gamma_vals[idx]
        # Normalize each curve by its maximum to compare shapes
        curve = mu_xx_grid[:, idx]
        if not np.all(np.isnan(curve)):
            ax.plot(alpha_vals, curve, marker='o', 
                   label=f'γ = {gamma_actual:.2f}')
    
    ax.set_xscale('log')
    ax.set_xlabel('$\\alpha = A / (k_B T)$', fontsize=12)
    ax.set_ylabel('$\\tilde{\\mu}_{xx}$', fontsize=12)
    ax.set_title('Mobility vs. Inverse Temperature\n(peaks should occur at similar α* if Kramers theory holds)', 
                fontweight='bold')
    
    # Add vertical line at predicted optimum
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, 
              label='Kramers prediction: α* = 1', alpha=0.7)
    
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'mobility_vs_alpha.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def plot_optimal_alpha_vs_gamma(alpha_vals, gamma_vals, mu_xx_grid, study_dir):
    """
    Plot α* vs γ - should be approximately constant if Kramers theory is right.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimal_alphas = []
    optimal_mu = []
    
    for j in range(len(gamma_vals)):
        col = mu_xx_grid[:, j]
        if not np.all(np.isnan(col)):
            opt_idx = np.nanargmax(col)
            optimal_alphas.append(alpha_vals[opt_idx])
            optimal_mu.append(col[opt_idx])
        else:
            optimal_alphas.append(np.nan)
            optimal_mu.append(np.nan)
    
    optimal_alphas = np.array(optimal_alphas)
    optimal_mu = np.array(optimal_mu)
    
    # Plot
    ax.plot(gamma_vals, optimal_alphas, 'bo-', linewidth=2, markersize=10,
           label='Simulation: α*(γ)')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
              label='Kramers prediction: α* = 1')
    
    ax.set_xscale('log')
    ax.set_xlabel('$\\gamma$ (damping)', fontsize=12)
    ax.set_ylabel('Optimal $\\alpha^*$', fontsize=12)
    ax.set_title('Optimal Temperature vs. Damping\n(should be γ-independent if Kramers theory holds)', 
                fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Print values
    print("\nOptimal α vs γ:")
    print(f"{'γ':>10} {'α*':>10} {'μ̃_max':>12}")
    for g, a, m in zip(gamma_vals, optimal_alphas, optimal_mu):
        print(f"{g:10.4f} {a:10.4f} {m:12.6f}")
    
    plt.tight_layout()
    
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    output_path = study_path / 'optimal_alpha_vs_gamma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


def run_params(local_args):
    """Helper for parallel execution."""
    i, params = local_args[0]
    n = local_args[1]
    args = local_args[2]
    alpha = params['alpha']
    gamma = params['gamma']
    outdir = get_output_dir(args.study_dir, alpha, gamma)
    stats_file = outdir / 'stats.json'
    
    print(f"\n[{i+1}/{n}] α={alpha:.4f}, γ={gamma:.4f}")
    
    if args.skip_existing and stats_file.exists():
        print(f"  Skipping (stats.json exists)")
        return np.array([0, 1, 0])
            
    success = run_simulation(params, outdir, args.weave_script, args.dry_run)
    
    if success:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 0, 1])


def main():
    """Main execution."""
    args = parse_arguments()
    
    print("="*60)
    print("α-γ PARAMETER STUDY")
    print("="*60)
    print(f"\nStudy directory: {args.study_dir}")
    print(f"\nFixed parameters:")
    print(f"  ε_x = {args.epsx:.2f} (x-forcing)")
    print(f"\nParameter ranges:")
    print(f"  α: [{args.alpha_min}, {args.alpha_max}] ({args.n_alpha} points)")
    print(f"  γ: [{args.gamma_min}, {args.gamma_max}] ({args.n_gamma} points)")
    print(f"Total simulations: {args.n_alpha * args.n_gamma}")

    param_list, alpha_vals, gamma_vals = generate_parameter_grid(args)
    
    if not args.plot_only:
        print(f"\n{'='*60}")
        print("RUNNING SIMULATIONS")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        with Pool(args.outer_ncores) as pool:
            results = pool.map(run_params, zip(enumerate(param_list), 
                                               [len(param_list)]*len(param_list),
                                               [args]*len(param_list),
                                              ))
        
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
    
    # Load and analyze
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS")
    print(f"{'='*60}\n")
    
    results = load_results(args.study_dir, alpha_vals, gamma_vals)
    mu_xx_grid, D_xx_grid = extract_mobility_grid(
        results, alpha_vals, gamma_vals, args.epsx, args.L, args.A, args.a)
    
    # Generate plots
    # eps_x = Fpx * L / (A*a) # Tilting parameter
    F = args.epsx * args.A*args.a / args.L
    alpha_star = args.A / (T_star_approx(args.A, args.a, F, args.L))
    plot_phase_diagram(alpha_vals, gamma_vals, mu_xx_grid, args.study_dir, 
                       args.epsx, log_contours=False, 
                       alpha_star_approx=alpha_star)
    plot_phase_diagram(alpha_vals, gamma_vals, mu_xx_grid, args.study_dir, 
                       args.epsx, log_contours=True,
                       alpha_star_approx=alpha_star)
    plot_mobility_vs_gamma(alpha_vals, gamma_vals, mu_xx_grid, args.study_dir)
    plot_mobility_vs_alpha(alpha_vals, gamma_vals, mu_xx_grid, args.study_dir)
    plot_optimal_alpha_vs_gamma(alpha_vals, gamma_vals, mu_xx_grid, args.study_dir)
    
    # Save summary
    summary = {
        'alpha_vals': alpha_vals.tolist(),
        'gamma_vals': gamma_vals.tolist(),
        'mu_xx_grid': mu_xx_grid.tolist(),
        'D_xx_grid': D_xx_grid.tolist(),
        'parameters': vars(args)
    }
    
    summary_file = Path(args.study_dir) / 'study_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved study summary: {summary_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

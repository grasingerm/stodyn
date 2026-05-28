#!/usr/bin/env python3
"""
Driver script for Sonication parameter study (Acoustic Pressure vs Raw Frequency).

Tests the effects of travelling and standing acoustic waves on particle mobility
across different raw frequencies and amplitudes to correlate with system resonances.
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import argparse
from pathlib import Path
import time
from multiprocessing import Pool

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Sonication parameter study for weave simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Sonication sweep parameters
    parser.add_argument('--dP_min', type=float, default=0.0,
                       help='Minimum Acoustic Pressure (dP)')
    parser.add_argument('--dP_max', type=float, default=4.0,
                       help='Maximum Acoustic Pressure (dP)')
    parser.add_argument('--n_dP', type=int, default=15,
                       help='Number of dP values (linear scale)')

    parser.add_argument('--omega_min', type=float, default=0.1,
                       help='Minimum Angular Frequency (omega)')
    parser.add_argument('--omega_max', type=float, default=50.0,
                       help='Maximum Angular Frequency (omega)')
    parser.add_argument('--n_omega', type=int, default=15,
                       help='Number of omega values (log scale)')

    parser.add_argument('--wave_type', type=str, default='traveling',
                       choices=['traveling', 'standing'],
                       help='Type of sonication wave')
    parser.add_argument('--theta', type=float, default=0.0,
                       help='Direction of the sonication wave in degrees')
    parser.add_argument('--wavelength', type=float, default=1.0,
                       help='Spatial wavelength of the sonication wave')

    # Fixed physical parameters
    parser.add_argument('--kT', type=float, default=1.0,
                       help='Thermal energy (kT)')
    parser.add_argument('--Fpx', type=float, default=1.0,
                       help='Static force in x-direction')
    parser.add_argument('--Fpy', type=float, default=0.0,
                       help='Static force in y-direction')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Fixed damping coefficient')
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
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Time step')
    parser.add_argument('--nsteps', type=int, default=20000,
                       help='Number of steps per trajectory')
    parser.add_argument('--ntrajs', type=int, default=50,
                       help='Number of trajectories')
    parser.add_argument('--outfreq', type=int, default=1,
                       help='Number of iterations per sample')
    parser.add_argument('--ncores', type=int, default=None,
                       help='Number of cores for parallelization')
    parser.add_argument('--outer_ncores', type=int, default=None,
                       help='Number of cores for outer parallelization')

    # Directory management
    parser.add_argument('--study_dir', type=str, default='sonication_study',
                       help='Base directory for study results')
    parser.add_argument('--weave_script', type=str, default='./weave_parallel_sonication.py',
                       help='Path to weave script with sonication')

    # Control
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip simulations that already have stats.json')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without running')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from existing data')
    parser.add_argument('--log_contours', default=False, action='store_true',
                       help='Use log scale for contour levels')

    return parser.parse_args()

def generate_parameter_grid(args):
    """Generate (dP, omega) parameter grid."""
    dP_vals = np.linspace(args.dP_min, args.dP_max, args.n_dP)
    omega_vals = np.logspace(np.log10(args.omega_min), np.log10(args.omega_max), args.n_omega)

    param_list = []
    for dP in dP_vals:
        for omega in omega_vals:
            params = {
                'dP': dP,
                'omega': omega,
                'wave_type': args.wave_type,
                'theta': args.theta,
                'wavelength': args.wavelength,
                'kT': args.kT,
                'gamma': args.gamma,
                'Fpx': args.Fpx,
                'Fpy': args.Fpy,
                'A': args.A,
                'a': args.a,
                'L': args.L,
                'M': args.M,
                'm': args.m,
                'dt': args.dt,
                'nsteps': args.nsteps,
                'ntrajs': args.ntrajs,
                'outfreq': args.outfreq,
                'ncores': args.ncores if args.ncores else ''
            }
            param_list.append(params)

    return param_list, dP_vals, omega_vals

def get_output_dir(study_dir, dP, omega):
    """Generate output directory name."""
    dirname = f"dP_{dP:.4f}_omega_{omega:.4f}"
    return Path(study_dir) / dirname

def run_simulation(params, outdir, weave_script, dry_run=False):
    """Run a single simulation via command line."""
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
        '--dP', str(params['dP']),
        '--omega', str(params['omega']),
        '--wave_type', str(params['wave_type']),
        '--theta', str(params['theta']),
        '--wavelength', str(params['wavelength']),
        '--outdir', str(outdir)
    ]

    if params['ncores']:
        cmd.extend(['--ncores', str(params['ncores'])])

    cmd_str = ' '.join(cmd)

    if dry_run:
        print(f"  (dry run) {cmd_str}")
        return True

    outdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed with error:\n{e.stderr}")
        return False

def load_results(study_dir, dP_vals, omega_vals):
    """Load all stats.json files."""
    results = {}
    for dP in dP_vals:
        results[dP] = {}
        for omega in omega_vals:
            outdir = get_output_dir(study_dir, dP, omega)
            stats_file = outdir / 'stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    results[dP][omega] = json.load(f)
            else:
                results[dP][omega] = None
    return results

def extract_mobility_grid(results, dP_vals, omega_vals, Fpx):
    """Extract dimensionless mobility from results."""
    mu_xx_grid = np.full((len(dP_vals), len(omega_vals)), np.nan)

    for i, dP in enumerate(dP_vals):
        for j, omega in enumerate(omega_vals):
            stats = results[dP][omega]
            if stats is not None and 'xf' in stats and 'tf' in stats:
                gamma = stats.get('parameters', {}).get('gamma', 1.0)
                if stats['tf'] > 0 and Fpx != 0:
                    mu_xx_grid[i, j] = (stats['xf'] / stats['tf']) / Fpx * gamma

    return mu_xx_grid

def plot_phase_diagram(dP_vals, omega_vals, mu_xx_grid, study_dir, wave_type, log_contours=False):
    """Create main phase diagram plot targeting raw physical frequency."""
    fig, ax = plt.subplots(figsize=(10, 8))
    Omega_mesh, DP = np.meshgrid(omega_vals, dP_vals)

    if log_contours:
        valid_data = mu_xx_grid[mu_xx_grid > 0]
        if len(valid_data) > 0:
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
            from matplotlib.colors import LogNorm
            contour = ax.contourf(Omega_mesh, DP, mu_xx_grid, levels=levels, cmap='viridis', norm=LogNorm())
            contour_lines = ax.contour(Omega_mesh, DP, mu_xx_grid, levels=levels[::2], colors='white', linewidths=0.5, alpha=0.5)
        else:
            contour = ax.contourf(Omega_mesh, DP, mu_xx_grid, levels=20, cmap='viridis')
    else:
        contour = ax.contourf(Omega_mesh, DP, mu_xx_grid, levels=20, cmap='viridis')
        contour_lines = ax.contour(Omega_mesh, DP, mu_xx_grid, levels=10, colors='white', linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')


    ax.set_xscale('log')
    ax.set_xlabel(r'Angular Frequency ($\omega$)', fontsize=14)
    ax.set_ylabel(r'Acoustic Pressure ($\Delta P$)', fontsize=14)

    title_str = 'Standing Wave' if wave_type == 'standing' else 'Traveling Wave'
    ax.set_title(f'{title_str} Sonication Phase Diagram\nMobility $\\tilde{{\\mu}}_{{xx}}$', fontsize=16, fontweight='bold')

    cbar_label = 'log scale' if log_contours else 'linear'
    plt.colorbar(contour, ax=ax, label=f'$\\tilde{{\\mu}}_{{xx}}$ ({cbar_label})')

    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    suffix = '_log' if log_contours else ''
    output_path = Path(study_dir) / f'sonication_phase_diagram{suffix}.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mobility_vs_frequency(dP_vals, omega_vals, mu_xx_grid, study_dir, n_slices=5):
    """Plot mu vs omega for fixed dP values."""
    fig, ax = plt.subplots(figsize=(10, 7))

    indices = np.linspace(0, len(dP_vals)-1, n_slices, dtype=int)

    for idx in indices:
        dP_actual = dP_vals[idx]
        ax.plot(omega_vals, mu_xx_grid[idx, :], marker='o', label=f'$\\Delta P$ = {dP_actual:.2f}')

    ax.set_xscale('log')
    ax.set_xlabel(r'Angular Frequency ($\omega$)', fontsize=12)
    ax.set_ylabel(r'Mobility $\tilde{\mu}_{xx}$', fontsize=12)
    ax.set_title('Mobility vs. Frequency at Fixed Acoustic Pressures', fontweight='bold')
    ax.legend(title="Acoustic Pressure")
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = Path(study_dir) / 'mobility_vs_frequency.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_params(local_args):
    """Helper for parallel execution."""
    i, params, n, args = local_args
    dP, omega = params['dP'], params['omega']
    outdir = get_output_dir(args.study_dir, dP, omega)
    stats_file = outdir / 'stats.json'

    print(f"[{i+1}/{n}] dP={dP:.4f}, omega={omega:.4f}")

    if args.skip_existing and stats_file.exists():
        return np.array([0, 1, 0])

    success = run_simulation(params, outdir, args.weave_script, args.dry_run)
    return np.array([1, 0, 0]) if success else np.array([0, 0, 1])

def main():
    args = parse_arguments()

    print("="*60)
    print("SONICATION PARAMETER STUDY (dP vs omega)")
    print("="*60)
    print(f"\nStudy directory: {args.study_dir}")
    print(f"Wave Type: {args.wave_type.upper()}")
    print(f"\nFixed Physics Parameters:")
    print(f"  kT={args.kT}, Fpx={args.Fpx}, A={args.A}, gamma={args.gamma}")
    print(f"\nParameter ranges:")
    print(f"  dP:    [{args.dP_min}, {args.dP_max}] ({args.n_dP} points)")
    print(f"  omega: [{args.omega_min}, {args.omega_max}] ({args.n_omega} points)")
    print(f"Total simulations: {args.n_dP * args.n_omega}")

    param_list, dP_vals, omega_vals = generate_parameter_grid(args)

    if not args.plot_only:
        print(f"\n{'='*60}\nRUNNING SIMULATIONS\n{'='*60}\n")
        start_time = time.time()

        pool_args = [(i, p, len(param_list), args) for i, p in enumerate(param_list)]

        with Pool(args.outer_ncores) as pool:
            results = pool.map(run_params, pool_args)

        elapsed = time.time() - start_time
        completed, skipped, failed = sum(results) if len(results) > 0 else (0, 0, 0)

        print(f"\n{'='*60}\nSIMULATION SUMMARY\n{'='*60}")
        print(f"Completed: {completed}\nSkipped:   {skipped}\nFailed:    {failed}")
        print(f"Time:      {elapsed/60:.1f} minutes\n{'='*60}\n")

    print(f"\n{'='*60}\nANALYZING RESULTS\n{'='*60}\n")
    results = load_results(args.study_dir, dP_vals, omega_vals)
    mu_xx_grid = extract_mobility_grid(results, dP_vals, omega_vals, args.Fpx)

    plot_phase_diagram(dP_vals, omega_vals, mu_xx_grid, args.study_dir, args.wave_type, log_contours=False)
    plot_phase_diagram(dP_vals, omega_vals, mu_xx_grid, args.study_dir, args.wave_type, log_contours=True)
    plot_mobility_vs_frequency(dP_vals, omega_vals, mu_xx_grid, args.study_dir)

    summary = {
        'dP_vals': dP_vals.tolist(),
        'omega_vals': omega_vals.tolist(),
        'mu_xx_grid': mu_xx_grid.tolist(),
        'parameters': vars(args)
    }

    summary_file = Path(args.study_dir) / 'study_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved study summary: {summary_file}")
    print("\nDone!")

if __name__ == '__main__':
    main()

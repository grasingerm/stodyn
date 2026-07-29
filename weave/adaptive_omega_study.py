#!/usr/bin/env python3
"""
Adaptive Peak Frequency (Omega) Search for Weave Simulations.

Sweeps a target physical parameter (e.g., A, gamma, dP) on a LOG scale.
Iteratively searches for the angular frequency (omega) that maximizes pure mobility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import subprocess
import json
import argparse
from pathlib import Path
import time
from multiprocessing import Pool

def parse_arguments():
    parser = argparse.ArgumentParser(description='Adaptive Omega Search')
    
    # Adaptive Search Parameters
    parser.add_argument('--target_param', type=str, required=True, 
                        help='Parameter to sweep (e.g., gamma, A, dP, Fpx)')
    parser.add_argument('--param_min', type=float, required=True, help='Min value of target param (>0)')
    parser.add_argument('--param_max', type=float, required=True, help='Max value of target param')
    parser.add_argument('--n_param', type=int, default=8, help='Number of points for target param')
    
    parser.add_argument('--omega_min_init', type=float, default=0.1, help='Initial min omega')
    parser.add_argument('--omega_max_init', type=float, default=50.0, help='Initial max omega')
    parser.add_argument('--n_omega_per_iter', type=int, default=6, help='Omega points per zoom iteration')
    parser.add_argument('--n_iters', type=int, default=3, help='Number of zoom iterations')
    parser.add_argument('--zoom_factor', type=float, default=0.8, help='Fraction of neighbor gap to keep for next zoom')

    # Fixed physical parameters (defaults)
    parser.add_argument('--wave_type', type=str, default='traveling')
    parser.add_argument('--theta', type=float, default=0.0)
    parser.add_argument('--wavelength', type=float, default=1.0)
    parser.add_argument('--dP', type=float, default=1.0)
    parser.add_argument('--kT', type=float, default=1.0)
    parser.add_argument('--Fpx', type=float, default=1.0)
    parser.add_argument('--Fpy', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--A', type=float, default=1.0)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--L', type=float, default=1.0)
    parser.add_argument('--M', type=float, default=1.0)
    parser.add_argument('--m', type=float, default=1.0)

    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--nsteps', type=int, default=20000)
    parser.add_argument('--ntrajs', type=int, default=50)
    parser.add_argument('--outfreq', type=int, default=1)
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--outer_ncores', type=int, default=48, help='Cores for the omega sweep pool')
    
    parser.add_argument('--study_dir', type=str, default='adaptive_study')
    parser.add_argument('--weave_script', type=str, default='./weave_parallel_sonication.py')
    parser.add_argument('--skip_existing', action='store_true')

    return parser.parse_args()

def get_output_dir(study_dir, param_val, omega):
    dirname = f"val_{param_val:.4f}_omega_{omega:.4f}"
    return Path(study_dir) / dirname

def run_simulation_task(local_args):
    p_val, w, params, outdir, weave_script, skip_existing = local_args
    stats_file = outdir / 'stats.json'
    
    if skip_existing and stats_file.exists():
        return p_val, w, True
        
    cmd = [
        'python', weave_script,
        '--m', str(params['m']), '--gamma', str(params['gamma']), '--kT', str(params['kT']),
        '--Fpx', str(params['Fpx']), '--Fpy', str(params['Fpy']),
        '--A', str(params['A']), '--a', str(params['a']),
        '--L', str(params['L']), '--M', str(params['M']),
        '--dt', str(params['dt']), '--nsteps', str(params['nsteps']),
        '--ntrajs', str(params['ntrajs']), '--outfreq', str(params['outfreq']),
        '--dP', str(params['dP']), '--omega', str(params['omega']),
        '--wave_type', str(params['wave_type']), '--theta', str(params['theta']),
        '--wavelength', str(params['wavelength']), '--outdir', str(outdir)
    ]
    if params['ncores']:
        cmd.extend(['--ncores', str(params['ncores'])])
        
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return p_val, w, True
    except subprocess.CalledProcessError as e:
        print(f"Error at {params['target_param']}={p_val}, omega={w}:\n{e.stderr}")
        return p_val, w, False

def extract_mobility(outdir, Fpx):
    stats_file = outdir / 'stats.json'
    if not stats_file.exists():
        return np.nan
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    if stats['tf'] > 0:
        safe_Fpx = Fpx if Fpx != 0 else 1.0 
        # Pure mobility = Velocity / Force
        mu_xx = (stats['xf'] / stats['tf']) / safe_Fpx
        return mu_xx
    return np.nan

def main():
    args = parse_arguments()
    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter values on a LOG scale
    param_vals = np.logspace(np.log10(args.param_min), np.log10(args.param_max), args.n_param)
    base_params = vars(args).copy()
    
    print("="*60)
    print(f"ADAPTIVE OMEGA SEARCH: Sweeping '{args.target_param}' (LOG SCALE)")
    print(f"Fully Parallelized - Gathering tasks for {args.n_param} values simultaneously")
    print(f"Zoom tightness factor: {args.zoom_factor}")
    print("="*60)
    
    search_states = {
        p_val: {
            'o_min': args.omega_min_init,
            'o_max': args.omega_max_init,
            'evaluated': {}, 
            'current_batch_w': [] 
        } for p_val in param_vals
    }
    
    start_time = time.time()
    
    # 1. Outer Loop: Iterations
    for iteration in range(args.n_iters):
        print(f"\n--- Iteration {iteration+1}/{args.n_iters} ---")
        pool_args = []
        
        # Gather tasks from ALL parameter values
        for p_val in param_vals:
            state = search_states[p_val]
            o_min, o_max = state['o_min'], state['o_max']
            
            if o_max / max(o_min, 1e-5) > 3:
                omegas = np.logspace(np.log10(max(o_min, 1e-5)), np.log10(o_max), args.n_omega_per_iter)
            else:
                omegas = np.linspace(o_min, o_max, args.n_omega_per_iter)
            
            omegas_to_run = [w for w in omegas if w not in state['evaluated']]
            state['current_batch_w'] = omegas
            
            for w in omegas_to_run:
                sim_params = base_params.copy()
                sim_params[args.target_param] = p_val
                sim_params['omega'] = w
                outdir = get_output_dir(study_dir, p_val, w)
                
                pool_args.append((p_val, w, sim_params, outdir, args.weave_script, args.skip_existing))
                
        print(f"Gathered {len(pool_args)} tasks. Launching pool with {args.outer_ncores} workers...")
        
        # 2. Execute massive batch in parallel
        if pool_args:
            with Pool(args.outer_ncores) as pool:
                _ = pool.map(run_simulation_task, pool_args)
                
        # 3. Process results and calculate bounds for next iteration
        for p_val in param_vals:
            state = search_states[p_val]
            batch_w = state['current_batch_w']
            batch_mu = []
            
            curr_Fpx = p_val if args.target_param == 'Fpx' else base_params['Fpx']
            
            for w in batch_w:
                outdir = get_output_dir(study_dir, p_val, w)
                mu = extract_mobility(outdir, curr_Fpx)
                state['evaluated'][w] = mu
                batch_mu.append(mu)
                
            valid_mu = np.array(batch_mu)
            valid_mu = np.where(np.isnan(valid_mu), -np.inf, valid_mu)
            
            if np.all(valid_mu == -np.inf):
                continue
                
            peak_idx = np.argmax(valid_mu)
            peak_w = batch_w[peak_idx]
            
            zf = args.zoom_factor
            
            if peak_idx == 0:
                state['o_min'] = batch_w[0] / 3.0
                state['o_max'] = batch_w[0] + (batch_w[1] - batch_w[0]) * zf
            elif peak_idx == len(batch_w) - 1:
                state['o_min'] = batch_w[-1] - (batch_w[-1] - batch_w[-2]) * zf
                state['o_max'] = batch_w[-1] * 3.0
            else:
                left_gap = peak_w - batch_w[peak_idx - 1]
                right_gap = batch_w[peak_idx + 1] - peak_w
                state['o_min'] = peak_w - (left_gap * zf)
                state['o_max'] = peak_w + (right_gap * zf)

    # --- Data Compilation ---
    all_results = {}
    peak_summaries = {}
    
    for p_val in param_vals:
        state = search_states[p_val]
        sorted_evals = sorted(state['evaluated'].items())
        all_results[p_val] = {
            'omegas': [x[0] for x in sorted_evals],
            'mobilities': [x[1] for x in sorted_evals]
        }
        
        valid_evals = {k: v for k, v in state['evaluated'].items() if not np.isnan(v)}
        if valid_evals:
            best_omega = max(valid_evals, key=valid_evals.get)
            peak_summaries[p_val] = {
                'peak_omega': best_omega,
                'peak_mobility': valid_evals[best_omega]
            }

    # --- Data Saving ---
    with open(study_dir / 'all_data.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    with open(study_dir / 'peak_summary.json', 'w') as f:
        json.dump(peak_summaries, f, indent=2)

    # --- Plotting ---
    print("\nGenerating plots...")
    
    # 1. Colormapped Mobility vs Frequency Plot
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Define Colormap and Normalization (LogNorm since params are log spaced)
    cmap = cm.viridis
    norm = mcolors.LogNorm(vmin=args.param_min, vmax=args.param_max)
    
    for p_val in sorted(all_results.keys()):
        data = all_results[p_val]
        w = np.array(data['omegas'])
        mu = np.array(data['mobilities'])
        color = cmap(norm(p_val))
        
        ax1.plot(w, mu, marker='o', markersize=4, linestyle='-', alpha=0.7, color=color)
        
    ax1.set_xscale('log')
    ax1.set_xlabel(r'Angular Frequency ($\omega$)', fontsize=12)
    ax1.set_ylabel(r'Pure Mobility $\mu_{xx}$', fontsize=12)
    ax1.set_title(f'Mobility vs Frequency (Sweeping {args.target_param})', fontweight='bold')
    
    # Add a continuous Colorbar instead of a massive legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label(f'{args.target_param} value', fontsize=12)
    
    ax1.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig1.savefig(study_dir / f'mobility_vs_freq_{args.target_param}.pdf', dpi=300)
    plt.close(fig1)

    # 2. Peak Frequency vs Target Parameter
    if peak_summaries:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        p_keys = sorted(list(peak_summaries.keys()))
        peak_ws = [peak_summaries[p]['peak_omega'] for p in p_keys]
        
        ax2.plot(p_keys, peak_ws, marker='s', color='firebrick', linewidth=2, markersize=8)
        ax2.set_xscale('log') # Set x-axis to log to match parameter generation
        ax2.set_xlabel(f'Parameter: {args.target_param} (log scale)', fontsize=12)
        ax2.set_ylabel(r'Peak Resonant Frequency ($\omega_{peak}$)', fontsize=12)
        ax2.set_title(f'Peak Frequency Dependency on {args.target_param}', fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        fig2.savefig(study_dir / f'peak_freq_vs_{args.target_param}.pdf', dpi=300)
        plt.close(fig2)

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed/60:.1f} minutes.")

if __name__ == '__main__':
    main()

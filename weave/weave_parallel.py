import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
from multiprocessing import Pool, cpu_count
import json
from pathlib import Path
import copy
import time
import pandas as pd
from pprint import pprint
from scipy.integrate import simpson

parser = argparse.ArgumentParser(description='Simulation of pressure-driven Brownian motion through a 2D weave', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--m', type=float, default=1.0, help='mass of the particle')
parser.add_argument('--gamma', type=float, default=1.0, help='drag coefficient')
parser.add_argument('--kT', type=float, default=1.0, help='temperature')
parser.add_argument('--Fpx', type=float, default=1.0, help='force from pressure gradient in x direction')
parser.add_argument('--Fpy', type=float, default=0.0, help='force from pressure gradient in y direction')
parser.add_argument('--A', type=float, default=1.0, help='amplitude of the "weave" potential')
parser.add_argument('--a', type=float, default=1.0, help='shape factor of the "weave" potential, greater "a" corresponds with sharper peaks and flatter wells')
parser.add_argument('--L', type=float, default=1.0, help='distance between peaks of "weave" potential in x-y direction')
parser.add_argument('--M', type=float, default=1.0, help='distance between peaks of "weave" potential in x+y direction')
parser.add_argument('--dt', type=float, default=0.001, help='time step')
parser.add_argument('--int', type=str, default='BAOAB', help='EM (Euler Murayama) | BAOAB (symmetric splitting)')
parser.add_argument('--nsteps', type=int, default=20000, help='number of steps')
parser.add_argument('--ntrajs', type=int, default=100, help='number of trajectories')
parser.add_argument('--x0_L', type=float, default=-0.5, help='initial x position in units of L, x/L')
parser.add_argument('--y0_M', type=float, default=0.0, help='initial y position in units of M, y/M')
parser.add_argument('--u0', type=float, default=0.0, help='initial x velocity')
parser.add_argument('--v0', type=float, default=0.0, help='initial y velocity')
parser.add_argument('--outfreq', type=int, default=1, help='number of iterations per sample')
parser.add_argument('--eqfrac', type=float, default=0.1, help='fraction of trajectories assumed to be equilibrated')
parser.add_argument('--max_lag', type=int, default=10, help='maximum amount of lag in steps for computing velocity autocorrelations')
parser.add_argument('--outdir', type=str, default="temp", help='output directory')
parser.add_argument('--do_plots', default=False, action="store_true", help='create plots')
parser.add_argument('--show_plots', default=False, action="store_true", help='show plots')
parser.add_argument('--save_trajs', default=False, action="store_true", help='save trajectories')
parser.add_argument('--print_escapes', default=False, action="store_true", help='print out escape events')
parser.add_argument('--seed', type=int, help='seed for random number generator')
parser.add_argument('--ncores', type=int, default=None, help='number of cores to use (default: all available)')

nargs = parser.parse_args()
args = vars(nargs)

def U(x, y, A, a, L, M):
    X = np.pi*x/L
    Y = np.pi*y/M
    return A*np.exp(a*(np.sin(X-Y) + np.sin(X+Y)))

def gradU(x, y, A, a, L, M):
    X = np.pi*x/L
    Y = np.pi*y/M
    sx, sy, cx, cy = np.sin(X), np.sin(Y), np.cos(X), np.cos(Y)
    Z = 2*np.pi * a*A * np.exp(2*a * (sx*cy))
    return Z*np.array([cx*cy/L, -sx*sy/M])
        
def plot_2d_trajectory_colored(x, y, potential_func=None, figsize=(12, 10)):
    """
    Plot a 2D trajectory with color gradient showing time progression.
    
    Parameters:
    -----------
    x, y : arrays
        Position trajectories
    potential_func : function, optional
        Function V(x, y) to plot potential energy landscape
    figsize : tuple
        Figure size
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ===== Left plot: Trajectory with color gradient =====
    
    # Create line segments for coloring
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create colors based on time (normalized 0 to 1)
    n_points = len(x)
    colors = np.linspace(0, 1, n_points)
    
    # Create LineCollection with color map
    lc = LineCollection(segments, cmap='viridis', linewidth=2)
    lc.set_array(colors)
    
    # Add to plot
    line = ax1.add_collection(lc)
    
    # Mark start and end points
    ax1.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=5, 
             markeredgecolor='black', markeredgewidth=2)
    ax1.plot(x[-1], y[-1], 'ro', markersize=12, label='End', zorder=5,
             markeredgecolor='black', markeredgewidth=2)
    
    # Add colorbar
    shrink = 0.4
    aspect = 30.0
    cbar = fig.colorbar(line, ax=ax1, label='Time (normalized)', shrink=shrink, aspect=aspect)
    
    # Set axis properties
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('y', fontsize=14)
    ax1.set_title('2D Trajectory (Color = Time)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Set axis limits with some padding
    x_margin = 0.1 * (x.max() - x.min())
    y_margin = 0.1 * (y.max() - y.min())
    ax1.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax1.set_ylim(y.min() - y_margin, y.max() + y_margin)
    
    # ===== Right plot: Potential energy landscape (if provided) =====
    
    if potential_func is not None:
        # Create mesh grid
        x_range = np.linspace(x.min() - x_margin, x.max() + x_margin, 200)
        y_range = np.linspace(y.min() - y_margin, y.max() + y_margin, 200)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        V_mesh = potential_func(X_mesh, Y_mesh)
        
        # Plot potential as contours
        levels = 20
        contour = ax2.contourf(X_mesh, Y_mesh, V_mesh, levels=levels, cmap='RdYlBu_r', alpha=0.8)
        ax2.contour(X_mesh, Y_mesh, V_mesh, levels=levels, colors='black', 
                   linewidths=0.5, alpha=0.3)
        
        # Overlay trajectory
        lc2 = LineCollection(segments, cmap='viridis', linewidth=2, alpha=0.8)
        lc2.set_array(colors)
        ax2.add_collection(lc2)
        
        # Mark start and end
        ax2.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=5,
                markeredgecolor='black', markeredgewidth=2)
        ax2.plot(x[-1], y[-1], 'ro', markersize=12, label='End', zorder=5,
                markeredgecolor='black', markeredgewidth=2)
        
        # Colorbar for potential
        cbar2 = fig.colorbar(contour, ax=ax2, label='Potential Energy V(x,y)', shrink=shrink, aspect=aspect)
        
        ax2.set_xlabel('x', fontsize=14)
        ax2.set_ylabel('y', fontsize=14)
        ax2.set_title('Trajectory on Potential Landscape', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax2.set_ylim(y.min() - y_margin, y.max() + y_margin)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    return fig
    
def identify_escape_events(x, y, t, ntrajs, nsteps, L, M):
    """
    Identify when particle hops between valleys.
    
    Returns:
    --------
    escape_events : list of dicts
        Each dict: {'time': t, 'direction': (dx, dy), 'angle': theta}
    """
    x_trajs = x.reshape(ntrajs, nsteps)
    y_trajs = y.reshape(ntrajs, nsteps)
    
    escape_events = []
    
    for traj_idx in range(ntrajs):
        # Coarse-grain positions to identify which valley particle is in
        x_traj = x_trajs[traj_idx, :]
        y_traj = y_trajs[traj_idx, :]
        
        # Map to valley indices (discretize to nearest valley)
        valley_x = np.round(x_traj / L + 0.5)
        valley_y = np.round(y_traj / M)
        #valley_x = np.round((x_traj-y_traj) / L + 0.25)
        #valley_y = np.round((x_traj+y_traj) / M + 0.25)

        # Find when valley changes (escape event!)
        prev_x = valley_x[0]
        prev_y = valley_y[0]
        for i in range(1, nsteps):
            if valley_x[i] == prev_x or valley_y[i] == prev_y:
                continue
            dx = int(valley_x[i] - prev_x)
            dy = int(valley_y[i] - prev_y)
            
            escape_events.append({
                'traj_idx': int(traj_idx),
                'time_idx': int(i),
                'time': float(t[traj_idx * nsteps + i]),
                'dx': dx,
                'dy': dy,
                'direction_vector': [dx, dy],
                'transition': [(int(prev_x), int(prev_y)), 
                               (int(valley_x[i]), int(valley_y[i]))]
            })
            
            prev_x = valley_x[i]
            prev_y = valley_y[i]
    
    return escape_events
    
def compute_escape_direction_autocorrelation(escape_events, max_lag=10):
    """
    Compute autocorrelation of escape directions.
    
    C(n) = <cos(θ_i - θ_{i+n})> where θ_i is direction of i-th escape
    
    Returns:
    --------
    C : array
        Direction autocorrelation vs lag (number of escapes)
    """
    # Group by trajectory
    events_by_traj = {}
    for event in escape_events:
        traj_idx = event['traj_idx']
        if traj_idx not in events_by_traj:
            events_by_traj[traj_idx] = []
        events_by_traj[traj_idx].append(event)
    
    C = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    for traj_idx, events in events_by_traj.items():
        n_events = len(events)
        
        for lag in range(max_lag):
            for i in range(n_events - lag):
                # Cosine similarity between directions
                u_i = np.array(events[i]['direction_vector'])
                u_j = np.array(events[i+lag]['direction_vector'])
                
                # Cosine of angle difference (measures alignment)
                #C[lag] += np.cos(angle_j - angle_i)
                C[lag] += np.dot(u_i, u_j) / 2
                counts[lag] += 1
    
    # Normalize
    C = C / counts
    
    return C

def run_single_trajectory_EM(params):
    """
    Simulate Brownian motion using the Langevin equation with Euler-Maruyama integration.
    
    The equation of motion is:
    m dv/dt = -k*x - gamma*v + xi(t)
    
    where xi(t) is white noise with <xi(t)xi(t')> = 2*gamma*kBT*delta(t-t')
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all simulation parameters
    
    Returns:
    --------
    t, x, v : arrays
        Time, position, and velocity trajectories
    """

    # Unpack parameters
    m = params['m']
    gamma = params['gamma']
    kT = params['kT']
    Fpx, Fpy = params['Fpx'], params['Fpy']
    A = params['A']
    a = params['a']
    L = params['L']
    M = params['M']
    dt = params['dt']
    nsteps = params['nsteps']
    x0 = params['x0_L'] * L
    y0 = params['y0_M'] * M
    u0 = params['u0']
    v0 = params['v0']
    traj_seed = params['traj_seed']

    # Set random seed for this trajectory (different for each trajectory)
    np.random.seed(traj_seed)

    # Initialize arrays
    t = np.zeros(nsteps)
    x = np.zeros(nsteps)
    y = np.zeros(nsteps)
    u = np.zeros(nsteps)
    v = np.zeros(nsteps)
        
    # Noise strength: sqrt(2 * gamma * kBT / mass)
    noise_strength = np.sqrt(2 * gamma * kT / m)
 
    x[0] = x0
    y[0] = y0
    u[0] = u0
    v[0] = v0
    
    # Euler-Maruyama integration
    for i in range(nsteps - 1):
        # Deterministic force
        F = np.array([
                Fpx - gamma*u[i],
                Fpy -gamma*v[i]
            ]) - gradU(x[i], y[i], A, a, L, M)
        Fx, Fy = F
        
        # Random force from standard normal distribution
        Frx = noise_strength * np.random.randn()
        Fry = noise_strength * np.random.randn()
        
        # Update velocity: dv = F/m*dt + noise*sqrt(dt)
        u[i + 1] = u[i] + Fx / m* dt + Frx * np.sqrt(dt)
        v[i + 1] = v[i] + Fy / m* dt + Fry * np.sqrt(dt)
        
        # Update position
        x[i + 1] = x[i] + u[i + 1] * dt
        y[i + 1] = y[i] + v[i + 1] * dt
        
        t[i + 1] = t[i] + dt

    incr = args['outfreq']
    return t[::incr], x[::incr], y[::incr], u[::incr], v[::incr]

def run_single_trajectory_BAOAB(params):
    """
    Simulate Brownian motion using the Langevin equation with BAOAB integration.
    
    The equation of motion is:
    m dv/dt = -k*x - gamma*v + xi(t)
    
    where xi(t) is white noise with <xi(t)xi(t')> = 2*gamma*kBT*delta(t-t')
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all simulation parameters
    
    Returns:
    --------
    t, x, v : arrays
        Time, position, and velocity trajectories
    """

    # Unpack parameters
    m = params['m']
    gamma = params['gamma']
    kT = params['kT']
    Fpx, Fpy = params['Fpx'], params['Fpy']
    Fp = np.array([Fpx, Fpy])
    a = params['a']
    L = params['L']
    M = params['M']
    dt = params['dt']
    nsteps = params['nsteps']
    x0 = params['x0_L'] * L
    y0 = params['y0_M'] * M
    u0 = params['u0']
    v0 = params['v0']
    traj_seed = params['traj_seed']

    # Set random seed for this trajectory (different for each trajectory)
    np.random.seed(traj_seed)

    # Initialize arrays
    t = np.zeros(nsteps)
    x = np.zeros(nsteps)
    y = np.zeros(nsteps)
    u = np.zeros(nsteps)
    v = np.zeros(nsteps)
        
    # Noise strength: sqrt(2 * gamma * kBT / mass)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1 - c1**2) * kT / m)
 
    x[0] = x0
    y[0] = y0
    u[0] = u0
    v[0] = v0
    
    # BAOAB integration
    for i in range(nsteps - 1):
        xvec = np.array([x[i], y[i]])
        uvec = np.array([u[i], v[i]])

        # B: Deterministic force, F(x, t)
        F = Fp - gradU(x[i], y[i], A, a, L, M)
        uvec_half = uvec + 0.5*(F / m)*dt

        # A: Half position step
        xvec_half = xvec + 0.5*uvec_half*dt
        
        # O: OU, Random force from standard normal distribution
        uvec_ou = c1*uvec_half + c2*np.random.randn(2)

        # A: second half position step
        xvec_new = xvec_half + 0.5*uvec_ou*dt

        # B: second half velocity step, F(xnew, t+dt)
        F = Fp - gradU(xvec_new[0], xvec_new[1], A, a, L, M)
        uvec_new = uvec_ou + 0.5*(F / m)*dt

        # Store results
        u[i + 1] = uvec_new[0]
        v[i + 1] = uvec_new[1]
        x[i + 1] = xvec_new[0]
        y[i + 1] = xvec_new[1]
        t[i + 1] = t[i] + dt

    incr = args['outfreq']
    return t[::incr], x[::incr], y[::incr], u[::incr], v[::incr]


def langevin_simulation_parallel(params):
    """
    Run multiple Langevin simulations in parallel.
    
    Parameters:
    -----------
    ncores : int or None
        Number of CPU cores to use. If None, uses all available cores.
    seed : int or None
        Base seed for random number generation. Each trajectory gets seed + traj_index.
    
    Returns:
    --------
    t, x, y, u, v : arrays
        Concatenated trajectories from all simulations
    """
    
    # Determine number of cores
    ntrajs = params['ntrajs']
    ncores = params['ncores']
    if ncores is None:
        ncores = cpu_count()
    ncores = min(ncores, ntrajs)  # Don't use more cores than trajectories
    
    print(f"Running {ntrajs} trajectories in parallel using {ncores} cores...")
    
    # Set base seed
    seed = params['seed']
    if seed is None:
        seed = np.random.randint(0, 2**31)
    
    # Prepare parameter dictionaries for each trajectory
    param_list = []
    for i in range(ntrajs):
        params_i = copy.copy(params)
        params_i['traj_seed'] = seed + i  # Unique seed for each trajectory
        param_list.append(params_i)

    run_single_trajectory_function = run_single_trajectory_BAOAB
    if params['int'] == 'EM':
        run_single_trajectory_function = run_single_trajectory_EM
    
    # Run simulations in parallel
    with Pool(ncores) as pool:
        results = pool.map(run_single_trajectory_function, param_list)
    
    # Concatenate results
    t_all = np.concatenate([r[0] for r in results])
    x_all = np.concatenate([r[1] for r in results])
    y_all = np.concatenate([r[2] for r in results])
    u_all = np.concatenate([r[3] for r in results])
    v_all = np.concatenate([r[4] for r in results])
    
    print(f"Parallel simulation complete!")
    
    return t_all, x_all, y_all, u_all, v_all

def ensemble_avg(z, ntrajs, nsteps):
    z_trajs = z.reshape(ntrajs, nsteps)
    z_mean = np.mean(z_trajs, axis=0)
    return z_mean

def compute_diffusion_coefficient(x, y, t, ntrajs, nsteps, eqfrac=0.5):
    """
    Compute diffusion coefficient from multiple trajectories.
    
    Returns:
    --------
    D_xx, D_yy, D_xy : floats
        Components of diffusion tensor
    """
    x_trajs = x.reshape(ntrajs, nsteps)
    y_trajs = y.reshape(ntrajs, nsteps)
    t_single = t[:nsteps]

    # Compute mean trajectory (ensemble average)
    x_mean = np.mean(x_trajs, axis=0)
    y_mean = np.mean(y_trajs, axis=0)
    
    # Compute deviations from mean
    dx = x_trajs - x_mean[np.newaxis, :]
    dy = y_trajs - y_mean[np.newaxis, :]
    
    # Compute mean squared deviations
    msd_x = np.mean(dx**2, axis=0)
    msd_y = np.mean(dy**2, axis=0)
    msd_xy = np.mean(dx * dy, axis=0)
    
    # Linear fit to extract diffusion coefficient
    # MSD = 2*D*t, so D = slope / 2
    # Use later times for fitting (equilibrated regime)
    fit_start = int(min(np.floor(nsteps*eqfrac), nsteps-1))  # Use second half
    
    # Fit MSD_x vs t
    coeffs_x = np.polyfit(t_single[fit_start:], msd_x[fit_start:], 1)
    D_xx = coeffs_x[0] / 2
    
    # Fit MSD_y vs t
    coeffs_y = np.polyfit(t_single[fit_start:], msd_y[fit_start:], 1)
    D_yy = coeffs_y[0] / 2
    
    # Fit cross-correlation
    coeffs_xy = np.polyfit(t_single[fit_start:], msd_xy[fit_start:], 1)
    D_xy = coeffs_xy[0] / 2
    
    return D_xx, D_yy, D_xy, msd_x, msd_y, t_single
    
def compute_vacf(u, v, ntrajs, nsteps, max_lag=None):
    """
    Compute velocity autocorrelation function.
    
    Parameters:
    -----------
    u, v : arrays
        Velocity components (all trajectories concatenated)
    ntrajs : int
        Number of trajectories
    nsteps : int
        Steps per trajectory
    max_lag : int or None
        Maximum lag time (in steps)
    
    Returns:
    --------
    C_uu, C_vv, C_uv, C_vu : arrays
        VACF for xx, yy, xy, and yx components
    """
    # Reshape to separate trajectories
    u_trajs = u.reshape(ntrajs, nsteps)
    v_trajs = v.reshape(ntrajs, nsteps)
    
    # Remove mean (drift)
    u_mean = ensemble_avg(u, ntrajs, nsteps)
    v_mean = ensemble_avg(v, ntrajs, nsteps)
    du = u_trajs - u_mean[np.newaxis, :]
    dv = v_trajs - v_mean[np.newaxis, :]
    
    if max_lag is None:
        max_lag = nsteps // 4  # Use first quarter for good statistics
    
    C_uu = np.zeros(max_lag)
    C_vv = np.zeros(max_lag)
    C_uv = np.zeros(max_lag)
    C_vu = np.zeros(max_lag)
    
    # Compute autocorrelation
    for lag in range(max_lag):
        # Average over all time origins and trajectories
        for t0 in range(nsteps - lag):
            C_uu[lag] += np.mean(du[:, t0] * du[:, t0 + lag])
            C_vv[lag] += np.mean(dv[:, t0] * dv[:, t0 + lag])
            C_uv[lag] += np.mean(du[:, t0] * dv[:, t0 + lag])
            C_vu[lag] += np.mean(dv[:, t0] * du[:, t0 + lag])
        
        C_uu[lag] /= (nsteps - lag)
        C_vv[lag] /= (nsteps - lag)
        C_uv[lag] /= (nsteps - lag)
        C_vu[lag] /= (nsteps - lag)
    
    return C_uu, C_vv, C_uv, C_vu

def compute_correlation_times(C_uu_norm, C_vv_norm, dt):
    return simpson(C_uu_norm, dx=dt), simpson(C_vv_norm, dx=dt)

def compute_diffusion_from_vacf(C_uu, C_vv, C_uv, C_vu, dt):
    """
    Compute diffusion coefficient from VACF using Green-Kubo.
    """
    return simpson(C_uu, dx=dt), simpson(C_vv, dx=dt), simpson(C_uv, dx=dt), simpson(C_vu, dx=dt)

def compute_mean_velocity_robust(x, y, t, ntrajs, nsteps, eqfrac):
    """
    Compute mean velocity from linear fit of position vs time.
    More robust to noise and equilibration issues.
    """
    # Reshape
    x_trajs = x.reshape(ntrajs, nsteps)
    y_trajs = y.reshape(ntrajs, nsteps)
    t_single = t[:nsteps]
    
    # Use second half (equilibrated)
    eq_start = int(np.round(nsteps*eqfrac))
    
    # Ensemble-averaged trajectory
    x_mean_traj = np.mean(x_trajs, axis=0)
    y_mean_traj = np.mean(y_trajs, axis=0)
    
    # Fit linear portion
    coeffs_x = np.polyfit(t_single[eq_start:], x_mean_traj[eq_start:], 1)
    coeffs_y = np.polyfit(t_single[eq_start:], y_mean_traj[eq_start:], 1)
    
    v_x_mean = coeffs_x[0]  # Slope
    v_y_mean = coeffs_y[0]
    
    # Uncertainty from fit
    x_fit = np.polyval(coeffs_x, t_single[eq_start:])
    y_fit = np.polyval(coeffs_y, t_single[eq_start:])
    
    residual_x = np.std(x_mean_traj[eq_start:] - x_fit)
    residual_y = np.std(y_mean_traj[eq_start:] - y_fit)
    
    return v_x_mean, v_y_mean, residual_x, residual_y

def z_eq(z, ntrajs, nsteps, eqfrac):
    eqsteps = int(np.round(eqfrac*nsteps))
    z_trajs = z.reshape(ntrajs, nsteps)[:,eqsteps:]
    return z_trajs.reshape(z_trajs.size)

def analyze_statistics(x, y, u, v, ntrajs, nsteps, escape_events, dt, m, kT, max_lag, eqfrac=0.5):
    """
    Calculate statistics from the trajectory and compare with theory.
    
    Parameters:
    -----------
    x, y : arrays
        Position trajectories
    u, v : arrays
        Velocity trajectories
    m : float
        Particle mass
    kT : float
        Thermal energy
    equilibration_fraction : float
        Fraction of trajectory to skip for equilibration
    
    Returns:
    --------
    stats : dict
        Dictionary containing statistical measures
    """

    x_eq = z_eq(x, ntrajs, nsteps, eqfrac)
    y_eq = z_eq(y, ntrajs, nsteps, eqfrac)
    u_eq = z_eq(u, ntrajs, nsteps, eqfrac)
    v_eq = z_eq(v, ntrajs, nsteps, eqfrac)
    
    # Calculate statistics
    x_mean = np.mean(x_eq)
    x_var = np.var(x_eq)
    y_mean = np.mean(y_eq)
    y_var = np.var(y_eq)
    u_mean = np.mean(u_eq)
    u_var = np.var(u_eq)
    v_mean = np.mean(v_eq)
    v_var = np.var(v_eq)
    
    # Theoretical predictions from equipartition theorem
    v_var_theory = kT / m  # <v^2> = kBT/m (Maxwell-Boltzmann)

    # Diffusion coefficients
    D_xx, D_yy, D_xy, msd_x, msd_y, t_single = compute_diffusion_coefficient(x, y, t, ntrajs, nsteps, eqfrac)

    # Velocity correlations
    C_uu, C_vv, C_uv, C_vu = compute_vacf(u, v, ntrajs, nsteps, max_lag)
    C_uu_norm, C_vv_norm = C_uu / C_uu[0], C_vv / C_vv[0]
    tau_x, tau_y = compute_correlation_times(C_uu_norm, C_vv_norm, dt)
    D_KG_xx, D_KG_yy, D_KG_xy, D_KG_yx = compute_diffusion_from_vacf(C_uu, C_vv, C_uv, C_vu, dt)
    
    # Escape direction autocorrelation
    escape_ac = compute_escape_direction_autocorrelation(escape_events)
    
    # Ensemble average distance travelled with time
    xea = ensemble_avg(x, ntrajs, nsteps)
    yea = ensemble_avg(y, ntrajs, nsteps)

    stats = {
        'x_mean': x_mean,
        'x_var': x_var,
        'y_mean': y_mean,
        'y_var': y_var,
        'u_mean': u_mean,
        'u_var': u_var,
        'v_mean': v_mean,
        'v_var': v_var,
        'u_var_theory': v_var_theory,
        'v_var_theory': v_var_theory,
        'D_xx': D_xx,
        'D_yy': D_yy,
        'D_xy': D_xy,
        'msd_x': msd_x.tolist(),
        'msd_y': msd_y.tolist(),
        'C_uu': C_uu.tolist(),
        'C_vv': C_vv.tolist(),
        'C_uv': C_uv.tolist(),
        'C_vu': C_vu.tolist(),
        'C_uu_norm': C_uu_norm.tolist(),
        'C_vv_norm': C_vv_norm.tolist(),
        'tau_x': tau_x,
        'tau_y': tau_y,
        'D_KG_xx': D_KG_xx,
        'D_KG_yy': D_KG_yy,
        'D_KG_xy': D_KG_xy,
        'D_KG_yx': D_KG_yx,
        'escape_ac' : escape_ac.tolist(),
        'x0': x[0],
        'y0': y[0],
        'xf': xea[-1],
        'yf': xea[-1],
        'tf': t[-1]
    }
    
    return stats


def plot_results(t, x, y, u, v, stats):
    """
    Create a comprehensive plot of the simulation results.
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Position vs time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, np.sqrt(x**2 + y**2), 'b-', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Time, $t$', fontsize=12)
    ax1.set_ylabel('Distance from origin, $r$', fontsize=12)
    ax1.set_title('$r$ trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Velocity vs time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, np.sqrt(u**2 + v**2), 'r-', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Speed', fontsize=12)
    ax2.set_title('Speed Trajectory', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
   
    # Phase space
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, u, 'purple', linewidth=0.5, alpha=0.3)
    ax3.scatter(x[::100], v[::100], c='purple', s=1, alpha=0.5)
    ax3.set_xlabel('Position, $x$', fontsize=12)
    ax3.set_ylabel('Velocity, $u$', fontsize=12)
    ax3.set_title('Phase Space $x$-$u$ Slice', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Position histogram
    ax4 = fig.add_subplot(gs[2, 0])
    eq_start = int(len(x) * 0.8)
    counts, bins, _ = ax4.hist(x[eq_start:], bins=50, density=True, 
                                alpha=0.7, color='green', edgecolor='black')
    
    ax4.set_xlabel('Position', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.set_title('Position Distribution (Equilibrated)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Velocity histogram
    ax5 = fig.add_subplot(gs[2, 1])
    counts, bins, _ = ax5.hist(v[eq_start:], bins=50, density=True,
                                alpha=0.7, color='orange', edgecolor='black')
    
    # Overlay theoretical distribution (Maxwell-Boltzmann)
    v_theory = np.linspace(bins[0], bins[-1], 200)
    v_var_theory = stats['v_var_theory']
    p_theory = np.exp(-v_theory**2 / (2 * v_var_theory)) / np.sqrt(2 * np.pi * v_var_theory)
    ax5.plot(v_theory, p_theory, 'r-', linewidth=2, label='Theory')
    
    ax5.set_xlabel('Velocity', fontsize=12)
    ax5.set_ylabel('Probability Density', fontsize=12)
    ax5.set_title('Velocity Distribution (Equilibrated)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Langevin Dynamics: Brownian Motion in a Weave Potential', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

# Main execution
if __name__ == "__main__":
    # Make path to outdir
    outdir = args['outdir']
    directory_path = Path(outdir)
    directory_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{directory_path}' is ready to use.")

    # Set random seed for reproducibility
    if nargs.seed != None:
        np.random.seed(nargs.seed)
        print(f"Using seed {nargs.seed}...")
    else:
        # Seed with current time (in nanoseconds for more randomness)
        seed = int(time.time() * 1e9) % (2**32)
        np.random.seed(seed)
        args['seed'] = seed

    A, a, kT = args['A'], args['a'], args['kT']
    args['alpha'] = A / kT          # barrier to thermal energy ratio
    Fpx, Fpy = args['Fpx'], args['Fpy']
    L, M = args['L'], args['M']
    args['beta_x'] = Fpx * L / kT   # Peclet number
    gamma, m = args['gamma'], args['m']
    if A*a != 0:
        args['eps_x'] = Fpx * L / (A*a) # Tilting parameter
        args['eps_y'] = Fpy * M / (A*a) # Tilting parameter
        args['zeta'] = gamma**2 / (4*m*A*a/L**2)
    else:
        args['eps_x'] = np.nan
        args['eps_y'] = np.nan
        args['zeta'] = np.nan
    args['beta_y'] = Fpy * M / kT   # Peclet number
    args['lambda'] = L / M          # Aspect ratio
    args['tau'] = kT / (gamma * L**2)
    
    print("\nRunning Langevin dynamics simulation...")
    print()
    for (k, v) in args.items():
        print(f"\t{k} = {v}")
    print()
    
    with open(os.path.join(args['outdir'], 'params.json'), 'w') as json_file:
        json.dump(args, json_file, indent=4)

    # Run simulation
    t, x, y, u, v = langevin_simulation_parallel(args)

    if args['save_trajs']:
        print()
        print('Saving trajectories...')
        # Save trajectories
        data_dict = {
                'time': t,
                'x': x,
                'y': y,
                'u': u,
                'v': v
                }
        df = pd.DataFrame(data_dict)
        df.to_csv(os.path.join(outdir, 'trajs.csv'), index=False)
    
    # Analyze statistics
    ntrajs = args['ntrajs']
    nsteps = args['nsteps']
    nsamples = nsteps // args['outfreq']
    dt = args['dt']
    print('Identifying escape events...')
    escape_events = identify_escape_events(x, y, t, ntrajs, nsteps, L, M)
    if args['print_escapes']:
        pprint(escape_events)
        print()
    with open(os.path.join(args['outdir'], 'escapes.json'), 'w') as json_file:
        json.dump(escape_events, json_file, indent=4)

    print()
    print('Analyzing statistics...')
    stats = analyze_statistics(x, y, u, v, ntrajs, nsamples, escape_events, dt, 
                               m, kT, args['max_lag'], args['eqfrac'])
    for (k, val) in stats.items():
        if type(val) == list:
            continue
        print(f"\t{k} = {val}")
    print()
    with open(os.path.join(args['outdir'], 'stats.json'), 'w') as json_file:
        json.dump(stats, json_file, indent=4)
    
    # Create plots
    do_plots, show_plots = args['do_plots'], args['show_plots']
    if do_plots:
        print('Plotting results...')
        fig = plot_results(t, x, y, u, v, stats)
        if show_plots:
            plt.show()
        fig.savefig(os.path.join(outdir, 'statistics.pdf'), dpi=300, bbox_inches='tight')

        fig2 = plot_2d_trajectory_colored(x[:nsamples-1], y[:nsamples-1], potential_func=lambda x, y: U(x, y, A, a, L, M))
        if show_plots:
            plt.show()
        fig2.savefig(os.path.join(outdir, 'trajectory1.pdf'), dpi=300, bbox_inches='tight')

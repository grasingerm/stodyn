#!/usr/bin/env python3
"""
Canopy Light-Sharing Model
Tests the canopy-level load-sharing hypothesis for green light rejection.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools
import sys

# --- Constants & Physics ---
# Photon energies in eV (approximate)
E_RED = 1.8
E_GREEN = 2.3
E_BLUE = 2.8

class CanopySimulation:
    def __init__(self, nx=10, ny=10, nz=5, bc_x='periodic', bc_y='periodic', bc_z='open',
                 absorb_probs=(0.9, 0.5, 0.9), scatter_anisotropy=0.5,
                 source_coverage=1.0, spectrum='flat', energy_model='quantum',
                 sat_cap=1.0, sat_curve=0.9, num_photons=100000):
        
        self.dims = np.array([nx, ny, nz])
        self.bc = (bc_x, bc_y, bc_z)
        self.absorb_probs = np.array(absorb_probs) # R, G, B
        self.anisotropy = scatter_anisotropy # 1.0 = all z, 0.0 = all x/y
        self.coverage = source_coverage
        self.spectrum = spectrum
        self.energy_model = energy_model
        self.cap = sat_cap
        self.theta_sat = sat_curve
        self.n_photons = num_photons
        
        # Site accumulators for absorbed quanta: shape (nx, ny, nz, 3)
        self.absorbed_quanta = np.zeros((nx, ny, nz, 3))
        self.escaped = {'sky': 0, 'ground': 0, 'lateral': 0}
        
    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # 1. Initialize photons
        bands = self._initialize_spectrum()
        positions = self._initialize_positions()
        active = np.ones(self.n_photons, dtype=bool)
        
        # 2. Random Walk
        while np.any(active):
            n_active = np.sum(active)
            curr_pos = positions[active]
            curr_bands = bands[active]
            
            # Decide absorption
            rand_abs = np.random.rand(n_active)
            p_abs = self.absorb_probs[curr_bands]
            absorbed = rand_abs < p_abs
            
            # Record absorptions
            abs_pos = curr_pos[absorbed]
            abs_bands = curr_bands[absorbed]
            for p, b in zip(abs_pos, abs_bands):
                self.absorbed_quanta[p[0], p[1], p[2], b] += 1
                
            active[np.where(active)[0][absorbed]] = False
            
            # For survivors, scatter
            survivors = ~absorbed
            if not np.any(survivors):
                break
                
            n_surv = np.sum(survivors)
            scatter_dirs = self._get_scattering_directions(n_surv)
            
            # Update positions for survivors
            active_indices = np.where(active)[0]
            positions[active_indices] += scatter_dirs
            
            # Handle Boundary Conditions
            active = self._apply_boundaries(positions, active)
            
        return self._calculate_metabolism()
        
    def _initialize_spectrum(self):
        if self.spectrum == 'flat':
            return np.random.choice([0, 1, 2], size=self.n_photons)
        elif self.spectrum == 'am1.5g':
            # Simplified weighting for photon flux
            return np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20], size=self.n_photons)
            
    def _initialize_positions(self):
        nx, ny, nz = self.dims
        positions = np.zeros((self.n_photons, 3), dtype=int)
        
        # Source coverage logic
        area = int(nx * ny * self.coverage)
        xs = np.random.randint(0, nx, size=self.n_photons)
        ys = np.random.randint(0, ny, size=self.n_photons)
        
        mask = (xs * ny + ys) < area
        positions[:, 0] = xs * mask
        positions[:, 1] = ys * mask
        positions[:, 2] = 0 # Top face
        return positions
        
    def _get_scattering_directions(self, n):
        dirs = np.zeros((n, 3), dtype=int)
        # Anisotropy: fraction through-thickness (z) vs in-plane (x, y)
        z_scatter = np.random.rand(n) < self.anisotropy
        
        # Z-direction (+1 or -1)
        dirs[z_scatter, 2] = np.random.choice([-1, 1], size=np.sum(z_scatter))
        
        # X/Y-direction
        xy_scatter = ~z_scatter
        axis = np.random.choice([0, 1], size=np.sum(xy_scatter))
        sign = np.random.choice([-1, 1], size=np.sum(xy_scatter))
        
        x_mask = xy_scatter & (axis == 0)
        y_mask = xy_scatter & (axis == 1)
        
        dirs[x_mask, 0] = sign[x_mask[:np.sum(x_mask)]]
        dirs[y_mask, 1] = sign[y_mask[:np.sum(y_mask)]]
        
        return dirs
        
    def _apply_boundaries(self, pos, active):
        nx, ny, nz = self.dims
        active_indices = np.where(active)[0]
        
        for i, idx in enumerate(active_indices):
            p = pos[idx]
            
            # X boundary
            if p[0] < 0 or p[0] >= nx:
                if self.bc[0] == 'periodic':
                    pos[idx, 0] %= nx
                else:
                    active[idx] = False
                    self.escaped['lateral'] += 1
                    continue
                    
            # Y boundary
            if p[1] < 0 or p[1] >= ny:
                if self.bc[1] == 'periodic':
                    pos[idx, 1] %= ny
                else:
                    active[idx] = False
                    self.escaped['lateral'] += 1
                    continue
                    
            # Z boundary
            if p[2] < 0:
                if self.bc[2] == 'periodic':
                    pos[idx, 2] %= nz
                else:
                    active[idx] = False
                    self.escaped['sky'] += 1
            elif p[2] >= nz:
                if self.bc[2] == 'periodic':
                    pos[idx, 2] %= nz
                else:
                    active[idx] = False
                    self.escaped['ground'] += 1
                    
        return active
        
    def _calculate_metabolism(self):
        # Energy definitions
        energies = np.array([E_RED, E_GREEN, E_BLUE])
        
        if self.energy_model == 'quantum':
            usable_energy = self.absorbed_quanta * E_RED
            thermal_waste = self.absorbed_quanta * (energies - E_RED)
        else: # thermodynamic
            usable_energy = self.absorbed_quanta * energies
            thermal_waste = np.zeros_like(usable_energy)
            
        total_usable_input = np.sum(usable_energy, axis=-1)
        
        # Non-rectangular hyperbola for saturation
        # theta * u^2 - (I + C) * u + I * C = 0
        I = total_usable_input
        C = self.cap * (self.n_photons / (self.dims[0]*self.dims[1])) # scaled by mean flux
        theta = self.theta_sat
        
        # Solve quadratic
        if theta > 0:
            discriminant = (I + C)**2 - 4 * theta * I * C
            u = ((I + C) - np.sqrt(np.maximum(0, discriminant))) / (2 * theta)
        else:
            u = np.minimum(I, C)
            
        sat_waste = I - u
        
        total_u = np.sum(u)
        total_q = np.sum(thermal_waste) + np.sum(sat_waste)
        
        return total_u, total_q

def sweep_pareto(args):
    """Sweeps theta to find the Pareto front and margin plot data."""
    results = []
    
    # Grid search for absorption probabilities (pinned red/blue vs unconstrained)
    if args.constrained:
        r_probs = [0.9]
        b_probs = [0.9]
        g_probs = np.linspace(0.1, 0.9, 9)
    else:
        r_probs = np.linspace(0.5, 0.9, 3)
        b_probs = np.linspace(0.5, 0.9, 3)
        g_probs = np.linspace(0.1, 0.9, 5)
        
    configs = list(itertools.product(r_probs, g_probs, b_probs))
    
    print(f"Running {len(configs)} configurations for Pareto front...")
    for r, g, b in configs:
        sim = CanopySimulation(
            nx=args.nx, ny=args.ny, nz=args.nz,
            absorb_probs=(r, g, b),
            scatter_anisotropy=args.anisotropy,
            source_coverage=args.coverage,
            spectrum=args.spectrum,
            energy_model=args.energy_model,
            sat_cap=args.cap,
            num_photons=args.photons
        )
        u, q = sim.run(seed=args.seed)
        results.append({'R': r, 'G': g, 'B': b, 'u': u, 'q': q})
        
    df = pd.DataFrame(results)
    
    # Calculate objectives for different thetas
    thetas = np.linspace(0, 1, 21)
    margins = []
    
    for theta in thetas:
        df['obj'] = theta * df['q'] - (1 - theta) * df['u']
        best_idx = df['obj'].idxmin()
        best = df.loc[best_idx]
        
        # Find best green-rejecting (G < R and G < B)
        gr_mask = (df['G'] < df['R']) & (df['G'] < df['B'])
        if gr_mask.any() and (~gr_mask).any():
            best_gr = df[gr_mask]['obj'].min()
            best_non_gr = df[~gr_mask]['obj'].min()
            margin = best_non_gr - best_gr # Positive means green-rejecting is better
        else:
            margin = 0
            
        margins.append(margin)
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pareto
    ax1.scatter(df['u'], df['q'], c=df['G'], cmap='viridis')
    ax1.set_xlabel('Metabolized Energy (u)')
    ax1.set_ylabel('Waste Heat (q)')
    ax1.set_title('Objective Space (Colored by Green Abs)')
    
    # Margin
    ax2.plot(thetas, margins, 'k-', marker='o')
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Heat Weight ($theta$)')
    ax2.set_ylabel('Margin (Non-GR best - GR best)')
    ax2.set_title('Green-Rejecting Advantage Margin')
    
    plt.tight_layout()
    plt.savefig('diagnostic_output.png')
    print("Saved diagnostic plots to 'diagnostic_output.png'.")
    df.to_csv('results_dump.csv', index=False)
    print("Saved machine-readable data to 'results_dump.csv'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Canopy Light-Sharing Model")
    parser.add_argument('--nx', type=int, default=10)
    parser.add_argument('--ny', type=int, default=10)
    parser.add_argument('--nz', type=int, default=5)
    parser.add_argument('--anisotropy', type=float, default=0.5, help="0.0 to 1.0 through-thickness scattering")
    parser.add_argument('--coverage', type=float, default=1.0, help="Illuminated top fraction")
    parser.add_argument('--spectrum', choices=['flat', 'am1.5g'], default='flat')
    parser.add_argument('--energy-model', choices=['quantum', 'thermo'], default='quantum')
    parser.add_argument('--cap', type=float, default=1.0, help="Metabolic saturation cap")
    parser.add_argument('--photons', type=int, default=50000, help="MC photon count")
    parser.add_argument('--constrained', action='store_true', help="Pin Red and Blue absorption to 0.9")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    sweep_pareto(args)
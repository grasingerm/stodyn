#!/usr/bin/env python3
"""
Markov Chain Photon Transport Simulator for Canopy Light Absorption

Simulates discrete photon random walks on a 2D lattice representing a plant canopy.
Tracks wavelength-dependent absorption, scattering, and escape dynamics.

Author: Research collaboration
Date: 2026-01-02
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum

class PhotonFate(Enum):
    """Possible outcomes for a photon"""
    ABSORBED_LEAF = 1
    ESCAPED_UP = 2
    ABSORBED_GROUND = 3
    ACTIVE = 4  # Still scattering

class BoundaryCondition(Enum):
    """Boundary conditions for lattice edges"""
    OPEN = "open"  # Photons escape at boundaries
    PERIODIC = "periodic"  # Wrap-around boundaries

@dataclass
class PhotonProperties:
    """Wavelength-dependent photon scattering properties"""
    wavelength: str
    p_absorb: float  # Probability absorbed by leaf
    p_scatter: float  # Probability scattered to neighbor
    p_escape_up: float  # Probability escapes to atmosphere
    p_ground: float  # Probability reaches ground
    
    def validate(self):
        """Check probabilities sum to 1"""
        total = self.p_absorb + self.p_scatter + self.p_escape_up + self.p_ground
        assert abs(total - 1.0) < 1e-6, f"Probabilities must sum to 1, got {total}"

class CanopyPhotonSimulator:
    """Simulates photon transport through a 2D canopy lattice"""
    
    def __init__(self, 
                 lattice_size: int,
                 boundary_condition: BoundaryCondition,
                 rho_ground: float = 0.0):
        """
        Initialize simulator
        
        Args:
            lattice_size: Size of square lattice (NxN)
            boundary_condition: OPEN (escape) or PERIODIC (wrap)
            rho_ground: Ground reflectance (0 = fully absorbing)
        """
        self.N = lattice_size
        self.boundary = boundary_condition
        self.rho_ground = rho_ground
        
        # Initialize absorption counters [time, wavelength, site_i, site_j]
        self.absorption_history = {}  # Will store for each wavelength
        
    def simulate_wavelength(self,
                          props: PhotonProperties,
                          n_photons: int,
                          max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate photons of a single wavelength
        
        Args:
            props: Photon properties (absorption, scattering probabilities)
            n_photons: Number of photons to simulate
            max_steps: Maximum time steps to simulate
            
        Returns:
            absorption_per_site: Array[time, i, j] of absorbed photons
            total_absorption_time: Array[time] of cumulative absorption
        """
        props.validate()
        
        # Initialize counters
        absorption_per_site = np.zeros((max_steps, self.N, self.N))
        total_absorption_time = np.zeros(max_steps)
        
        # Simulate each photon
        for photon_idx in range(n_photons):
            # Start at center of lattice
            i, j = self.N // 2, self.N // 2
            
            for t in range(max_steps):
                # Determine fate
                r = np.random.random()
                
                if r < props.p_absorb:
                    # Absorbed by leaf at current site
                    absorption_per_site[t, i, j] += 1
                    total_absorption_time[t] += 1
                    break
                    
                elif r < props.p_absorb + props.p_escape_up:
                    # Escaped to atmosphere
                    break
                    
                elif r < props.p_absorb + props.p_escape_up + props.p_ground:
                    # Reached ground
                    if np.random.random() < self.rho_ground:
                        # Reflected back up - continue scattering
                        pass
                    else:
                        # Absorbed by ground
                        break
                else:
                    # Scatter to neighbor
                    i, j = self._scatter_to_neighbor(i, j)
                    if i is None:  # Escaped through boundary
                        break
        
        # Convert to cumulative absorption over time
        cumulative_absorption = np.cumsum(total_absorption_time)
        
        return absorption_per_site, cumulative_absorption
    
    def _scatter_to_neighbor(self, i: int, j: int) -> Tuple[int, int]:
        """
        Scatter photon to random neighbor
        
        Returns:
            (i_new, j_new) or (None, None) if escaped through boundary
        """
        # Choose random direction: up, down, left, right
        direction = np.random.randint(0, 4)
        
        di_dj = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        di, dj = di_dj[direction]
        
        i_new = i + di
        j_new = j + dj
        
        # Handle boundaries
        if self.boundary == BoundaryCondition.PERIODIC:
            i_new = i_new % self.N
            j_new = j_new % self.N
        else:  # OPEN boundaries
            if i_new < 0 or i_new >= self.N or j_new < 0 or j_new >= self.N:
                return None, None  # Escaped
        
        return i_new, j_new
    
    def run_simulation(self,
                      wavelengths: List[PhotonProperties],
                      n_photons: int,
                      max_steps: int) -> dict:
        """
        Run full simulation for multiple wavelengths
        
        Returns:
            results: Dict with absorption data for each wavelength
        """
        results = {}
        
        for props in wavelengths:
            print(f"Simulating {props.wavelength} light: {n_photons} photons, {max_steps} steps...")
            abs_per_site, cumulative = self.simulate_wavelength(props, n_photons, max_steps)
            
            results[props.wavelength] = {
                'absorption_per_site': abs_per_site,
                'cumulative_absorption': cumulative,
                'properties': props
            }
        
        return results

def plot_results(results: dict, n_photons: int, lattice_size: int):
    """Create visualization of simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color map for wavelengths
    colors = {'red': 'red', 'green': 'green', 'blue': 'blue'}
    
    # Plot 1: Cumulative absorption over time
    ax = axes[0, 0]
    for wavelength, data in results.items():
        cumulative = data['cumulative_absorption']
        time_steps = np.arange(len(cumulative))
        efficiency = cumulative / n_photons * 100
        ax.plot(time_steps, efficiency, 
                label=f'{wavelength}', 
                color=colors.get(wavelength, 'black'),
                linewidth=2)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Absorption (%)', fontsize=12)
    ax.set_title('Photon Absorption Efficiency Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Absorption rate (derivative)
    ax = axes[0, 1]
    for wavelength, data in results.items():
        cumulative = data['cumulative_absorption']
        rate = np.gradient(cumulative)
        time_steps = np.arange(len(rate))
        ax.plot(time_steps, rate,
                label=f'{wavelength}',
                color=colors.get(wavelength, 'black'),
                alpha=0.7,
                linewidth=2)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Absorption Rate (photons/step)', fontsize=12)
    ax.set_title('Instantaneous Absorption Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spatial distribution (final state for green)
    ax = axes[1, 0]
    if 'green' in results:
        # Sum over all time steps to get total spatial distribution
        spatial = np.sum(results['green']['absorption_per_site'], axis=0)
        im = ax.imshow(spatial, cmap='Greens', origin='lower')
        ax.set_xlabel('Lattice X', fontsize=12)
        ax.set_ylabel('Lattice Y', fontsize=12)
        ax.set_title('Green Light: Spatial Absorption Pattern', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Total Absorbed Photons')
    
    # Plot 4: Comparison of final efficiencies
    ax = axes[1, 1]
    wavelengths_list = list(results.keys())
    final_absorption = [results[w]['cumulative_absorption'][-1] / n_photons * 100 
                       for w in wavelengths_list]
    bars = ax.bar(wavelengths_list, final_absorption,
                  color=[colors.get(w, 'gray') for w in wavelengths_list],
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=2)
    
    ax.set_ylabel('Final Absorption Efficiency (%)', fontsize=12)
    ax.set_title('Final Capture Efficiency by Wavelength', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, val in zip(bars, final_absorption):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('canopy_photon_simulation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'canopy_photon_simulation.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Simulate photon transport in plant canopy using Markov chain',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument('--lattice-size', type=int, default=50,
                       help='Size of square lattice (NxN)')
    parser.add_argument('--n-photons', type=int, default=10000,
                       help='Number of photons to simulate per wavelength')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum time steps per photon')
    parser.add_argument('--boundary', type=str, default='open',
                       choices=['open', 'periodic'],
                       help='Boundary condition: open (escape) or periodic (wrap)')
    parser.add_argument('--rho-ground', type=float, default=0.1,
                       help='Ground reflectance (0-1)')
    
    # Red light properties
    parser.add_argument('--red-absorb', type=float, default=0.85,
                       help='Red: probability absorbed by leaf')
    parser.add_argument('--red-scatter', type=float, default=0.10,
                       help='Red: probability scattered to neighbor')
    parser.add_argument('--red-escape', type=float, default=0.03,
                       help='Red: probability escapes to atmosphere')
    parser.add_argument('--red-ground', type=float, default=0.02,
                       help='Red: probability reaches ground')
    
    # Green light properties  
    parser.add_argument('--green-absorb', type=float, default=0.15,
                       help='Green: probability absorbed by leaf')
    parser.add_argument('--green-scatter', type=float, default=0.75,
                       help='Green: probability scattered to neighbor')
    parser.add_argument('--green-escape', type=float, default=0.05,
                       help='Green: probability escapes to atmosphere')
    parser.add_argument('--green-ground', type=float, default=0.05,
                       help='Green: probability reaches ground')
    
    # Blue light properties
    parser.add_argument('--blue-absorb', type=float, default=0.80,
                       help='Blue: probability absorbed by leaf')
    parser.add_argument('--blue-scatter', type=float, default=0.12,
                       help='Blue: probability scattered to neighbor')
    parser.add_argument('--blue-escape', type=float, default=0.05,
                       help='Blue: probability escapes to atmosphere')
    parser.add_argument('--blue-ground', type=float, default=0.03,
                       help='Blue: probability reaches ground')
    
    args = parser.parse_args()
    
    # Create photon properties
    wavelengths = [
        PhotonProperties('red', args.red_absorb, args.red_scatter, 
                        args.red_escape, args.red_ground),
        PhotonProperties('green', args.green_absorb, args.green_scatter,
                        args.green_escape, args.green_ground),
        PhotonProperties('blue', args.blue_absorb, args.blue_scatter,
                        args.blue_escape, args.blue_ground),
    ]
    
    # Validate probabilities
    for props in wavelengths:
        props.validate()
    
    # Print configuration
    print("=" * 60)
    print("CANOPY PHOTON TRANSPORT SIMULATION")
    print("=" * 60)
    print(f"Lattice size: {args.lattice_size} x {args.lattice_size}")
    print(f"Photons per wavelength: {args.n_photons}")
    print(f"Max steps per photon: {args.max_steps}")
    print(f"Boundary condition: {args.boundary}")
    print(f"Ground reflectance: {args.rho_ground}")
    print("\nWavelength Properties:")
    for props in wavelengths:
        print(f"\n{props.wavelength.upper()}:")
        print(f"  Absorption:  {props.p_absorb:.3f}")
        print(f"  Scattering:  {props.p_scatter:.3f}")
        print(f"  Escape (up): {props.p_escape_up:.3f}")
        print(f"  Ground:      {props.p_ground:.3f}")
    print("=" * 60)
    
    # Create simulator
    boundary = BoundaryCondition.OPEN if args.boundary == 'open' else BoundaryCondition.PERIODIC
    simulator = CanopyPhotonSimulator(args.lattice_size, boundary, args.rho_ground)
    
    # Run simulation
    results = simulator.run_simulation(wavelengths, args.n_photons, args.max_steps)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for wavelength, data in results.items():
        final_absorbed = data['cumulative_absorption'][-1]
        efficiency = final_absorbed / args.n_photons * 100
        print(f"{wavelength.upper()}: {final_absorbed:.0f}/{args.n_photons} photons absorbed ({efficiency:.1f}%)")
    print("=" * 60)
    
    # Plot results
    plot_results(results, args.n_photons, args.lattice_size)

if __name__ == '__main__':
    main()

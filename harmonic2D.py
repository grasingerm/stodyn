import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

parser = argparse.ArgumentParser(description='Simulation of Brownian motion in 2D')
parser.add_argument('--m', type=float, default=1.0, help='mass of the particle')
parser.add_argument('--gamma', type=float, default=1.0, help='drag coefficient')
parser.add_argument('--kT', type=float, default=1.0, help='temperature')
parser.add_argument('--k', type=float, default=1.0, help='spring constant')
parser.add_argument('--dt', type=float, default=0.01, help='time step')
parser.add_argument('--nsteps', type=int, default=100000, help='number of steps')
parser.add_argument('--x0', type=float, default=0.0, help='initial x position')
parser.add_argument('--y0', type=float, default=0.0, help='initial y position')
parser.add_argument('--u0', type=float, default=0.0, help='initial x velocity')
parser.add_argument('--v0', type=float, default=0.0, help='initial y velocity')
parser.add_argument('--outfreq', type=int, default=100, help='number of iterations per diagnostic information')
parser.add_argument('--outdir', type=str, default="temp", help='output directory')
parser.add_argument('--do_plots', default=False, action="store_true", help='create plots of microstates and clusters')
parser.add_argument('--seed', type=int, help='seed for random number generator')

#namespace_args = parser.parse_args()
#args = vars(namespace_args)
args = parser.parse_args()

def U(x, y, a=1.0):
    return np.exp(a*(np.sin(x-y) + np.sin(x+y)))

def gradU(x, y, a=1.0):
    sx, sy, cx, cy = np.sin(x), np.sin(y), np.cos(x), np.cos(y)
    return 2*a*np.exp(2*a*cy*sx) * np.array([cx*cy, -sx*sy])
        
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
    cbar = fig.colorbar(line, ax=ax1, label='Time (normalized)')
    
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
        cbar2 = fig.colorbar(contour, ax=ax2, label='Potential Energy V(x,y)')
        
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

def langevin_simulation(mass=1.0, gamma=1.0, kBT=1.0, k=1.0, 
                        dt=0.01, n_steps=5000, x0=0.0, y0=0.0, u0=0.0, v0=0.0):
    """
    Simulate Brownian motion using the Langevin equation with Euler-Maruyama integration.
    
    The equation of motion is:
    m dv/dt = -k*x - gamma*v + xi(t)
    
    where xi(t) is white noise with <xi(t)xi(t')> = 2*gamma*kBT*delta(t-t')
    
    Parameters:
    -----------
    mass : float
        Particle mass
    gamma : float
        Damping coefficient
    kBT : float
        Thermal energy (Boltzmann constant * Temperature)
    k : float
        Spring constant for harmonic potential
    dt : float
        Time step
    n_steps : int
        Number of integration steps
    x0, v0 : float
        Initial position and velocity
    
    Returns:
    --------
    t, x, v : arrays
        Time, position, and velocity trajectories
    """
    
    # Initialize arrays
    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    
    x[0] = x0
    y[0] = y0
    u[0] = u0
    v[0] = v0
    
    # Noise strength: sqrt(2 * gamma * kBT / mass)
    noise_strength = np.sqrt(2 * gamma * kBT / mass)
    
    # Euler-Maruyama integration
    for i in range(n_steps - 1):
        # Deterministic force
        Fx = -k * x[i] - gamma * u[i]
        Fy = -k * y[i] - gamma * v[i]
        
        # Random force from standard normal distribution
        Frx = noise_strength * np.random.randn()
        Fry = noise_strength * np.random.randn()
        
        # Update velocity: dv = F/m*dt + noise*sqrt(dt)
        u[i + 1] = u[i] + Fx / mass * dt + Frx * np.sqrt(dt)
        v[i + 1] = v[i] + Fy / mass * dt + Fry * np.sqrt(dt)
        
        # Update position
        x[i + 1] = x[i] + u[i + 1] * dt
        y[i + 1] = y[i] + v[i + 1] * dt
        
        t[i + 1] = t[i] + dt
    
    return t, x, y, u, v


def analyze_statistics(x, y, u, v, mass, kBT, k, equilibration_fraction=0.8):
    """
    Calculate statistics from the trajectory and compare with theory.
    
    Parameters:
    -----------
    x, y : arrays
        Position trajectories
    u, v : arrays
        Velocity trajectories
    mass : float
        Particle mass
    kBT : float
        Thermal energy
    k : float
        Spring constant
    equilibration_fraction : float
        Fraction of trajectory to skip for equilibration
    
    Returns:
    --------
    stats : dict
        Dictionary containing statistical measures
    """
    
    # Use only equilibrated portion
    eq_start = int(len(x) * equilibration_fraction)
    x_eq = x[eq_start:]
    u_eq = v[eq_start:]
    y_eq = x[eq_start:]
    v_eq = v[eq_start:]
    
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
    x_var_theory = kBT / k  # <x^2> = kBT/k for harmonic oscillator
    v_var_theory = kBT / mass  # <v^2> = kBT/m (Maxwell-Boltzmann)
    
    stats = {
        'x_mean': x_mean,
        'x_var': x_var,
        'y_mean': y_mean,
        'y_var': y_var,
        'u_mean': u_mean,
        'u_var': u_var,
        'v_mean': v_mean,
        'v_var': v_var,
        'x_var_theory': x_var_theory,
        'y_var_theory': x_var_theory,
        'u_var_theory': v_var_theory,
        'v_var_theory': v_var_theory
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
    
    # Overlay theoretical distribution (Boltzmann)
    x_theory = np.linspace(bins[0], bins[-1], 200)
    x_var_theory = stats['x_var_theory']
    p_theory = np.exp(-x_theory**2 / (2 * x_var_theory)) / np.sqrt(2 * np.pi * x_var_theory)
    ax4.plot(x_theory, p_theory, 'r-', linewidth=2, label='Theory')
    
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
    
    plt.suptitle('Langevin Dynamics: Brownian Motion in Harmonic Potential', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def print_statistics(stats):
    """
    Print simulation statistics and comparison with theory.
    """
    print("\n" + "="*60)
    print("SIMULATION STATISTICS (Last 20% of trajectory)")
    print("="*60)
    print("\nPosition Statistics:")
    print(f"  Mean:               {stats['x_mean']:10.6f}")
    print(f"  Variance (sim):     {stats['x_var']:10.6f}")
    print(f"  Variance (theory):  {stats['x_var_theory']:10.6f}")
    print(f"  Relative error:     {100*abs(stats['x_var']-stats['x_var_theory'])/stats['x_var_theory']:10.2f}%")
    
    print("\nVelocity Statistics:")
    print(f"  Mean:               {stats['u_mean']:10.6f}")
    print(f"  Variance (sim):     {stats['u_var']:10.6f}")
    print(f"  Variance (theory):  {stats['u_var_theory']:10.6f}")
    print(f"  Relative error:     {100*abs(stats['u_var']-stats['u_var_theory'])/stats['u_var_theory']:10.2f}%")
    
    print("\nTheoretical predictions:")
    print(f"  <x²> = k_B T / k     (equipartition)")
    print(f"  <v²> = k_B T / m     (Maxwell-Boltzmann)")
    print("="*60 + "\n")


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    if args.seed != None:
        np.random.seed(args.seed)
        print(f"Using seed {args.seed}...")
    else:
        np.random.seed()

    print("\nRunning Langevin dynamics simulation...")
    print(f"Parameters: m={args.m}, γ={args.gamma}, k_BT={args.kT}, k={args.k},")
    print(f"            x0={args.x0}, y0={args.y0}, u0={args.u0}, v0={args.v0}")
    print(f"Time step: dt={args.dt}, Total steps: {args.nsteps}")
    
    # Run simulation
    t, x, y, u, v = langevin_simulation(mass=args.m, gamma=args.gamma,
                                        kBT=args.kT, k=args.k,
                                        dt=args.dt, n_steps=args.nsteps,
                                        x0=args.x0, y0=args.y0, 
                                        u0=args.u0, v0=args.v0
                                       )
    
    # Analyze statistics
    stats = analyze_statistics(x, y, u, v, args.m, args.kT, args.k)
    
    # Print results
    print_statistics(stats)
    
    # Create plots
    fig = plot_results(t, x, y, u, v, stats)
    plt.show()

    fig2 = plot_2d_trajectory_colored(x, y)
    plt.show()
    
    # Optional: Save figure
    # fig.savefig('langevin_simulation.png', dpi=300, bbox_inches='tight')

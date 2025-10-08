import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

parser = argparse.ArgumentParser(description='Simulation of pressure-driven Brownian motion through a 2D weave')
parser.add_argument('--m', type=float, default=1.0, help='mass of the particle')
parser.add_argument('--gamma', type=float, default=1.0, help='drag coefficient')
parser.add_argument('--kT', type=float, default=1.0, help='temperature')
parser.add_argument('--gradp', type=float, default=1.0, help='pressure gradient')
parser.add_argument('--a', type=float, default=1.0, help='amplitude of the "weave" potential')
parser.add_argument('--L', type=float, default=1.0, help='distance between peaks of "weave" potential in x-y direction')
parser.add_argument('--M', type=float, default=1.0, help='distance between peaks of "weave" potential in x+y direction')
parser.add_argument('--dt', type=float, default=0.01, help='time step')
parser.add_argument('--nsteps', type=int, default=2000, help='number of steps')
parser.add_argument('--ntrajs', type=int, default=100, help='number of trajectories')
parser.add_argument('--x0', type=float, default=0.0, help='initial x position')
parser.add_argument('--y0', type=float, default=0.0, help='initial y position')
parser.add_argument('--u0', type=float, default=0.0, help='initial x velocity')
parser.add_argument('--v0', type=float, default=0.0, help='initial y velocity')
parser.add_argument('--outfreq', type=int, default=100, help='number of iterations per diagnostic information')
parser.add_argument('--outdir', type=str, default="temp", help='output directory')
parser.add_argument('--do_plots', default=False, action="store_true", help='create plots of microstates and clusters')
parser.add_argument('--seed', type=int, help='seed for random number generator')

nargs = parser.parse_args()
args = vars(nargs)

def U(x, y, a=1.0, w1=2*np.pi, w2=2*np.pi):
    return np.exp(a*(np.sin(w1*(x-y)) + np.sin(w2*(x+y))))

def gradU(x, y, a=1.0, w1=2*np.pi, w2=2*np.pi):
    arg1 = w1*(x-y)
    arg2 = w2*(x+y)
    s1, s2, c1, c2 = np.sin(arg1), np.sin(arg2), np.cos(arg1), np.cos(arg2)
    A = a*np.exp(a*(s1 + s2))
    return np.array([A*(w1*c1 + w2*c2), A*(-w1*c1 + w2*c2)])
        
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

def langevin_simulation(m, gamma, kT, gradp, a, L, M, dt, nsteps, ntrajs,
                        x0, y0, u0, v0, **kwargs):
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
    kT : float
        Thermal energy (Boltzmann constant * Temperature)
    dt : float
        Time step
    nsteps : int
        Number of integration steps
    x0, v0 : float
        Initial position and velocity
    
    Returns:
    --------
    t, x, v : arrays
        Time, position, and velocity trajectories
    """
    
    # Initialize arrays
    t = np.zeros(nsteps*ntrajs)
    x = np.zeros(nsteps*ntrajs)
    y = np.zeros(nsteps*ntrajs)
    u = np.zeros(nsteps*ntrajs)
    v = np.zeros(nsteps*ntrajs)
        
    # Noise strength: sqrt(2 * gamma * kBT / mass)
    noise_strength = np.sqrt(2 * gamma * kT / m)
 
    i = 0
    for itr in range(ntrajs):
        x[i] = x0
        y[i] = y0
        u[i] = u0
        v[i] = v0
        
        # Euler-Maruyama integration
        for step in range(nsteps - 1):
            # Deterministic force
            F = np.array([
                    gradp - gamma*u[i],
                    -gamma*v[i]
                ]) - gradU(x[i], y[i], a, 2*np.pi/L, 2*np.pi/M)
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

            # Increment i
            i += 1
    
    return t, x, y, u, v


def analyze_statistics(x, y, u, v, mass, kBT, equilibration_fraction=0.8):
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
    
    print("\nVelocity Statistics:")
    print(f"  Mean:               {stats['u_mean']:10.6f}")
    print(f"  Variance (sim):     {stats['u_var']:10.6f}")
    print(f"  Variance (theory):  {stats['u_var_theory']:10.6f}")
    print(f"  Relative error:     {100*abs(stats['u_var']-stats['u_var_theory'])/stats['u_var_theory']:10.2f}%")
    
    print("\nTheoretical predictions:")
    print(f"  <v²> = k_B T / m     (Maxwell-Boltzmann)")
    print("="*60 + "\n")


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    if nargs.seed != None:
        np.random.seed(nargs.seed)
        print(f"Using seed {nargs.seed}...")
    else:
        np.random.seed()

    print("\nRunning Langevin dynamics simulation...")
    print(f"Parameters: m={nargs.m}, γ={nargs.gamma}, k_BT={nargs.kT},")
    print(f"            gradp={nargs.gradp}, a={nargs.a}, L={nargs.L}, M={nargs.M},")
    print(f"            x0={nargs.x0}, y0={nargs.y0}, u0={nargs.u0}, v0={nargs.v0}")
    print(f"Time step: dt={nargs.dt}, Total steps: {nargs.nsteps}, Total trajectories: {nargs.ntrajs}")
    
    # Run simulation
    t, x, y, u, v = langevin_simulation(**args)
    
    # Analyze statistics
    stats = analyze_statistics(x, y, u, v, args['m'], args['kT'])
    
    # Print results
    print_statistics(stats)
    
    # Create plots
    fig = plot_results(t, x, y, u, v, stats)
    plt.show()

    fig2 = plot_2d_trajectory_colored(x[:args['nsteps']-1], y[:args['nsteps']-1], potential_func=U)
    plt.show()
    
    # Optional: Save figure
    # fig.savefig('langevin_simulation.png', dpi=300, bbox_inches='tight')

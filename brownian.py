import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def langevin_simulation(mass=1.0, gamma=1.0, kBT=1.0, k=1.0, 
                        dt=0.01, n_steps=5000, x0=2.0, v0=0.0):
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
    v = np.zeros(n_steps)
    
    x[0] = x0
    v[0] = v0
    
    # Noise strength: sqrt(2 * gamma * kBT / mass / dt)
    noise_strength = np.sqrt(2 * gamma * kBT / mass / dt)
    
    # Euler-Maruyama integration
    for i in range(n_steps - 1):
        # Deterministic force from harmonic potential: F = -k*x
        force = -k * x[i]
        
        # Random force from standard normal distribution
        random_force = noise_strength * np.random.randn()
        
        # Update velocity: dv = (F/m - gamma*v/m)*dt + noise*sqrt(dt)
        v[i + 1] = v[i] + (force / mass - gamma * v[i] / mass) * dt + random_force * np.sqrt(dt)
        
        # Update position
        x[i + 1] = x[i] + v[i + 1] * dt
        
        t[i + 1] = t[i] + dt
    
    return t, x, v


def analyze_statistics(x, v, mass, kBT, k, equilibration_fraction=0.8):
    """
    Calculate statistics from the trajectory and compare with theory.
    
    Parameters:
    -----------
    x, v : arrays
        Position and velocity trajectories
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
    v_eq = v[eq_start:]
    
    # Calculate statistics
    x_mean = np.mean(x_eq)
    x_var = np.var(x_eq)
    v_mean = np.mean(v_eq)
    v_var = np.var(v_eq)
    
    # Theoretical predictions from equipartition theorem
    x_var_theory = kBT / k  # <x^2> = kBT/k for harmonic oscillator
    v_var_theory = kBT / mass  # <v^2> = kBT/m (Maxwell-Boltzmann)
    
    stats = {
        'x_mean': x_mean,
        'x_var': x_var,
        'x_var_theory': x_var_theory,
        'v_mean': v_mean,
        'v_var': v_var,
        'v_var_theory': v_var_theory
    }
    
    return stats


def plot_results(t, x, v, stats):
    """
    Create a comprehensive plot of the simulation results.
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Position vs time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, x, 'b-', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Position Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Velocity vs time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, v, 'r-', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Velocity Trajectory', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Phase space
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, v, 'purple', linewidth=0.5, alpha=0.3)
    ax3.scatter(x[::100], v[::100], c='purple', s=1, alpha=0.5)
    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Velocity', fontsize=12)
    ax3.set_title('Phase Space', fontsize=14, fontweight='bold')
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
    print(f"  Mean:               {stats['v_mean']:10.6f}")
    print(f"  Variance (sim):     {stats['v_var']:10.6f}")
    print(f"  Variance (theory):  {stats['v_var_theory']:10.6f}")
    print(f"  Relative error:     {100*abs(stats['v_var']-stats['v_var_theory'])/stats['v_var_theory']:10.2f}%")
    
    print("\nTheoretical predictions:")
    print(f"  <x²> = k_B T / k     (equipartition)")
    print(f"  <v²> = k_B T / m     (Maxwell-Boltzmann)")
    print("="*60 + "\n")


# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulation parameters
    params = {
        'mass': 1.0,
        'gamma': 1.0,
        'kBT': 1.0,
        'k': 1.0,
        'dt': 0.01,
        'n_steps': 10000,
        'x0': 2.0,
        'v0': 0.0
    }
    
    print("\nRunning Langevin dynamics simulation...")
    print(f"Parameters: m={params['mass']}, γ={params['gamma']}, k_BT={params['kBT']}, k={params['k']}")
    print(f"Time step: dt={params['dt']}, Total steps: {params['n_steps']}")
    
    # Run simulation
    t, x, v = langevin_simulation(**params)
    
    # Analyze statistics
    stats = analyze_statistics(x, v, params['mass'], params['kBT'], params['k'])
    
    # Print results
    print_statistics(stats)
    
    # Create plots
    fig = plot_results(t, x, v, stats)
    plt.show()
    
    # Optional: Save figure
    # fig.savefig('langevin_simulation.png', dpi=300, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def simulate_langevin(n_particles, n_steps, dt, kappa, shear_rate, p_ratio, initial_type='uniform'):
    """
    Simulates LC orientation evolution using Langevin dynamics.
    
    Args:
        n_particles: Number of particles in ensemble
        n_steps: Number of time steps
        dt: Time step size
        kappa: Potential strength (Order parameter coupling)
        shear_rate: Strength of shear flow
        p_ratio: Aspect ratio (a/b). p > 1 (rods), p < 1 (discs)
        initial_type: 'uniform' or 'peaked'
    """
    
    # Initialize orientations (u) on unit sphere
    if initial_type == 'peaked':
        # Approximation of vMF peaked at z-axis
        u = np.random.randn(n_particles, 3)
        u[:, 2] += 5.0 # Bias towards z
        u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]
    else:
        # Uniform distribution
        u = np.random.randn(n_particles, 3)
        u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]

    # Define velocity gradient for simple shear: v = (gamma*y, 0, 0)
    # g_ij = dv_i / dx_j. Non-zero component is g_xy = shear_rate
    g = np.zeros((3,3))
    g[0, 1] = shear_rate
    
    # Symmetric and Antisymmetric parts
    g_s = 0.5 * (g + g.T)
    g_a = 0.5 * (g - g.T)
    
    # Molecular shape factor (p^2 - 1)/(p^2 + 1) (Eq. 3 in paper)
    shape_factor = (p_ratio**2 - 1) / (p_ratio**2 + 1)
    
    # Simulation Loop
    for step in range(n_steps):
        # 1. Deterministic Torque from Flow (Jeffery's Orbit terms - Eq 3)
        # term1 = shape_factor * (u x (g_s . u))
        gs_u = np.dot(u, g_s.T) # Batch dot product
        term1 = shape_factor * np.cross(u, gs_u)
        
        # term2 = u x (g_a . u) -> This simplifies to just the rotation vector of the flow
        ga_u = np.dot(u, g_a.T)
        term2 = np.cross(u, ga_u)
        
        omega_flow = term1 + term2
        
        # 2. Deterministic Torque from Nematic Potential (Maier-Saupe)
        # Potential U ~ -kappa * (u . n)^2. Let director n be along x-axis for this test.
        # Torque Gamma = - rot(U) ~ - u x grad(U)
        # For U = -J * (u_x)^2, grad U is proportional to -2 * u_x * x_hat
        # Torque pulls u towards x-axis
        n_director = np.array([1.0, 0.0, 0.0]) # Director along Flow direction X
        dot_un = np.dot(u, n_director) # (N,)
        
        # Gradient of potential w.r.t u is proportional to -2 * (u.n) * n
        # Torque is u x (-grad U)
        # We approximate the potential torque simply
        potential_force = 2 * kappa * dot_un[:, np.newaxis] * n_director
        # Project force to be tangent to sphere (u x force x u is overkill, just u x force gives torque)
        # The rotational drift is torque/friction. 
        # We simply add a step proportional to the gradient projected on the sphere.
        # A simple way in Langevin on sphere: Move u towards force
        
        # 3. Stochastic Noise (Brownian Motion)
        noise = np.random.randn(n_particles, 3) * np.sqrt(2 * dt)
        
        # Update u (Euler-Maruyama integration)
        # du = (Omega_flow x u)dt + (Potential_Torque x u)dt + noise_projected
        
        # Flow update
        du_flow = np.cross(omega_flow, u) * dt
        
        # Potential update (relaxation to director)
        # This effectively pushes u towards n_director if dot > 0 and -n_director if dot < 0
        du_pot = kappa * (n_director * dot_un[:,np.newaxis] - u * (dot_un[:,np.newaxis]**2)) * dt

        # Combine
        u_new = u + du_flow + du_pot + noise
        
        # Re-normalize to stay on sphere
        u = u_new / np.linalg.norm(u_new, axis=1)[:, np.newaxis]

    return u

def plot_distribution(u, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot unit sphere wireframe for reference
    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    ax.plot_wireframe(x, y, z, color='k', alpha=0.1)
    
    # Scatter particles
    # Color by Z component to visualize orthogonality to XY plane
    ax.scatter(u[:,0], u[:,1], u[:,2], c=u[:,2], cmap='viridis', s=5, alpha=0.6)
    
    ax.set_xlabel('X (Flow)')
    ax.set_ylabel('Y (Gradient)')
    ax.set_zlabel('Z (Vorticity)')
    ax.set_title('LC Orientation Distribution')
    
    plt.savefig(filename)
    print(f"Snapshot saved to {filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LC Orientation Evolution')
    parser.add_argument('--steps', type=int, default=500, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step size')
    parser.add_argument('--particles', type=int, default=2000, help='Number of particles')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear rate')
    parser.add_argument('--kappa', type=float, default=0.0, help='Potential strength')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--p_ratio', type=float, default=5.0, help='Aspect ratio (5.0 for rods, 0.2 for discs)')
    parser.add_argument('--initial', type=str, default='uniform', choices=['uniform', 'peaked'], help='Initial distribution')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Starting simulation: Shear={args.shear}, Kappa={args.kappa}, p={args.p_ratio}")
    
    final_u = simulate_langevin(
        args.particles, 
        args.steps, 
        args.dt, 
        args.kappa, 
        args.shear, 
        args.p_ratio, 
        args.initial
    )
    
    plot_distribution(final_u, os.path.join(args.output, 'final_distribution.png'))

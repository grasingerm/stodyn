import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def initialize_distribution(n_particles, distribution_type, kappa=0, mu=np.array([1, 0, 0])):
    """
    Initialize particle orientations u.
    distribution_type: 'uniform' or 'watson'
    """
    u = np.zeros((n_particles, 3))
    
    if distribution_type == 'uniform':
        # Standard algorithm for uniform sampling on sphere
        phi = np.random.uniform(0, 2*np.pi, n_particles)
        costheta = np.random.uniform(-1, 1, n_particles)
        theta = np.arccos(costheta)
        
        u[:, 0] = np.sin(theta) * np.cos(phi)
        u[:, 1] = np.sin(theta) * np.sin(phi)
        u[:, 2] = np.cos(theta)
        
    elif distribution_type == 'watson':
        # Approximate sampling for Watson using Von Mises-Fisher rejection or simple Boltzmann factor weighting
        # For simplicity in this CLI tool: Start uniform and reject based on Boltzmann weight
        # w ~ exp(kappa * (u . mu)^2)
        count = 0
        while count < n_particles:
            # Sample batch
            batch_size = n_particles - count
            phi = np.random.uniform(0, 2*np.pi, batch_size)
            costheta = np.random.uniform(-1, 1, batch_size)
            theta = np.arccos(costheta)
            
            ux = np.sin(theta) * np.cos(phi)
            uy = np.sin(theta) * np.sin(phi)
            uz = np.cos(theta)
            u_cand = np.column_stack((ux, uy, uz))
            
            # Acceptance probability
            energy = kappa * (u_cand @ mu)**2
            # Shift energy to avoid overflow/underflow, mostly for acceptance ratio
            prob = np.exp(energy - np.max(energy)) 
            rand = np.random.rand(batch_size)
            
            accepted = u_cand[rand < prob]
            num_acc = len(accepted)
            
            u[count:count+num_acc] = accepted[:min(num_acc, n_particles-count)]
            count += num_acc
            
    return u

def get_flow_torque(u, g_sym, g_anti, p_val):
    """
    Calculate angular velocity Omega due to flow (Equation 3).
    Omega = u x { [p^2/(p^2+1)] g.u  - [1/(p^2+1)] g^T.u }
    Rearranged in paper as:
    Omega = 0.5 * (p^2-1)/(p^2+1) * (u x (g_sym . u)) + 0.5 * (u x (g_anti . u))
    Actually Eq 3 simplifies to: 
    Omega = shape_factor * (u x (g_sym . u)) + vorticity_term
    where vorticity_term = 0.5 * curl(v) - 0.5 * (u . curl(v)) u
    which is equivalent to 0.5 * (u x (g_anti . u)) for rigid rotation.
    """
    # Shape factor B = (p^2 - 1) / (p^2 + 1)
    # p = aspect ratio. Rods > 1, Discs < 1.
    B = (p_val**2 - 1) / (p_val**2 + 1)
    
    # g_sym . u
    gu_s = u @ g_sym.T # (N,3)
    # g_anti . u
    gu_a = u @ g_anti.T
    
    # Cross products
    # u x (g_sym . u)
    term1 = np.cross(u, gu_s)
    # u x (g_anti . u)
    term2 = np.cross(u, gu_a)
    
    omega = 0.5 * B * term1 + 0.5 * term2 # The vorticity term 
    # Note: g_anti is defined such that g_anti_ij = 0.5(dv_i/dx_j - dv_j/dx_i)
    
    return omega

def get_potential_torque(u, local_director, U_strength):
    """
    Torque due to Maier-Saupe potential U = -0.5 * J * S * (u.n)^2
    Gamma = -dUd\u (rotational gradient) = u x grad U
    grad U = - J * S * (u.n) * n
    Gamma = u x ( - Strength * (u.n) * n ) = Strength * (u.n) * (n x u)
    """
    un = (u @ local_director) # Dot product (N,)
    # n x u
    nxu = np.cross(local_director, u) # (N,3)
    
    # Gamma needs to be broadcast
    gamma = U_strength * un[:, np.newaxis] * nxu
    return gamma

def run_simulation(args):
    # Parameters
    N = 10000
    dt = args.dt
    steps = args.steps
    p_val = 5.0 # Rod-like particles (aspect ratio 5)
    Dr = 1.0 # Rotational diffusion constant
    kT = 1.0
    friction = kT / Dr
    
    # Flow setup: Simple Shear v_x = gamma_dot * y
    # Velocity Gradient Tensor L
    gamma_dot = args.shear_rate
    L = np.zeros((3,3))
    L[0,1] = gamma_dot # dv_x / dy
    
    g_sym = 0.5 * (L + L.T)
    g_anti = 0.5 * (L - L.T)
    
    # Potential setup
    local_director = np.array([1.0, 0.0, 0.0]) # Along X
    U_strength = args.potential_strength # e.g., J*S2
    
    # Initialize
    if args.initial == 'uniform':
        u = initialize_distribution(N, 'uniform')
    else:
        u = initialize_distribution(N, 'watson', kappa=5.0, mu=np.array([0,0,1])) # Start Z-aligned
        
    # Storage for snapshots
    snapshots = {}
    snap_indices = [0, steps//2, steps-1]
    labels = ['Initial', 'Intermediate', 'Final']
    
    # Time Integration (Euler-Maruyama on Sphere)
    for t in range(steps):
        if t in snap_indices:
            idx = snap_indices.index(t)
            snapshots[labels[idx]] = u.copy()
            
        # Deterministic angular velocity
        omega_flow = get_flow_torque(u, g_sym, g_anti, p_val)
        
        # Potential torque -> angular velocity (overdamped: omega = Gamma/friction)
        gamma_pot = get_potential_torque(u, local_director, U_strength)
        omega_pot = gamma_pot / friction
        
        omega_det = omega_flow + omega_pot
        
        # Stochastic angular displacement d_theta
        # Variance = 2 * Dr * dt
        noise_mag = np.sqrt(2 * Dr * dt)
        # Random vector perpendicular to u? 
        # Easier: Random 3D vector, project out u component (tangent plane)
        rand_vec = np.random.normal(0, 1, (N, 3))
        # Project to tangent space: v_tan = v - (v.u)u
        vu = np.sum(rand_vec * u, axis=1)[:, np.newaxis]
        noise_vec = rand_vec - vu * u
        # Normalize noise direction and scale magnitude??
        # Actually, usually modeled as simply additive random rotation.
        
        # Total rotation vector
        rot_vec = omega_det * dt + noise_mag * noise_vec
        
        # Rotate u
        # For small angles, u_new = u + rot_vec x u
        delta_u = np.cross(rot_vec, u)
        u = u + delta_u
        
        # Renormalize to stay on sphere
        norms = np.linalg.norm(u, axis=1)
        u = u / norms[:, np.newaxis]

    # Save final snapshot
    snapshots['Final'] = u.copy()
    
    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    for i, label in enumerate(labels):
        data = snapshots[label]
        
        # Convert to Spherical Coordinates for Histograms
        # theta (polar): 0 to pi (angle from Z)
        # phi (azimuth): 0 to 2pi (angle in XY)
        theta = np.arccos(data[:, 2])
        phi = np.arctan2(data[:, 1], data[:, 0])
        
        # Plot Polar Angle Histogram
        ax_pol = axes[i, 0]
        ax_pol.hist(theta, bins=50, density=True, color='skyblue', edgecolor='black')
        ax_pol.set_title(f"{label}: Polar Angle (Theta)")
        ax_pol.set_xlim(0, np.pi)
        ax_pol.set_xlabel(r"$\theta$ (rad)")
        
        # Plot Azimuth Angle Histogram
        ax_azi = axes[i, 1]
        ax_azi.hist(phi, bins=50, density=True, color='salmon', edgecolor='black')
        ax_azi.set_title(f"{label}: Azimuth Angle (Phi)")
        ax_azi.set_xlim(-np.pi, np.pi)
        ax_azi.set_xlabel(r"$\phi$ (rad)")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "orientation_evolution.png"))
    print(f"Simulation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LC Orientation Distribution Evolution")
    parser.add_argument("--initial", type=str, default="uniform", choices=["uniform", "watson"], help="Initial distribution type")
    parser.add_argument("--shear_rate", type=float, default=0.0, help="Shear rate of flow")
    parser.add_argument("--potential_strength", type=float, default=0.0, help="Strength of local nematic potential")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    
    args = parser.parse_args()
    run_simulation(args)

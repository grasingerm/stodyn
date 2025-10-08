import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

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


# Example usage with your potential
def example_potential(x, y, V0=1.0, F=0.1):
    """Your fabric potential: V = V0*sin(x)*cos(y) - F*x"""
    return V0 * np.sin(x) * np.cos(y) - F * x


# Generate some example data (replace with your simulation data)
# For demonstration, I'll create a simple trajectory
t = np.linspace(0, 10, 1000)
x_example = 3 * np.sin(t) + 0.5 * t
y_example = 2 * np.cos(1.5 * t) + 0.2 * np.sin(3 * t)

# Create the plot
fig = plot_2d_trajectory_colored(x_example, y_example, 
                                  potential_func=lambda x, y: example_potential(x, y, V0=1.0, F=0.1))
plt.show()

# visualization.py
import numpy as np
import plotly.graph_objects as go
import geomstats.backend as gs
import config

def create_sphere_mesh():
    theta, phi = np.mgrid[0:np.pi:20j, 0:2*np.pi:40j]
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)
    return X, Y, Z

def create_tangent_plane_grid(mean_estimate, principal_directions):
    t1_vals = gs.linspace(-1, 1, 10)
    t2_vals = gs.linspace(-1, 1, 10)
    t1, t2 = gs.meshgrid(t1_vals, t2_vals)
    tangent_grid = (mean_estimate 
                    + t1[..., None] * principal_directions[0]
                    + t2[..., None] * principal_directions[1])
    X_tangent, Y_tangent, Z_tangent = tangent_grid[..., 0], tangent_grid[..., 1], tangent_grid[..., 2]
    return X_tangent, Y_tangent, Z_tangent

def create_pc_lines(mean_estimate, principal_directions):
    pc1_start = mean_estimate - config.SCALING_FACTOR_PC1 * principal_directions[0]
    pc1_end   = mean_estimate + config.SCALING_FACTOR_PC1 * principal_directions[0]
    pc2_start = mean_estimate - config.SCALING_FACTOR_PC2 * principal_directions[1]
    pc2_end   = mean_estimate + config.SCALING_FACTOR_PC2 * principal_directions[1]
    return (pc1_start, pc1_end), (pc2_start, pc2_end)

def build_figure(data, mean_estimate, X_sphere, Y_sphere, Z_sphere,
                 X_tangent, Y_tangent, Z_tangent, principal_directions,
                 reconstructed_points, full_reconstructed_points):
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_sphere, y=Y_sphere, z=Z_sphere,
        colorscale=config.COLORS["sphere"], opacity=config.OPACITIES["sphere"],
        showscale=False, name="Sphere"
    ))

    # Add original data points
    fig.add_trace(go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers', marker=dict(color=config.COLORS["data"], size=config.MARKER_SIZES["data"]),
        name="Data Points"
    ))

    fig.add_trace(go.Scatter3d(
        x=[mean_estimate[0]], y=[mean_estimate[1]], z=[mean_estimate[2]],
        mode='markers', marker=dict(color=config.COLORS["frechet_mean"], size=config.MARKER_SIZES["frechet_mean"]),
        name="Fréchet Mean"
    ))

    fig.add_trace(go.Surface(
        x=X_tangent, y=Y_tangent, z=Z_tangent,
        colorscale=config.COLORS["tangent_plane"], opacity=config.OPACITIES["tangent_plane"],
        showscale=False, name="Tangent Plane"
    ))

    (pc1_start, pc1_end), (pc2_start, pc2_end) = create_pc_lines(mean_estimate, principal_directions)
    fig.add_trace(go.Scatter3d(
        x=[pc1_start[0], pc1_end[0]],
        y=[pc1_start[1], pc1_end[1]],
        z=[pc1_start[2], pc1_end[2]],
        mode="lines", line=dict(color=config.COLORS["pc1"], width=config.LINE_WIDTHS["pc1"]),
        name="PC1"
    ))
    fig.add_trace(go.Scatter3d(
        x=[pc2_start[0], pc2_end[0]],
        y=[pc2_start[1], pc2_end[1]],
        z=[pc2_start[2], pc2_end[2]],
        mode="lines", line=dict(color=config.COLORS["pc2"], width=config.LINE_WIDTHS["pc2"]),
        name="PC2"
    ))

    # Add reconstructed points
    fig.add_trace(go.Scatter3d(
        x=reconstructed_points[:, 0],
        y=reconstructed_points[:, 1],
        z=reconstructed_points[:, 2],
        mode='markers', marker=dict(color=config.COLORS["reconstructed"], size=config.MARKER_SIZES["reconstructed"]),
        name="Reconstructed Points"
    ))

    fig.add_trace(go.Scatter3d(
        x=full_reconstructed_points[:, 0],
        y=full_reconstructed_points[:, 1],
        z=full_reconstructed_points[:, 2],
        mode='markers', marker=dict(color=config.COLORS["full_reconstruction"], size=config.MARKER_SIZES["full_reconstruction"]),
        name="Full Reconstruction (2 components)"
    ))

    fig.update_layout(
        title="Tangent Plane & PCA Reconstruction on Sphere (Interactive)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        margin=dict(l=0, r=0, b=0, t=40),
        width=config.FIG_WIDTH,
        height=config.FIG_HEIGHT
    )
    return fig

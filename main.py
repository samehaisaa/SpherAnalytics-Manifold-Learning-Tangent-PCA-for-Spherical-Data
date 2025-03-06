# main.py
import streamlit as st
import geomstats.backend as gs
from data import generate_data, compute_frechet_mean, compute_tangent_pca,generate_data_normal
from visualization import create_sphere_mesh, create_tangent_plane_grid, build_figure
import config

def main():
    st.sidebar.title("Input Parameters")
    n_samples = st.sidebar.slider("Number of Data Points", min_value=10, max_value=500,
                                  value=config.N_SAMPLES, step=10)
    num_components_retained = st.sidebar.selectbox("Number of Components to Retain", [1, 2])
    
    # Data generation & PCA
    sphere, data = generate_data_normal(n_samples, config.PRECISION)
    mean_estimate = compute_frechet_mean(sphere, data)
    tpca = compute_tangent_pca(sphere, data, mean_estimate)
    principal_directions = tpca.components_

    # Create tangent plane grid and sphere mesh
    X_tangent, Y_tangent, Z_tangent = create_tangent_plane_grid(mean_estimate, principal_directions)
    X_sphere, Y_sphere, Z_sphere = create_sphere_mesh()

    # Transform data to the tangent space
    tangent_projected_data = tpca.transform(data)
    if num_components_retained == 1:
        reconstructed_points = mean_estimate + tangent_projected_data[:, 0, None] * principal_directions[0]
    else:
        reconstructed_points = (mean_estimate
                                  + tangent_projected_data[:, 0, None] * principal_directions[0]
                                  + tangent_projected_data[:, 1, None] * principal_directions[1])
    # Full reconstruction (using both components)
    full_reconstructed_points = (mean_estimate
                                 + tangent_projected_data[:, 0, None] * principal_directions[0]
                                 + tangent_projected_data[:, 1, None] * principal_directions[1])

    # Build and display the Plotly figure
    fig = build_figure(data, mean_estimate, X_sphere, Y_sphere, Z_sphere,
                       X_tangent, Y_tangent, Z_tangent, principal_directions,
                       reconstructed_points, full_reconstructed_points)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

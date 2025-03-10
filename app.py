# app.py

import streamlit as st
import numpy as np
from data import generate_two_classes, compute_frechet_mean, compute_tangent_pca
from classification import compare_dimensionality_reduction, plot_reduced_data
from visualization import create_sphere_mesh, create_tangent_plane_grid, build_figure
from config import N_SAMPLES

st.title("Dimensionality Reduction Comparison on Spherical Data")

# Sidebar: Choose interweaving level
level = st.sidebar.selectbox("Select Interweaving Level", ["separate", "slight", "high"])

if st.button("Run Comparison"):
    # Generate data and perform comparison
    sphere, data, labels = generate_two_classes(n_samples=N_SAMPLES, interweaving_level=level)
    results = compare_dimensionality_reduction(sphere, data, labels, n_neighbors=5)
    
    st.subheader("Euclidean PCA Metrics")
    for metric, value in results['euclidean_pca']['metrics'].items():
        if metric != 'confusion_matrix':
            st.write(f"{metric.capitalize()}: {value:.4f}")
    
    st.subheader("Tangent PCA Metrics")
    for metric, value in results['tangent_pca']['metrics'].items():
        if metric != 'confusion_matrix':
            st.write(f"{metric.capitalize()}: {value:.4f}")
    
    # Plot the 2D reduced data for both methods
    all_pca_data = np.vstack([results['euclidean_pca']['reduced_train'], results['euclidean_pca']['reduced_test']])
    all_tpca_data = np.vstack([results['tangent_pca']['reduced_train'], results['tangent_pca']['reduced_test']])
    all_labels = labels  # Ensure the labels match the plotted data
    
    fig = plot_reduced_data(all_pca_data, all_tpca_data, all_labels, title=f"Dimensionality Reduction Comparison - {level.capitalize()}")
    st.pyplot(fig)
    
    # 3D Visualization using Plotly
    mean_estimate = compute_frechet_mean(sphere, data)
    tpca = compute_tangent_pca(sphere, data, mean_estimate)
    X_sphere, Y_sphere, Z_sphere = create_sphere_mesh()
    principal_directions = tpca.components_
    X_tangent, Y_tangent, Z_tangent = create_tangent_plane_grid(mean_estimate, principal_directions, scale=0.5)
    tangent_vectors = tpca.transform(data)
    reconstructed_points = tpca.inverse_transform(tangent_vectors)
    fig3d = build_figure(
        data, mean_estimate, X_sphere, Y_sphere, Z_sphere,
        X_tangent, Y_tangent, Z_tangent, principal_directions,
        reconstructed_points, reconstructed_points, labels
    )
    st.plotly_chart(fig3d)

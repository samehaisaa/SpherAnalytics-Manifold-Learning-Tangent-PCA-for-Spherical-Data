import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Correct geomstats imports
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA

# Set page configuration
st.set_page_config(page_title="Dimensionality Reduction Comparison", layout="wide")
st.title("Linear vs. Non-linear Dimensionality Reduction Comparison")

# Sidebar for parameters
st.sidebar.header("Data Generation Parameters")
n_samples = st.sidebar.slider("Number of samples per class", 50, 500, 200)
kappa = st.sidebar.slider("Concentration parameter (kappa)", 1.0, 50.0, 10.0, 1.0)
precision = st.sidebar.slider("Precision parameter", 1.0, 20.0, 5.0, 0.5)
interweaving = st.sidebar.slider("Class interweaving factor", 0.0, 1.0, 0.3, 0.05)
random_state = st.sidebar.slider("Random seed", 0, 100, 42)

# Function to generate data on a sphere with two classes using correct geomstats notation
def generate_spherical_data(n_samples, precision, interweaving, random_state=42):
    np.random.seed(random_state)
    
    # Create the sphere (2D manifold in 3D space)
    sphere = Hypersphere(dim=2)
    
    # Generate base points for two classes
    # Class 0 centered around a point
    random_mu_0 = np.array([1.0, 0.0, 0.0])  # Starting point
    
    # Class 1 centered around a different point
    # Adjust based on interweaving - higher interweaving means closer centers
    angle = np.pi * (1 - interweaving)
    random_mu_1 = np.array([np.cos(angle), np.sin(angle), 0.0])
    
    # Generate data using riemannian_normal distribution
    data_0 = sphere.random_riemannian_normal(
        mean=random_mu_0, 
        precision=precision, 
        n_samples=n_samples
    )
    
    data_1 = sphere.random_riemannian_normal(
        mean=random_mu_1,
        precision=precision,
        n_samples=n_samples
    )
    
    # Combine the data and create labels
    X = np.vstack([data_0, data_1])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    return sphere, X, y

# Function to perform dimensionality reduction and evaluate classification
def evaluate_dimensionality_reduction(sphere, X, y, method="PCA", n_components=2):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Calculate Frechet mean for TPCA
    mean = FrechetMean(sphere)
    mean.fit(X_train)
    base_point = mean.estimate_
    
    # Apply dimensionality reduction
    if method == "PCA":
        reducer = PCA(n_components=n_components)
        reducer.fit(X_train)
        X_train_reduced = reducer.transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        explained_variance = reducer.explained_variance_ratio_.sum()
        components = reducer.components_  # For visualization of the PCA plane
    elif method == "TPCA":
        reducer = TangentPCA(sphere, n_components=n_components)
        reducer.fit(X_train, base_point=base_point)
        X_train_reduced = reducer.transform(X_train, base_point=base_point)
        X_test_reduced = reducer.transform(X_test, base_point=base_point)
        explained_variance = reducer.explained_variance_ratio_.sum()
        components = reducer.components_  # For visualization of the TPCA plane
    
    # Train a classifier on the reduced data
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_reduced, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, y_pred)
    
    return X_train_reduced, X_test_reduced, y_train, y_test, accuracy, explained_variance, base_point, components, reducer

# Generate the data based on parameters
if st.sidebar.button("Generate New Data"):
    st.session_state.sphere, st.session_state.X, st.session_state.y = generate_spherical_data(
        n_samples, precision, interweaving, random_state
    )
    st.session_state.data_generated = True

# Initialize session state if needed
if 'data_generated' not in st.session_state:
    st.session_state.sphere, st.session_state.X, st.session_state.y = generate_spherical_data(
        n_samples, precision, interweaving, random_state
    )
    st.session_state.data_generated = True

# Dimensionality reduction parameters
st.sidebar.header("Dimensionality Reduction Parameters")
n_components = st.sidebar.slider("Number of components for reduction", 2, 2, 2)  # Fixed to 2 for visualization

# Perform the dimensionality reduction
if st.session_state.data_generated:
    sphere, X, y = st.session_state.sphere, st.session_state.X, st.session_state.y
    
    # Apply both methods
    (X_train_pca, X_test_pca, y_train_pca, y_test_pca, pca_accuracy, pca_var, 
     _, pca_components, pca_model) = evaluate_dimensionality_reduction(
        sphere, X, y, method="PCA", n_components=n_components
    )
    
    (X_train_tpca, X_test_tpca, y_train_tpca, y_test_tpca, tpca_accuracy, tpca_var, 
     base_point, tpca_components, tpca_model) = evaluate_dimensionality_reduction(
        sphere, X, y, method="TPCA", n_components=n_components
    )
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Linear PCA")
        st.metric("Classification Accuracy", f"{pca_accuracy:.4f}")
        st.metric("Explained Variance", f"{pca_var:.4f}")
        
        # Visualize PCA results
        fig_pca = px.scatter(
            x=X_train_pca[:, 0], 
            y=X_train_pca[:, 1],
            color=np.array(["Class 0" if c == 0 else "Class 1" for c in y_train_pca]),
            title="PCA Projection (2D)",
            labels={"x": "PC1", "y": "PC2", "color": "Class"}
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    
    with col2:
        st.header("Tangent PCA")
        st.metric("Classification Accuracy", f"{tpca_accuracy:.4f}")
        st.metric("Explained Variance", f"{tpca_var:.4f}")
        
        # Visualize TPCA results
        fig_tpca = px.scatter(
            x=X_train_tpca[:, 0], 
            y=X_train_tpca[:, 1],
            color=np.array(["Class 0" if c == 0 else "Class 1" for c in y_train_tpca]),
            title="Tangent PCA Projection (2D)",
            labels={"x": "TPCA1", "y": "TPCA2", "color": "Class"}
        )
        st.plotly_chart(fig_tpca, use_container_width=True)
    
    # Visualization of original data in 3D with PCA and TPCA planes
    st.header("Original Data with PCA and TPCA Planes")
    
    # Create figure for 3D visualization
    fig_planes = go.Figure()
    
    # Add data points
    fig_planes.add_trace(
        go.Scatter3d(
            x=X[:, 0], 
            y=X[:, 1], 
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=['blue' if c == 0 else 'red' for c in y],
                opacity=0.8
            ),
            name='Data Points'
        )
    )
    
    # Add a unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig_planes.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            opacity=0.2,
            showscale=False,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            name='Unit Sphere'
        )
    )
    
    # Add Frechet mean point
    fig_planes.add_trace(
        go.Scatter3d(
            x=[base_point[0]],
            y=[base_point[1]],
            z=[base_point[2]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name='Frechet Mean'
        )
    )
    
    # Create and add PCA plane
    # For standard PCA, we need to create a plane passing through the data mean
    # and spanned by the first two principal components
    data_mean = X.mean(axis=0)
    
    # Get PCA components
    pc1 = pca_components[0]
    pc2 = pca_components[1]
    
    # Create a grid of points on the PCA plane
    grid_size = 10
    grid_scale = 1.5
    
    pca_plane_points = np.zeros((grid_size**2, 3))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Create points in the PCA plane
            coef1 = (i - grid_size/2) * grid_scale / (grid_size/2)
            coef2 = (j - grid_size/2) * grid_scale / (grid_size/2)
            pca_plane_points[idx] = data_mean + coef1 * pc1 + coef2 * pc2
            idx += 1
    
    # Create a mesh for the PCA plane
    pca_plane_x = pca_plane_points[:, 0].reshape(grid_size, grid_size)
    pca_plane_y = pca_plane_points[:, 1].reshape(grid_size, grid_size)
    pca_plane_z = pca_plane_points[:, 2].reshape(grid_size, grid_size)
    
    fig_planes.add_trace(
        go.Surface(
            x=pca_plane_x,
            y=pca_plane_y,
            z=pca_plane_z,
            opacity=0.6,
            showscale=False,
            colorscale=[[0, 'rgba(255,0,0,0.7)'], [1, 'rgba(255,0,0,0.7)']],
            name='PCA Plane'
        )
    )
    
    # Create and add TPCA plane (tangent at the Frechet mean)
    # Get TPCA components in ambient space
    tpc1 = tpca_components[0]
    tpc2 = tpca_components[1]
    
    # Make sure these vectors are tangent to the sphere at base_point
    # (They should be orthogonal to the base_point)
    tpc1 = tpc1 - np.dot(tpc1, base_point) * base_point
    tpc1 = tpc1 / np.linalg.norm(tpc1)
    
    tpc2 = tpc2 - np.dot(tpc2, base_point) * base_point
    tpc2 = tpc2 / np.linalg.norm(tpc2)
    
    # Create a grid of points on the TPCA plane
    tpca_plane_points = np.zeros((grid_size**2, 3))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Create points in the TPCA plane
            coef1 = (i - grid_size/2) * grid_scale / (grid_size/2)
            coef2 = (j - grid_size/2) * grid_scale / (grid_size/2)
            tpca_plane_points[idx] = base_point + coef1 * tpc1 + coef2 * tpc2
            idx += 1
    
    # Create a mesh for the TPCA plane
    tpca_plane_x = tpca_plane_points[:, 0].reshape(grid_size, grid_size)
    tpca_plane_y = tpca_plane_points[:, 1].reshape(grid_size, grid_size)
    tpca_plane_z = tpca_plane_points[:, 2].reshape(grid_size, grid_size)
    
    fig_planes.add_trace(
        go.Surface(
            x=tpca_plane_x,
            y=tpca_plane_y,
            z=tpca_plane_z,
            opacity=0.6,
            showscale=False,
            colorscale=[[0, 'rgba(0,0,255,0.7)'], [1, 'rgba(0,0,255,0.7)']],
            name='TPCA Plane'
        )
    )
    
    # Update layout
    fig_planes.update_layout(
        title="Comparison of PCA and TPCA Planes",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube'
        ),
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_planes, use_container_width=True)
    
    # Add explanation about the planes
    st.write("""
    ### Understanding the Visualization:
    
    In the 3D plot above, you can see:
    
    - **Blue and Red Points**: The original data points from the two classes distributed on the sphere
    - **Gray Surface**: The unit sphere manifold where the data naturally lives
    - **Green Point**: The Frechet mean of the data on the sphere
    - **Red Surface**: The PCA plane, which is the best-fit plane in Euclidean space
    - **Blue Surface**: The TPCA plane, which is tangent to the sphere at the Frechet mean
    
    The key difference:
    - The **PCA plane** cuts through the sphere, ignoring its curvature
    - The **TPCA plane** is tangent to the sphere, respecting the local geometry
    
    This visualization shows why Tangent PCA often performs better for data that naturally lives on 
    manifolds - it projects data onto a plane that respects the underlying geometry.
    """)
    
    # Comparison bar chart
    st.header("Method Comparison")
    comparison_data = {
        "Method": ["PCA", "Tangent PCA"],
        "Accuracy": [pca_accuracy, tpca_accuracy],
        "Explained Variance": [pca_var, tpca_var]
    }
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        x=comparison_data["Method"],
        y=comparison_data["Accuracy"],
        name="Accuracy",
        marker_color="blue"
    ))
    fig_comparison.add_trace(go.Bar(
        x=comparison_data["Method"],
        y=comparison_data["Explained Variance"],
        name="Explained Variance",
        marker_color="red"
    ))
    
    fig_comparison.update_layout(
        title="Comparison of Dimensionality Reduction Methods",
        xaxis_title="Method",
        yaxis_title="Score",
        barmode="group"
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Additional analysis
    st.header("Analysis")
    better_method = "Tangent PCA" if tpca_accuracy > pca_accuracy else "PCA"
    difference = abs(tpca_accuracy - pca_accuracy)
    
    st.write(f"Based on classification accuracy, **{better_method}** performs better by {difference:.4f} for this dataset.")
    
    st.write("""
    ### Why the planes are different:
    
    1. **PCA Plane**:
       - Standard PCA finds the directions of maximum variance in Euclidean space
       - The resulting plane minimizes the squared Euclidean distances to data points
       - Ignores the fact that the data lives on a curved manifold
       - Can introduce distortions when projecting spherical data
    
    2. **Tangent PCA Plane**:
       - First computes the Frechet mean, which is the "center" of the data on the sphere
       - Creates a tangent plane at this point (the green point in the visualization)
       - Projects data onto this tangent plane using the logarithmic map
       - Preserves local geometric relationships between points
       - Respects the manifold structure of the data
    
    As the interweaving parameter increases and classes become more mixed on the sphere,
    the advantage of using Tangent PCA becomes more apparent.
    """)
    
    # Download options
    st.header("Download Results")
    
    # Create CSV strings for download
    pca_results = np.hstack([X_test_pca, y_test_pca.reshape(-1, 1)])
    tpca_results = np.hstack([X_test_tpca, y_test_tpca.reshape(-1, 1)])
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download PCA Results",
            data="\n".join([",".join(map(str, row)) for row in pca_results]),
            file_name="pca_results.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="Download TPCA Results",
            data="\n".join([",".join(map(str, row)) for row in tpca_results]),
            file_name="tpca_results.csv",
            mime="text/csv"
        )
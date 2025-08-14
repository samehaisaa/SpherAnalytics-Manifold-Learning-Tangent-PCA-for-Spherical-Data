# classification.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from geomstats.learning.pca import TangentPCA
from data import compute_frechet_mean

def split_data(data, labels, test_size=0.3, random_state=42):
    return train_test_split(data, labels, test_size=test_size, random_state=random_state, stratify=labels)

def train_knn_classifier(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    return plt.gcf()

def plot_decision_boundary(X, y, classifier, title):
    plt.figure(figsize=(10, 8))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter)
    plt.tight_layout()
    return plt.gcf()

def perform_euclidean_pca(data, n_components=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

def perform_tangent_pca(sphere, data, base_point, n_components=2):
    tpca = TangentPCA(sphere, n_components=n_components)
    tpca.fit(data, base_point=base_point)
    tangent_projected = tpca.transform(data)
    return tangent_projected, tpca

def compare_dimensionality_reduction(sphere, data, labels, n_neighbors=5, test_size=0.3):
    X_train_3d, X_test_3d, y_train, y_test = split_data(data, labels, test_size=test_size)
    mean_estimate = compute_frechet_mean(sphere, X_train_3d)
    
    # Euclidean PCA
    X_train_pca, pca = perform_euclidean_pca(X_train_3d, n_components=2)
    X_test_pca = pca.transform(X_test_3d)
    knn_pca = train_knn_classifier(X_train_pca, y_train, n_neighbors=n_neighbors)
    pca_metrics = evaluate_classifier(knn_pca, X_test_pca, y_test)
    
    # Tangent PCA
    X_train_tpca, tpca = perform_tangent_pca(sphere, X_train_3d, mean_estimate, n_components=2)
    X_test_tpca = tpca.transform(X_test_3d)
    knn_tpca = train_knn_classifier(X_train_tpca, y_train, n_neighbors=n_neighbors)
    tpca_metrics = evaluate_classifier(knn_tpca, X_test_tpca, y_test)
    
    # Create visualizations
    pca_cm_fig = plot_confusion_matrix(pca_metrics['confusion_matrix'], 'Euclidean PCA')
    tpca_cm_fig = plot_confusion_matrix(tpca_metrics['confusion_matrix'], 'Tangent PCA')
    pca_decision_fig = plot_decision_boundary(
        np.vstack([X_train_pca, X_test_pca]), 
        np.concatenate([y_train, y_test]),
        knn_pca, 
        'Decision Boundary - Euclidean PCA'
    )
    tpca_decision_fig = plot_decision_boundary(
        np.vstack([X_train_tpca, X_test_tpca]), 
        np.concatenate([y_train, y_test]),
        knn_tpca, 
        'Decision Boundary - Tangent PCA'
    )
    
    results = {
        'euclidean_pca': {
            'metrics': pca_metrics,
            'reduced_train': X_train_pca,
            'reduced_test': X_test_pca,
            'model': knn_pca,
            'confusion_matrix_fig': pca_cm_fig,
            'decision_boundary_fig': pca_decision_fig
        },
        'tangent_pca': {
            'metrics': tpca_metrics,
            'reduced_train': X_train_tpca,
            'reduced_test': X_test_tpca,
            'model': knn_tpca,
            'confusion_matrix_fig': tpca_cm_fig,
            'decision_boundary_fig': tpca_decision_fig
        }
    }
    
    return results

def plot_reduced_data(pca_data, tpca_data, labels, title="Dimensionality Reduction Comparison"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = ax1.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    ax1.set_title("Euclidean PCA")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    scatter2 = ax2.scatter(tpca_data[:, 0], tpca_data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    ax2.set_title("Tangent PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.colorbar(scatter1, ax=ax1, label="Class")
    plt.colorbar(scatter2, ax=ax2, label="Class")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

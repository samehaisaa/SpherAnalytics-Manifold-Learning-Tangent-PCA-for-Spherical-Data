# evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from data import generate_two_classes, compute_frechet_mean
from classification import perform_euclidean_pca, perform_tangent_pca, split_data
from visualization_comparison import plot_dimensionality_reduction_comparison
import os
import json

def parameter_tuning(X_train, y_train, param_grid=None, cv=5):
    if param_grid is None:
        param_grid = {'n_neighbors': range(1, 21, 2)}
    
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring=scorers,
        refit='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

def compare_methods_across_parameters(interweaving_levels=['separate', 'slight', 'high'], 
                                     n_samples=200, n_components_range=[1, 2], 
                                     n_neighbors_range=[3, 5, 7, 9]):
    results = {}
    if not os.path.exists("results"):
        os.makedirs("results")

    for level in interweaving_levels:
        print(f"\n======= Evaluating {level} interweaving level =======")
        results[level] = {}
        sphere, data, labels = generate_two_classes(n_samples=n_samples, interweaving_level=level)
        mean_estimate = compute_frechet_mean(sphere, data)
        X_train, X_test, y_train, y_test = split_data(data, labels)
        
        for n_components in n_components_range:
            print(f"  Testing with {n_components} components")
            results[level][f'components_{n_components}'] = {}
            X_train_pca, pca = perform_euclidean_pca(X_train, n_components=n_components)
            X_test_pca = pca.transform(X_test)
            X_train_tpca, tpca = perform_tangent_pca(sphere, X_train, mean_estimate, n_components=n_components)
            X_test_tpca = tpca.transform(X_test)
            
            param_grid = {'n_neighbors': n_neighbors_range}
            grid_search_pca = parameter_tuning(X_train_pca, y_train, param_grid)
            grid_search_tpca = parameter_tuning(X_train_tpca, y_train, param_grid)
            
            best_k_pca = grid_search_pca.best_params_['n_neighbors']
            best_k_tpca = grid_search_tpca.best_params_['n_neighbors']
            
            best_knn_pca = KNeighborsClassifier(n_neighbors=best_k_pca)
            best_knn_pca.fit(X_train_pca, y_train)
            best_knn_tpca = KNeighborsClassifier(n_neighbors=best_k_tpca)
            best_knn_tpca.fit(X_train_tpca, y_train)
            
            y_pred_pca = best_knn_pca.predict(X_test_pca)
            y_pred_tpca = best_knn_tpca.predict(X_test_tpca)
            
            pca_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_pca),
                'precision': precision_score(y_test, y_pred_pca),
                'recall': recall_score(y_test, y_pred_pca),
                'f1': f1_score(y_test, y_pred_pca),
                'best_k': best_k_pca
            }
            tpca_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_tpca),
                'precision': precision_score(y_test, y_pred_tpca),
                'recall': recall_score(y_test, y_pred_tpca),
                'f1': f1_score(y_test, y_pred_tpca),
                'best_k': best_k_tpca
            }
            
            results[level][f'components_{n_components}']['euclidean'] = pca_metrics
            results[level][f'components_{n_components}']['tangent'] = tpca_metrics
            
            print(f"    Euclidean PCA - Best k: {best_k_pca}, Accuracy: {pca_metrics['accuracy']:.4f}")
            print(f"    Tangent PCA - Best k: {best_k_tpca}, Accuracy: {tpca_metrics['accuracy']:.4f}")
    
    with open('results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def plot_comprehensive_results(results):
    interweaving_levels = list(results.keys())
    for n_components in [1, 2]:
        euclidean_metrics = {}
        tangent_metrics = {}
        for level in interweaving_levels:
            euclidean_metrics[level] = results[level][f'components_{n_components}']['euclidean']
            tangent_metrics[level] = results[level][f'components_{n_components}']['tangent']
        fig = plot_dimensionality_reduction_comparison(euclidean_metrics, tangent_metrics, interweaving_levels)
        fig.suptitle(f'Performance Comparison with {n_components} Components', fontsize=16)
        fig.savefig(f'results/comparison_components_{n_components}.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(interweaving_levels))
    width = 0.2
    for i, n_components in enumerate([1, 2]):
        euc_k = [results[level][f'components_{n_components}']['euclidean']['best_k'] for level in interweaving_levels]
        tan_k = [results[level][f'components_{n_components}']['tangent']['best_k'] for level in interweaving_levels]
        plt.bar(x + (i-0.5)*width, euc_k, width, label=f'Euclidean ({n_components} comp)')
        plt.bar(x + (i+0.5)*width, tan_k, width, label=f'Tangent ({n_components} comp)')
    plt.title('Optimal Number of Neighbors (k) by Method')
    plt.xticks(x, [level.capitalize() for level in interweaving_levels])
    plt.ylabel('Best k Value')
    plt.xlabel('Interweaving Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/best_k_values.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    for i, n_components in enumerate([1, 2]):
        diff = [
            results[level][f'components_{n_components}']['tangent']['accuracy'] - 
            results[level][f'components_{n_components}']['euclidean']['accuracy']
            for level in interweaving_levels
        ]
        plt.plot(interweaving_levels, diff, 'o-', linewidth=2, label=f'{n_components} Components')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    plt.title('Tangent PCA Advantage over Euclidean PCA (Accuracy)')
    plt.xlabel('Interweaving Level')
    plt.ylabel('Accuracy Difference (Tangent - Euclidean)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('results/accuracy_difference.png', dpi=300, bbox_inches='tight')

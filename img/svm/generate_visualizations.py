#!/usr/bin/env python3
"""
SVM Visualization Generator
Generate comprehensive visualizations for SVM article
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# 1. Support Vectors Visualization (Hard vs Soft Margin)
# ============================================================================
def plot_support_vectors():
    """
    Visualize hard margin vs soft margin SVM with support vectors highlighted
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate linearly separable data
    np.random.seed(42)
    X_separable = np.vstack([
        np.random.randn(30, 2) + [2, 2],
        np.random.randn(30, 2) + [-2, -2]
    ])
    y_separable = np.array([1]*30 + [-1]*30)
    
    # Generate not separable data (with noise)
    X_noisy = X_separable.copy()
    # Add some noise points
    X_noisy[28] = [-2, 2]  # Misplaced point in positive class
    X_noisy[58] = [2, -2]  # Misplaced point in negative class
    y_noisy = y_separable.copy()
    
    # Standardize
    scaler = StandardScaler()
    X_separable = scaler.fit_transform(X_separable)
    X_noisy = scaler.fit_transform(X_noisy)
    
    # Hard Margin SVM
    svm_hard = SVC(kernel='linear', C=1e10)  # Very large C for hard margin
    svm_hard.fit(X_separable, y_separable)
    
    ax = axes[0]
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X_separable[:, 0].min()-0.5, X_separable[:, 0].max()+0.5, 100),
                         np.linspace(X_separable[:, 1].min()-0.5, X_separable[:, 1].max()+0.5, 100))
    Z = svm_hard.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
    
    # Plot points
    ax.scatter(X_separable[y_separable==1, 0], X_separable[y_separable==1, 1], 
              c='red', marker='o', s=100, edgecolors='k', label='Class +1')
    ax.scatter(X_separable[y_separable==-1, 0], X_separable[y_separable==-1, 1], 
              c='blue', marker='s', s=100, edgecolors='k', label='Class -1')
    
    # Highlight support vectors
    ax.scatter(svm_hard.support_vectors_[:, 0], svm_hard.support_vectors_[:, 1],
              s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
    
    ax.set_title('Hard Margin SVM\n(Linearly Separable Data)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    # Soft Margin SVM
    svm_soft = SVC(kernel='linear', C=1)  # Smaller C for soft margin
    svm_soft.fit(X_noisy, y_noisy)
    
    ax = axes[1]
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(X_noisy[:, 0].min()-0.5, X_noisy[:, 0].max()+0.5, 100),
                         np.linspace(X_noisy[:, 1].min()-0.5, X_noisy[:, 1].max()+0.5, 100))
    Z = svm_soft.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
    
    # Plot points
    colors = ['red' if y==1 else 'blue' for y in y_noisy]
    ax.scatter(X_noisy[:, 0], X_noisy[:, 1], c=colors, marker='o', s=100, 
              edgecolors='k', alpha=0.7)
    
    # Mark misclassified points
    misclassified = svm_soft.predict(X_noisy) != y_noisy
    ax.scatter(X_noisy[misclassified, 0], X_noisy[misclassified, 1],
              marker='X', s=250, edgecolors='red', facecolors='none', linewidths=2,
              label='Misclassified')
    
    # Highlight support vectors
    ax.scatter(svm_soft.support_vectors_[:, 0], svm_soft.support_vectors_[:, 1],
              s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
    
    ax.set_title('Soft Margin SVM\n(Non-linearly Separable Data)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='best')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    plt.savefig('01_support_vectors.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 01_support_vectors.png")
    plt.close()


# ============================================================================
# 2. Kernel Functions Comparison
# ============================================================================
def plot_kernel_comparison():
    """
    Compare different kernel functions on moons dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    # Generate non-linearly separable data
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.1)
    X = StandardScaler().fit_transform(X)
    y[y==0] = -1  # Convert to -1/+1 labels
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    params = [
        {},
        {'degree': 3, 'coef0': 1},
        {'gamma': 'scale'},
        {}
    ]
    titles = [
        'Linear Kernel\n$K(x_i, x_j) = x_i^T x_j$',
        'Polynomial Kernel (d=3)\n$K(x_i, x_j) = (x_i^T x_j + 1)^3$',
        'RBF Kernel\n$K(x_i, x_j) = \\exp(-\\gamma ||x_i - x_j||^2)$',
        'Sigmoid Kernel\n$K(x_i, x_j) = \\tanh(x_i^T x_j + 1)$'
    ]
    
    for idx, (kernel, param, title) in enumerate(zip(kernels, params, titles)):
        ax = axes[idx]
        
        # Train SVM with specific kernel
        svm = SVC(kernel=kernel, C=1, **param)
        svm.fit(X, y)
        
        # Create mesh
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), 
                   cmap=plt.cm.RdBu, alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
        
        # Plot points
        scatter = ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', 
                            s=80, edgecolors='k', alpha=0.8, label='Class +1')
        ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', 
                  s=80, edgecolors='k', alpha=0.8, label='Class -1')
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=150, linewidth=1, facecolors='none', edgecolors='green', 
                  label=f'Support Vectors (n={len(svm.support_vectors_)})')
        
        # Accuracy
        accuracy = svm.score(X, y)
        
        ax.set_title(f'{title}\nAccuracy: {accuracy:.1%}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    plt.savefig('02_kernel_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 02_kernel_comparison.png")
    plt.close()


# ============================================================================
# 3. Effect of C Parameter (Regularization)
# ============================================================================
def plot_c_parameter_effect():
    """
    Show how C parameter affects decision boundary
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    # Generate data with noise
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [1.5, 1.5],
        np.random.randn(20, 2) + [-1.5, -1.5]
    ])
    y = np.array([1]*20 + [-1]*20)
    # Add some noise
    X[[10, 35]] = [[-1.5, 1.5], [1.5, -1.5]]
    
    X = StandardScaler().fit_transform(X)
    
    C_values = [0.1, 1, 10, 100]
    
    for idx, C in enumerate(C_values):
        ax = axes[idx]
        
        # Train SVM
        svm = SVC(kernel='rbf', C=C, gamma='scale')
        svm.fit(X, y)
        
        # Create mesh
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), 
                   cmap=plt.cm.RdBu, alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
        
        # Plot points
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', 
                  s=100, edgecolors='k', alpha=0.8)
        ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', 
                  s=100, edgecolors='k', alpha=0.8)
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=200, linewidth=1.5, facecolors='none', edgecolors='green')
        
        # Misclassified points
        misclassified = svm.predict(X) != y
        if misclassified.any():
            ax.scatter(X[misclassified, 0], X[misclassified, 1],
                      marker='X', s=250, edgecolors='red', facecolors='none', linewidths=2)
        
        ax.set_title(f'C = {C}\nSupport Vectors: {len(svm.support_vectors_)}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        if C == 0.1:
            ax.text(0.05, 0.95, 'Underfitting\n(allows more errors)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        elif C == 100:
            ax.text(0.05, 0.95, 'Overfitting\n(strict margin)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('03_c_parameter_effect.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 03_c_parameter_effect.png")
    plt.close()


# ============================================================================
# 4. Gamma Parameter Effect (RBF Kernel)
# ============================================================================
def plot_gamma_parameter_effect():
    """
    Show how gamma parameter affects RBF kernel
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    # Generate data
    np.random.seed(42)
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3)
    X = StandardScaler().fit_transform(X)
    y[y==0] = -1
    
    gamma_values = [0.1, 1, 10, 100]
    
    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx]
        
        # Train SVM
        svm = SVC(kernel='rbf', C=1, gamma=gamma)
        svm.fit(X, y)
        
        # Create mesh
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), 
                   cmap=plt.cm.RdBu, alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors='black', linestyles='dashed')
        
        # Plot points
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', 
                  s=80, edgecolors='k', alpha=0.8)
        ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', 
                  s=80, edgecolors='k', alpha=0.8)
        
        # Highlight support vectors
        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=150, linewidth=1, facecolors='none', edgecolors='green')
        
        accuracy = svm.score(X, y)
        ax.set_title(f'$\\gamma$ = {gamma}\nAccuracy: {accuracy:.1%}, SV: {len(svm.support_vectors_)}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        if gamma == 0.1:
            ax.text(0.05, 0.95, 'Large influence radius\n(smooth boundary)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        elif gamma == 100:
            ax.text(0.05, 0.95, 'Small influence radius\n(complex boundary)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('04_gamma_parameter_effect.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 04_gamma_parameter_effect.png")
    plt.close()


# ============================================================================
# 5. Multi-class Classification Methods
# ============================================================================
def plot_multiclass_methods():
    """
    Visualize One-vs-Rest and One-vs-One approaches
    """
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate 3-class data
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(30, 2) + [2, 2],
        np.random.randn(30, 2) + [-2, 2],
        np.random.randn(30, 2) + [0, -2.5]
    ])
    y = np.array([0]*30 + [1]*30 + [2]*30)
    X = StandardScaler().fit_transform(X)
    
    methods = [
        ('One-vs-Rest (OvR)', OneVsRestClassifier(SVC(kernel='rbf', C=1))),
        ('One-vs-One (OvO)', OneVsOneClassifier(SVC(kernel='rbf', C=1)))
    ]
    
    for idx, (method_name, clf) in enumerate(methods):
        ax = axes[idx]
        
        clf.fit(X, y)
        
        # Create mesh
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                             np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        colors_region = ['#ffcccc', '#ccffcc', '#ccccff']
        for i in range(3):
            ax.contourf(xx, yy, Z, levels=[i, i+0.99], colors=[colors_region[i]], alpha=0.6)
        
        # Plot decision boundaries
        ax.contour(xx, yy, Z, levels=[0.5, 1.5], linewidths=2, colors='black')
        
        # Plot points
        colors = ['red', 'green', 'blue']
        markers = ['o', 's', '^']
        for i in range(3):
            ax.scatter(X[y==i, 0], X[y==i, 1], c=colors[i], marker=markers[i], 
                      s=100, edgecolors='k', label=f'Class {i}')
        
        accuracy = clf.score(X, y)
        n_classifiers = len(clf.estimators_) if hasattr(clf, 'estimators_') else 3
        
        ax.set_title(f'{method_name}\nAccuracy: {accuracy:.1%}, # Classifiers: {n_classifiers}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(loc='best')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    plt.savefig('05_multiclass_methods.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 05_multiclass_methods.png")
    plt.close()


# ============================================================================
# 6. Confusion Matrix and Performance Metrics
# ============================================================================
def plot_confusion_matrix():
    """
    Show confusion matrix for spam classification
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Simulate spam classifier results
    # Case 1: C=10 (original)
    cm1 = np.array([[95, 45], [3, 57]])
    
    # Case 2: C=100 (improved)
    cm2 = np.array([[128, 12], [2, 58]])
    
    cases = [('Original (C=10)', cm1), ('Improved (C=100)', cm2)]
    
    for idx, (title, cm) in enumerate(cases):
        ax = axes[idx]
        
        # Normalize for percentage display
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Plot heatmap
        sns.heatmap(cm, annot=np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                                        for j in range(2)] for i in range(2)]),
                   fmt='', cmap='Blues', cbar=False, ax=ax,
                   xticklabels=['Predicted\nNormal', 'Predicted\nSpam'],
                   yticklabels=['Actual\nNormal', 'Actual\nSpam'],
                   annot_kws={'fontsize': 11})
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / cm.sum()
        
        metrics_text = f'Accuracy: {accuracy:.1%}\nPrecision: {precision:.1%}\n'
        metrics_text += f'Recall: {recall:.1%}\nF1-Score: {f1:.1%}'
        
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.text(1.2, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('06_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 06_confusion_matrix.png")
    plt.close()


# ============================================================================
# 7. Support Vectors in Feature Space
# ============================================================================
def plot_feature_space_visualization():
    """
    Show how data looks in original vs feature space (conceptual)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate 2D data that's not linearly separable
    np.random.seed(42)
    X_pos = np.random.randn(40, 2) + [1.5, 0]
    X_neg = np.random.randn(40, 2) + [-1.5, 0]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*40 + [-1]*40)
    
    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X, y)
    
    # Left plot: Original feature space
    ax = axes[0]
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', s=100, 
              edgecolors='k', alpha=0.8, label='Class +1')
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', s=100, 
              edgecolors='k', alpha=0.8, label='Class -1')
    
    # Decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20), 
               cmap=plt.cm.RdBu, alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    
    # Support vectors
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=250, linewidth=2, facecolors='none', edgecolors='green', 
              label=f'Support Vectors (n={len(svm.support_vectors_)})')
    
    ax.set_title('Original Feature Space\n(Non-linearly Separable)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1: $x_1$')
    ax.set_ylabel('Feature 2: $x_2$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Right plot: Conceptual high-dimensional space
    ax = axes[1]
    ax.text(0.5, 0.95, 'Implicit High-Dimensional Feature Space', 
           ha='center', va='top', transform=ax.transAxes, fontsize=11, fontweight='bold')
    ax.text(0.5, 0.75, 'RBF Kernel: $K(x_i, x_j) = \\exp(-\\gamma ||x_i - x_j||^2)$', 
           ha='center', va='top', transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.text(0.5, 0.55, 'After kernel transformation:\n• Data becomes linearly separable\n'
                       '• Dimension ≈ ∞ (but never explicitly computed)\n'
                       '• Kernel trick: compute inner product without explicit mapping\n'
                       '• Support vectors: key points determining the hyperplane',
           ha='center', va='top', transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Draw conceptual separation
    circle = patches.Circle((0.5, 0.15), 0.2, transform=ax.transAxes, 
                           fill=True, alpha=0.3, color='red', label='Class +1')
    ax.add_patch(circle)
    rect = patches.Rectangle((0.65, 0.05), 0.25, 0.25, transform=ax.transAxes, 
                             fill=True, alpha=0.3, color='blue')
    ax.add_patch(rect)
    
    ax.text(0.5, 0.1, 'Linearly\nSeparable', ha='center', va='center', 
           transform=ax.transAxes, fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('07_feature_space_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 07_feature_space_visualization.png")
    plt.close()


# ============================================================================
# 8. Margin Visualization
# ============================================================================
def plot_margin_visualization():
    """
    Show the concept of margin and support vectors clearly
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Generate simple 2D data
    np.random.seed(42)
    X_pos = np.random.randn(15, 2) * 0.3 + [1.5, 1.5]
    X_neg = np.random.randn(15, 2) * 0.3 + [-1.5, -1.5]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*15 + [-1]*15)
    
    # Train SVM
    svm = SVC(kernel='linear', C=1e10)
    svm.fit(X, y)
    
    # Plot background
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, np.sign(Z), levels=[-1.5, 0, 1.5], 
               colors=['#ffcccc', '#ccccff'], alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], linewidths=3, colors='black', label='Decision Boundary')
    ax.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='gray', 
              linestyles='dashed', label='Margin Boundary')
    
    # Margin area
    margin_x = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    # Decision boundary: w·x + b = 0
    margin_y_center = (-w[0] * margin_x - b) / w[1]
    # Margin boundaries: w·x + b = ±1
    margin_y_upper = (-w[0] * margin_x - b - 1) / w[1]
    margin_y_lower = (-w[0] * margin_x - b + 1) / w[1]
    
    ax.fill_between(margin_x, margin_y_lower, margin_y_upper, alpha=0.2, 
                   color='green', label='Margin Area')
    
    # Plot data points
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', s=150, 
              edgecolors='darkred', linewidth=2, alpha=0.9, label='Class +1', zorder=5)
    ax.scatter(X[y==-1, 0], X[y==-1, 1], c='blue', marker='s', s=150, 
              edgecolors='darkblue', linewidth=2, alpha=0.9, label='Class -1', zorder=5)
    
    # Highlight support vectors
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
              s=400, linewidth=3, facecolors='none', edgecolors='green', 
              label=f'Support Vectors (n={len(svm.support_vectors_)})', zorder=10)
    
    # Add annotations for margin
    margin_distance = 1.0 / np.linalg.norm(w)
    mid_x = 0
    mid_y_center = (-w[0] * mid_x - b) / w[1]
    mid_y_up = mid_y_center + margin_distance * w[1] / np.linalg.norm(w) / 2
    mid_y_down = mid_y_center - margin_distance * w[1] / np.linalg.norm(w) / 2
    
    ax.annotate('', xy=(mid_x, mid_y_up), xytext=(mid_x, mid_y_down),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(mid_x + 0.15, mid_y_center, f'Margin = {margin_distance:.3f}', 
           fontsize=11, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
    ax.set_ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Maximum Margin Principle\nSupport Vectors Define the Optimal Hyperplane', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('08_margin_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 08_margin_visualization.png")
    plt.close()


# ============================================================================
# 9. Algorithm Complexity Comparison
# ============================================================================
def plot_complexity_analysis():
    """
    Show time/space complexity of different datasets
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time complexity
    ax = axes[0]
    dataset_sizes = np.array([100, 500, 1000, 5000, 10000])
    time_linear = 0.001 * dataset_sizes  # O(n)
    time_quadratic = 0.0001 * dataset_sizes**2  # O(n^2)
    time_cubic = 0.0000005 * dataset_sizes**2.5  # O(n^2.5) SMO
    
    ax.plot(dataset_sizes, time_linear, 'o-', linewidth=2, markersize=8, label='Linear: O(n)')
    ax.plot(dataset_sizes, time_quadratic, 's-', linewidth=2, markersize=8, label='Quadratic: O(n²)')
    ax.plot(dataset_sizes, time_cubic, '^-', linewidth=2, markersize=8, label='SMO: O(n² ~ n³)')
    
    ax.set_xlabel('Dataset Size (m)', fontsize=11)
    ax.set_ylabel('Training Time (seconds, log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Time Complexity Analysis\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Space complexity
    ax = axes[1]
    space_kernel_matrix = 8 * dataset_sizes**2 / (1024**2)  # MB
    space_weight_vector = 8 * 5000 / (1024**2)  # Fixed, assuming 5000 features
    
    ax.fill_between(dataset_sizes, 0, space_kernel_matrix, alpha=0.3, label='Kernel Matrix: O(m²)')
    ax.plot(dataset_sizes, space_kernel_matrix, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=space_weight_vector, color='orange', linestyle='--', linewidth=2, 
              label='Weight Vector: O(d)')
    
    ax.set_xlabel('Dataset Size (m)', fontsize=11)
    ax.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax.set_title('Space Complexity Analysis\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(8000, 150, f'At m=10k:\nKernel: {space_kernel_matrix[-1]:.1f} MB\nWeight: {space_weight_vector:.1f} MB',
           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('09_complexity_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 09_complexity_analysis.png")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating SVM visualizations...\n")
    
    plot_support_vectors()
    plot_kernel_comparison()
    plot_c_parameter_effect()
    plot_gamma_parameter_effect()
    plot_multiclass_methods()
    plot_confusion_matrix()
    plot_feature_space_visualization()
    plot_margin_visualization()
    plot_complexity_analysis()
    
    print("\n✓ All visualizations generated successfully!")
    print("  Images saved in: img/svm/")

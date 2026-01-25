"""
生成聚类算法的可视化图片
为无监督学习：聚类算法深度解析文章配图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# 创建输出目录路径
output_dir = '/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/clustering-algorithms'

# ============================================================================
# 辅助函数：PNG转WebP
# ============================================================================
def convert_png_to_webp(png_path, quality=85):
    """将PNG图片转换为WebP格式"""
    try:
        webp_path = png_path.replace('.png', '.webp')
        img = Image.open(png_path)
        
        # 转换为RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        # 保存为WebP
        img.save(webp_path, 'webp', quality=quality)
        print(f"  ✓ {os.path.basename(png_path)} → {os.path.basename(webp_path)}")
        return True
    except Exception as e:
        print(f"  ✗ 转换失败 {png_path}: {e}")
        return False


def batch_convert_to_webp(quality=85):
    """批量将PNG转换为WebP"""
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    if not png_files:
        print("没有PNG文件需要转换")
        return
    
    print(f"\n开始转换 {len(png_files)} 个PNG文件为WebP...")
    for png_file in png_files:
        png_path = os.path.join(output_dir, png_file)
        convert_png_to_webp(png_path, quality=quality)


# ============================================================================
# 1. K-Means 迭代过程可视化
# ============================================================================
def generate_kmeans_iteration():
    """生成K-Means迭代过程的图片"""
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('K-Means 聚类迭代过程', fontsize=16, fontweight='bold', y=0.98)
    
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], 3, replace=False)]
    
    iterations = [0, 1, 2, 5, 10, 20]
    
    for idx, iteration in enumerate(iterations):
        ax = axes[idx // 3, idx % 3]
        
        if iteration == 0:
            ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=30, c='gray', label='数据点')
            ax.scatter(centroids[:, 0], centroids[:, 1], c=['red', 'green', 'blue'], 
                      s=300, marker='*', edgecolors='black', linewidths=2, 
                      label='初始簇心')
            ax.set_title(f'初始化', fontsize=11, fontweight='bold')
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1, max_iter=iteration, 
                           random_state=42)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
            
            colors = ['red', 'green', 'blue']
            for k in range(3):
                mask = labels == k
                ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, 
                          s=30, label=f'Cluster {k+1}')
            
            ax.scatter(centroids[:, 0], centroids[:, 1], c=['red', 'green', 'blue'], 
                      s=300, marker='*', edgecolors='black', linewidths=2)
            
            ax.set_title(f'第 {iteration} 次迭代', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_kmeans_iteration.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 01_kmeans_iteration.png")
    plt.close()


# ============================================================================
# 2. 肘部法则（WCSS）可视化
# ============================================================================
def generate_elbow_method():
    """生成肘部法则图片"""
    
    from sklearn.cluster import KMeans
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8, label='WCSS')
    ax.plot(3, wcss[2], 'r*', markersize=20, label='最优K值 (K=3)')
    
    ax.annotate('肘部', xy=(3, wcss[2]), xytext=(4, wcss[2] + 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    ax.set_xlabel('簇数 K', fontsize=12, fontweight='bold')
    ax.set_ylabel('WCSS (类内距离平方和)', fontsize=12, fontweight='bold')
    ax.set_title('肘部法则：选择最优K值', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(k_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_elbow_method.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 02_elbow_method.png")
    plt.close()


# ============================================================================
# 3. 轮廓系数可视化
# ============================================================================
def generate_silhouette():
    """生成轮廓系数图片"""
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('轮廓系数：聚类质量评估', fontsize=14, fontweight='bold')
    
    k_values = [2, 3, 4]
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)
        
        y_lower = 10
        colors = plt.cm.viridis(np.linspace(0, 1, k))
        
        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, ith_cluster_silhouette_values,
                           facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, 
                  label=f'平均值: {silhouette_avg:.3f}')
        
        ax.set_xlabel('轮廓系数', fontsize=11)
        ax.set_ylabel('簇标签', fontsize=11)
        ax.set_title(f'K={k}', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='best')
        ax.set_xlim([-0.2, 1])
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_silhouette_coefficient.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 03_silhouette_coefficient.png")
    plt.close()


# ============================================================================
# 4. DBSCAN 与 K-Means 对比
# ============================================================================
def generate_kmeans_vs_dbscan():
    """生成DBSCAN与K-Means对比图"""
    
    from sklearn.cluster import KMeans, DBSCAN
    
    np.random.seed(42)
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('K-Means vs DBSCAN: 非凸形状聚类', fontsize=14, fontweight='bold')
    
    # K-Means
    ax = axes[0]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)
    
    colors = ['red', 'blue']
    for label in np.unique(labels_km):
        ax.scatter(X[labels_km == label, 0], X[labels_km == label, 1],
                  c=colors[label], alpha=0.6, s=50, label=f'Cluster {label}')
    
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='yellow', marker='*', s=400, edgecolors='black', linewidths=2)
    
    ax.set_title('K-Means (K=2)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # DBSCAN
    ax = axes[1]
    dbscan = DBSCAN(eps=0.15, min_samples=5)
    labels_db = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = list(labels_db).count(-1)
    
    unique_labels = set(labels_db)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'black'
            marker = 'x'
            size = 100
        else:
            marker = 'o'
            size = 50
        
        class_member_mask = (labels_db == label)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, s=size,
                  alpha=0.6, label=f'Cluster {label}' if label != -1 else '噪声')
    
    ax.set_title(f'DBSCAN (ε=0.15, MinPts=5)\n{n_clusters} 个簇, {n_noise} 个噪声点', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_kmeans_vs_dbscan.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 04_kmeans_vs_dbscan.png")
    plt.close()


# ============================================================================
# 5. DBSCAN 的 K-distance 图
# ============================================================================
def generate_kdistance():
    """生成K-distance图"""
    
    from sklearn.neighbors import NearestNeighbors
    
    np.random.seed(42)
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances[:, k], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(distances, linewidth=1.5, color='steelblue')
    
    eps_idx = np.argmax(np.diff(distances)) + 1
    eps_value = distances[eps_idx]
    ax.plot(eps_idx, eps_value, 'r*', markersize=20, label=f'推荐 ε ≈ {eps_value:.3f}')
    ax.axhline(y=eps_value, color='red', linestyle='--', alpha=0.5)
    
    ax.annotate('肘部（推荐ε值）', xy=(eps_idx, eps_value), 
                xytext=(eps_idx + 50, eps_value + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    ax.set_xlabel('数据点索引（排序）', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'第{k}个邻域距离', fontsize=12, fontweight='bold')
    ax.set_title(f'K-distance 图：选择 DBSCAN 的 ε 参数 (k={k})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_kdistance_plot.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 05_kdistance_plot.png")
    plt.close()


# ============================================================================
# 6. GMM 概率分布可视化
# ============================================================================
def generate_gmm_visualization():
    """生成GMM概率分布图"""
    
    from sklearn.mixture import GaussianMixture
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('高斯混合模型 (GMM)', fontsize=14, fontweight='bold')
    
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
    labels = gmm.fit_predict(X)
    
    # 聚类结果
    ax = axes[0]
    colors = ['red', 'green', 'blue']
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, 
                  s=50, label=f'Cluster {k+1}')
    
    for k in range(3):
        mean = gmm.means_[k]
        covar = gmm.covariances_[k]
        
        eigenvalues, eigenvectors = np.linalg.eig(covar)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        ellipse = Ellipse(xy=mean, width=2*np.sqrt(eigenvalues[0]), 
                         height=2*np.sqrt(eigenvalues[1]), 
                         angle=angle, facecolor='none', 
                         edgecolor=colors[k], linewidth=2, linestyle='--')
        ax.add_patch(ellipse)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('聚类结果和高斯分布', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 概率密度
    ax = axes[1]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    contourf = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1', s=30, alpha=0.5, edgecolors='black')
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', 
              s=300, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('概率密度分布', fontsize=11, fontweight='bold')
    plt.colorbar(contourf, ax=ax, label='Log-likelihood')
    
    # 混合系数
    ax = axes[2]
    weights = gmm.weights_
    colors_bar = ['red', 'green', 'blue']
    bars = ax.bar(range(1, 4), weights, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{weight:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('高斯分量')
    ax.set_ylabel('混合系数 (π)')
    ax.set_title('各分量的混合系数', fontsize=11, fontweight='bold')
    ax.set_xticks(range(1, 4))
    ax.set_ylim([0, max(weights) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_gmm_visualization.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 06_gmm_visualization.png")
    plt.close()


# ============================================================================
# 7. BIC/AIC 模型选择
# ============================================================================
def generate_bic_aic():
    """生成BIC/AIC选择模型图"""
    
    from sklearn.mixture import GaussianMixture
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    bic_scores = []
    aic_scores = []
    n_components_range = range(1, 11)
    
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8, 
           label='BIC', color='steelblue')
    ax.plot(n_components_range, aic_scores, 's-', linewidth=2, markersize=8, 
           label='AIC', color='coral')
    
    best_k_bic = n_components_range[np.argmin(bic_scores)]
    best_k_aic = n_components_range[np.argmin(aic_scores)]
    
    ax.plot(best_k_bic, min(bic_scores), 'r*', markersize=20, 
           label=f'最优K (BIC) = {best_k_bic}')
    ax.plot(best_k_aic, min(aic_scores), 'g*', markersize=20, 
           label=f'最优K (AIC) = {best_k_aic}')
    
    ax.set_xlabel('组件数 K', fontsize=12, fontweight='bold')
    ax.set_ylabel('信息准则值', fontsize=12, fontweight='bold')
    ax.set_title('BIC/AIC准则：GMM模型选择', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(n_components_range)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_bic_aic_criterion.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 07_bic_aic_criterion.png")
    plt.close()


# ============================================================================
# 8. 三种算法的性能对比
# ============================================================================
def generate_algorithm_comparison():
    """生成三种算法性能对比图"""
    
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      cluster_std=0.8, random_state=42)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X)
    km_silhouette = silhouette_score(X, km_labels)
    km_davies_bouldin = davies_bouldin_score(X, km_labels)
    km_calinski = calinski_harabasz_score(X, km_labels)
    
    dbscan = DBSCAN(eps=0.7, min_samples=5)
    db_labels = dbscan.fit_predict(X)
    mask = db_labels != -1
    if (db_labels != -1).sum() > 0:
        db_silhouette = silhouette_score(X[mask], db_labels[mask])
        db_davies_bouldin = davies_bouldin_score(X[mask], db_labels[mask])
        db_calinski = calinski_harabasz_score(X[mask], db_labels[mask])
    else:
        db_silhouette = db_davies_bouldin = db_calinski = 0
    
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(X)
    gmm_silhouette = silhouette_score(X, gmm_labels)
    gmm_davies_bouldin = davies_bouldin_score(X, gmm_labels)
    gmm_calinski = calinski_harabasz_score(X, gmm_labels)
    
    algorithms = ['K-Means', 'DBSCAN', 'GMM']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('三种聚类算法性能对比', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    silhouette_scores = [km_silhouette, db_silhouette, gmm_silhouette]
    colors_bar = ['steelblue', 'coral', 'lightgreen']
    bars = ax.bar(algorithms, silhouette_scores, color=colors_bar, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    ax.set_ylabel('轮廓系数', fontsize=11, fontweight='bold')
    ax.set_title('轮廓系数\n(值越高越好)', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, silhouette_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[1]
    davies_scores = [km_davies_bouldin, db_davies_bouldin, gmm_davies_bouldin]
    bars = ax.bar(algorithms, davies_scores, color=colors_bar, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    ax.set_ylabel('Davies-Bouldin指数', fontsize=11, fontweight='bold')
    ax.set_title('Davies-Bouldin指数\n(值越低越好)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, davies_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(davies_scores)*0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax = axes[2]
    calinski_scores = [km_calinski, db_calinski, gmm_calinski]
    bars = ax.bar(algorithms, calinski_scores, color=colors_bar, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    ax.set_ylabel('Calinski-Harabasz指数', fontsize=11, fontweight='bold')
    ax.set_title('Calinski-Harabasz指数\n(值越高越好)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars, calinski_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(calinski_scores)*0.02,
               f'{score:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_algorithm_comparison.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 08_algorithm_comparison.png")
    plt.close()


# ============================================================================
# 9. 三种算法在不同数据形状上的效果
# ============================================================================
def generate_algorithm_shapes():
    """生成算法在不同形状数据上的效果对比"""
    
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    
    np.random.seed(42)
    
    X1, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                       cluster_std=0.6, random_state=42)
    X2, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X3, _ = make_circles(n_samples=300, noise=0.05, random_state=42)
    
    datasets = [X1, X2, X3]
    titles = ['球形簇', '月形簇', '圆形簇']
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('三种聚类算法在不同数据形状上的表现', fontsize=16, fontweight='bold')
    
    for data_idx, (X, title) in enumerate(zip(datasets, titles)):
        # K-Means
        ax = axes[data_idx, 0]
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)
        
        colors = ['red', 'green', 'blue']
        for k in range(3):
            mask = km_labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='yellow', marker='*', s=300, edgecolors='black', linewidths=2)
        
        ax.set_title(f'K-Means - {title}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # DBSCAN
        ax = axes[data_idx, 1]
        
        if data_idx == 0:
            dbscan = DBSCAN(eps=0.7, min_samples=5)
        elif data_idx == 1:
            dbscan = DBSCAN(eps=0.15, min_samples=5)
        else:
            dbscan = DBSCAN(eps=0.15, min_samples=5)
        
        db_labels = dbscan.fit_predict(X)
        
        unique_labels = set(db_labels)
        colors_db = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors_db):
            if label == -1:
                color = 'black'
                marker = 'x'
            else:
                marker = 'o'
            
            mask = db_labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker, s=30, alpha=0.6)
        
        ax.set_title(f'DBSCAN - {title}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # GMM
        ax = axes[data_idx, 2]
        gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
        gmm_labels = gmm.fit_predict(X)
        
        colors_gmm = ['red', 'green', 'blue']
        for k in range(3):
            mask = gmm_labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c=colors_gmm[k], alpha=0.6, s=30)
        
        ax.set_title(f'GMM - {title}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_algorithm_shapes.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 09_algorithm_shapes.png")
    plt.close()


# ============================================================================
# 10. 计算复杂度对比
# ============================================================================
def generate_complexity_analysis():
    """生成计算复杂度分析图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('聚类算法的计算复杂度分析', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    
    n_samples = np.logspace(2, 5, 20)
    time_kmeans = n_samples * 3 * 2 * 10 / 1e6
    time_dbscan_index = n_samples * np.log(n_samples) / 1e6
    time_gmm = n_samples * 3 * 4 * 20 / 1e6
    
    ax.loglog(n_samples, time_kmeans, 'o-', linewidth=2, markersize=6, 
             label='K-Means: O(nKdt)', color='steelblue')
    ax.loglog(n_samples, time_dbscan_index, 's-', linewidth=2, markersize=6,
             label='DBSCAN (with index): O(n log n)', color='coral')
    ax.loglog(n_samples, time_gmm, '^-', linewidth=2, markersize=6,
             label='GMM: O(nKd²t)', color='lightgreen')
    
    ax.set_xlabel('样本数量 n', fontsize=12, fontweight='bold')
    ax.set_ylabel('运行时间 (相对单位)', fontsize=12, fontweight='bold')
    ax.set_title('时间复杂度对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    ax = axes[1]
    
    algorithms = ['K-Means', 'DBSCAN', 'GMM']
    space_complexity = ['O(n+K)', 'O(n)', 'O(nK)']
    space_values = [1, 1, 3]
    
    colors_bar = ['steelblue', 'coral', 'lightgreen']
    bars = ax.bar(algorithms, space_values, color=colors_bar, alpha=0.7, 
                 edgecolor='black', linewidth=2)
    
    ax.set_ylabel('空间复杂度 (相对单位)', fontsize=12, fontweight='bold')
    ax.set_title('空间复杂度对比', fontsize=12, fontweight='bold')
    
    for bar, algo, complexity in zip(bars, algorithms, space_complexity):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               complexity, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim([0, 3.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_complexity_analysis.png', format='png', bbox_inches='tight')
    print("✓ 已生成: 10_complexity_analysis.png")
    plt.close()


# ============================================================================
# 主函数
# ============================================================================
def main():
    """生成所有可视化图片"""
    
    print("=" * 60)
    print("开始生成聚类算法可视化图片...")
    print("=" * 60)
    
    try:
        generate_kmeans_iteration()
        generate_elbow_method()
        generate_silhouette()
        generate_kmeans_vs_dbscan()
        generate_kdistance()
        generate_gmm_visualization()
        generate_bic_aic()
        generate_algorithm_comparison()
        generate_algorithm_shapes()
        generate_complexity_analysis()
        
        # 转换PNG为WebP
        batch_convert_to_webp(quality=85)
        
        print("\n" + "=" * 60)
        print("✓ 所有图片生成完成！")
        print("=" * 60)
        print(f"\n图片位置: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ 生成过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

---
layout: post
title: "机器学习——无监督学习中的降维算法(PCA，流形学习)"
subtitle: "深入理解主成分分析、流形学习与降维的数学原理及实际应用"
date: 2026-01-24
author: "DoraemonJack"
header-img: "img/post-bg-ml.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - 机器学习
    - Machine Learning
    - 无监督学习
    - Unsupervised Learning
    - 降维
    - Dimensionality Reduction
    - PCA
    - Principal Component Analysis
    - t-SNE
    - 流形学习
    - 数学原理
---

> 维数灾难是高维空间中的幽灵，而降维算法是驱散它的魔法。本文将深入探讨无监督学习中最重要的降维算法，从数学基础到实际应用。

## 目录
1. [降维问题的定义与意义](#降维问题的定义与意义)
2. [主成分分析(PCA)](#主成分分析pca)
3. [核主成分分析(KPCA)](#核主成分分析kpca)
4. [独立成分分析(ICA)](#独立成分分析ica)
5. [流形学习](#流形学习)
6. [实际案例：人脸识别系统](#实际案例人脸识别系统)
7. [代码实现](#代码实现)

---

## 降维问题的定义与意义

### 什么是维数灾难？

在高维空间中会发生一些反直觉的现象：
- **距离的失效**：在高维空间中，大多数点对之间的距离趋向相同
- **数据稀疏性**：要填充高维空间需要指数级增长的样本
- **计算复杂性**：算法时间复杂度随维度指数增长

设在单位超立方体 $[0,1]^d$ 中均匀分布 $n$ 个样本点：

$$P(\text{max distance} > r) \approx 1 - (1-r^d)^n$$

当 $d \to \infty$ 时，距离集中在 $[c_1\sqrt{d}, c_2\sqrt{d}]$ 范围内。

### 降维的目标

降维致力于在低维空间中保留原始数据的重要特征，形式上可表述为：

给定高维数据 $X \in \mathbb{R}^{n \times D}$，寻找映射 $f: \mathbb{R}^D \to \mathbb{R}^d$（其中 $d \ll D$），使得：
- 保留数据的方差信息（线性降维）
- 保留数据的局部邻域结构（非线性降维）
- 保留数据的流形结构（流形学习）

---

## 主成分分析(PCA)

### 1. 数学原理

#### 问题设定

给定数据矩阵 $X \in \mathbb{R}^{n \times D}$，其中 $n$ 是样本数，$D$ 是特征维数。

首先进行中心化处理：
$$X_c = X - \mathbb{E}[X]$$

其中 $\mathbb{E}[X] = \frac{1}{n}\sum_{i=1}^n x_i$ 是特征均值。

#### 目标函数

PCA 的目标是找到一组正交基向量 $\{w_1, w_2, ..., w_d\}$，使得数据在这些方向上的方差最大。

对于第一主成分方向 $w_1$（$\|w_1\| = 1$），最大化投影方差：

$$\max_{w_1} \frac{1}{n}\sum_{i=1}^n (w_1^T x_i)^2 = \max_{w_1} w_1^T \Sigma w_1$$

其中协方差矩阵为：
$$\Sigma = \frac{1}{n}X_c^T X_c$$

约束条件：$w_1^T w_1 = 1$

#### 求解方法：特征值分解

使用拉格朗日乘数法：
$$\mathcal{L}(w_1, \lambda) = w_1^T \Sigma w_1 - \lambda(w_1^T w_1 - 1)$$

求导得：
$$\Sigma w_1 = \lambda w_1$$

这表明 $w_1$ 是协方差矩阵 $\Sigma$ 的特征向量，$\lambda$ 是对应的特征值。

为了最大化方差，我们选择最大特征值对应的特征向量作为第一主成分。

**一般地**，第 $k$ 个主成分满足：
$$\Sigma w_k = \lambda_k w_k, \quad \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_D$$

#### 降维映射

选择前 $d$ 个最大特征值对应的特征向量组成投影矩阵 $W \in \mathbb{R}^{D \times d}$：

$$Z = X_c W$$

得到降维后的数据 $Z \in \mathbb{R}^{n \times d}$。

#### 解释方差

第 $k$ 个主成分解释的方差比例：
$$\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{i=1}^D \lambda_i}$$

累积解释方差比例：
$$\text{Cumulative EVR}_d = \frac{\sum_{i=1}^d \lambda_i}{\sum_{i=1}^D \lambda_i}$$

通常选择使累积解释方差达到 95% 或 99% 的维数。

### 2. PCA 的数学性质

#### 重建误差最小

PCA 等价于最小化重建误差：
$$\min_W \|X_c - X_c W W^T\|_F^2$$

其中 $\|·\|_F$ 表示 Frobenius 范数。

展开：
$$\|X_c - X_c W W^T\|_F^2 = \text{tr}(X_c^T X_c) - \text{tr}(W^T X_c^T X_c W)$$

$$= \sum_{i=1}^D \lambda_i - \sum_{i=1}^d \lambda_i = \sum_{i=d+1}^D \lambda_i$$

因此，最小化重建误差等价于最大化投影方差。

#### 协方差椭球体主轴

PCA 几何解释：数据的协方差椭球体的主轴就是主成分方向，轴长正比于对应的特征值 $\sqrt{\lambda_i}$。

#### 旋转不变性

PCA 对数据的旋转不敏感，但对尺度敏感。需要标准化：
$$\tilde{X} = \frac{X - \mu}{\sigma}$$

### 3. PCA 的计算算法

#### 方法1：特征值分解(EVD)

```
Input: X ∈ ℝ^(n×D), d
1. 中心化: X_c = X - mean(X)
2. 计算协方差: Σ = (1/n)X_c^T X_c
3. 特征值分解: Σ = UΛU^T
4. 取前d个特征向量: W = U[:, 1:d]
5. 降维: Z = X_c W
Output: Z ∈ ℝ^(n×d)
```

时间复杂度：$O(D^3 + nd^2)$

#### 方法2：奇异值分解(SVD)

对中心化数据进行 SVD 分解：
$$X_c = U \Sigma V^T$$

其中 $U \in \mathbb{R}^{n \times n}$，$\Sigma \in \mathbb{R}^{n \times D}$，$V \in \mathbb{R}^{D \times D}$

PCA 投影矩阵：$W = V[:, 1:d]$

相比 EVD，SVD 数值稳定性更好，特别适合 $n \ll D$ 的情况。

时间复杂度：$O(nD^2)$ 或 $O(n^2D)$

#### 方法3：增量 PCA

对于大规模数据，可以使用增量 PCA，逐批处理数据：

$$\Sigma^{(t)} = \frac{n^{(t-1)}}{n^{(t)}} \Sigma^{(t-1)} + \frac{1}{n^{(t)}} X_{new}^T X_{new} + \frac{n^{(t-1)}n^{(t-1)}}{n^{(t)}(n^{(t)} + n^{(t-1)})} (\mu^{(t-1)} - \mu^{(t)})(\mu^{(t-1)} - \mu^{(t)})^T$$

时间复杂度：$O(nDd)$，无需存储整个数据矩阵。

---

## 核主成分分析(KPCA)

### 1. 非线性问题

PCA 是线性方法，不能处理非线性流形。考虑瑞士卷数据集（Swiss Roll）：

$$x(u,v) = u\cos(u), v, u\sin(u), \quad u \in [0, 4\pi], v \in [0, 10]$$

数据在 3D 空间中，但本质上是 2D 流形。标准 PCA 无法有效降维。

### 2. 核技巧

KPCA 使用核技巧将数据映射到高维特征空间 $\mathcal{H}$：

$$\phi: \mathbb{R}^D \to \mathcal{H}, \quad x \mapsto \phi(x)$$

在特征空间中执行 PCA。

#### 核函数

常用的核函数：

1. **线性核**：$k(x_i, x_j) = x_i^T x_j$

2. **多项式核**：$k(x_i, x_j) = (x_i^T x_j + c)^p$

3. **RBF 核**：$k(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$，其中 $\gamma > 0$

4. **Sigmoid 核**：$k(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$

### 3. KPCA 算法

核 Gram 矩阵：
$$K = \Phi^T \Phi, \quad K_{ij} = k(x_i, x_j)$$

其中 $\Phi = [\phi(x_1), ..., \phi(x_n)]^T$。

中心化 Gram 矩阵：
$$K_c = K - \mathbf{1} K - K \mathbf{1} + \mathbf{1} K \mathbf{1}$$

其中 $\mathbf{1} = \frac{1}{n}\mathbf{1}_n\mathbf{1}_n^T$。

特征值分解 $K_c$，得到特征向量 $\alpha_k$：

降维映射：
$$z_k(x) = \sum_{i=1}^n \alpha_k^{(i)} k(x, x_i)$$

---

## 独立成分分析(ICA)

### 1. 盲源分离问题

ICA 旨在解决盲源分离(Blind Source Separation)问题。

**观测模型**：
$$x = As$$

其中：
- $x \in \mathbb{R}^m$ 是观测向量
- $s \in \mathbb{R}^n$ 是源信号向量
- $A \in \mathbb{R}^{m \times n}$ 是未知混合矩阵（$m \geq n$）

**目标**：在仅知道 $x$ 的情况下恢复 $s$ 和 $A$。

### 2. ICA 的假设

ICA 基于以下关键假设：

1. **源信号统计独立**：$p(s_i, s_j) = p(s_i)p(s_j)$ for $i \neq j$

2. **源信号非高斯分布**：至多一个源信号可以是高斯分布

3. **混合矩阵可逆**（对于确定情况）

### 3. 数学原理

#### 信息论基础

使用互信息度量独立性：
$$I(s_i; s_j) = D_{KL}(p(s_i, s_j) \| p(s_i)p(s_j)) = \int p(s_i, s_j) \log \frac{p(s_i, s_j)}{p(s_i)p(s_j)} ds_i ds_j$$

对于独立的源信号：$I(s_i; s_j) = 0$

#### 负熵的使用

负熵(Negentropy)：
$$J(s) = H(\text{Gaussian}) - H(s) = \frac{1}{2}\log(2\pi e \sigma^2) - \frac{1}{2}\log|\text{Cov}(s)|$$

其中 $\sigma^2 = \text{Var}(s)$，且 $\text{Cov}(s) = I$（白化后）。

近似公式（计算高效）：
$$J(s) \approx c[\mathbb{E}(G(s)) - \mathbb{E}(G(v))]^2$$

其中 $v \sim \mathcal{N}(0, I)$，$G$ 是非二次函数。

常用函数：
- $G_1(s) = \frac{1}{a_1}\log\cosh(a_1 s)$
- $G_2(s) = -\exp(-s^2/2)$

#### FastICA 算法

目标：最大化 $J(w^T x)$，其中 $w$ 是单位向量。

使用牛顿法：
$$w \leftarrow \mathbb{E}[x g(w^T x)] - \mathbb{E}[g'(w^T x)] w$$

其中 $g = G'$。

**算法流程**：
```
Input: x (中心化、白化), p (源信号数)
1. 初始化W随机
2. for each component i:
     for iteration:
         w_i ← E[x g(w_i^T x)] - E[g'(w_i^T x)] w_i
         w_i ← w_i / ||w_i||
         正交化: w_i ← w_i - W_{i-1} W_{i-1}^T w_i
     if converged: break
Output: W (源信号的混合矩阵估计)
```

---

## 流形学习

### 1. 流形假设

数据分布在高维空间的低维流形上。直观例子：

- **瑞士卷**：3D 空间中的 2D 流形
- **S 曲线**：2D 空间中的 1D 流形
- **MNIST 数字**：784D 像素空间中的~100D 流形

### 2. 局部线性嵌入(LLE)

#### 原理

假设流形在局部是线性的，每个点可用邻域内的点的线性组合表示。

#### 算法步骤

**Step 1: 构建邻域图**

对每个样本 $x_i$，找到其 $k$ 个最近邻 $\mathcal{N}(i)$。

**Step 2: 计算重建权重**

最小化局部重建误差：
$$\min_W \sum_{i=1}^n \|x_i - \sum_{j \in \mathcal{N}(i)} w_{ij} x_j\|^2$$

约束：$\sum_{j \in \mathcal{N}(i)} w_{ij} = 1$

对于每个 $i$，可独立求解（$m \times m$ 系统，$m = k+1$）：

$$w_i = \frac{G_i^{-1} \mathbf{1}}{\mathbf{1}^T G_i^{-1} \mathbf{1}}$$

其中 $G_i$ 是中心化邻域点的 Gram 矩阵，$\mathbf{1}$ 是全一向量。

**Step 3: 学习低维嵌入**

保持权重固定，在低维空间中求解：
$$\min_Z \sum_{i=1}^n \|z_i - \sum_{j \in \mathcal{N}(i)} w_{ij} z_j\|^2$$

这等价于求解广义特征值问题的最小特征值：
$$(I - W)^T(I - W) z_k = \lambda_k M z_k$$

其中 $M = (I - W)^T(I - W)$。

#### 时间复杂度

- 最近邻搜索：$O(n^2)$
- 权重计算：$O(nk^3)$
- 特征值求解：$O(n^3)$

总体：$O(n^3)$

### 3. t-SNE(t-Distributed Stochastic Neighbor Embedding)

#### SNE 基础

构造高维和低维概率分布，最小化 KL 散度。

**高维相似度**：
$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

对称化：
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**低维相似度**（高斯分布）：
$$q_{ij} = \frac{\exp(-\|z_i - z_j\|^2)}{\sum_{k \neq l} \exp(-\|z_k - z_l\|^2)}$$

#### t-SNE 改进

使用 t 分布替代高斯分布（长尾特性）：
$$q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1}}$$

#### 目标函数

KL 散度：
$$D_{KL}(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**性质**：吸引力项(attractive forces)和排斥力项(repulsive forces)

- **吸引**：高维中相近的点在低维中应该接近
- **排斥**：高维中相远的点在低维中应该相离

### 4. UMAP(Uniform Manifold Approximation and Projection)

#### 核心思想

构造图拓扑，基于 Riemannian 几何学。

**相似度**：
$$\rho_{ij} = \max(0, d(x_i, x_j) - \rho_i)$$

其中 $\rho_i$ 是到最近邻的距离。

$$\mu_{ij} = \begin{cases}
\rho_{ij} & \text{if } d_{ij} \leq \rho_i \\
\exp(-(d_{ij} - \rho_i)/\sigma_i) & \text{otherwise}
\end{cases}$$

#### 低维投影

使用二元交叉熵损失：
$$\mathcal{L} = \sum_{(i,j) \in E^+} -\log(\sigma(\tilde{d}_{ij})) - \sum_{(i,j) \in E^-} \log(1 - \sigma(\tilde{d}_{ij}))$$

其中 $E^+$ 是邻域内的点对，$E^-$ 是邻域外的点对。

---

## 实际案例：人脸识别系统

### 问题描述

构建一个人脸识别系统，能够：
1. 从高分辨率图像提取有效特征
2. 快速检索相似人脸
3. 实现人脸验证和识别

### 数据集

使用 LFW(Labeled Faces in the Wild) 数据集：
- 13,233 张图像
- 5,749 个人
- 每张图像 250×250 像素，RGB 3 通道
- 原始特征维数：$250 \times 250 \times 3 = 187,500$

### 解决方案架构

#### 第 1 阶段：预处理

1. **人脸检测与对齐**
   - 使用 Dlib 或 MTCNN 检测人脸
   - 对齐眼睛、鼻子等关键点
   - 标准化为 224×224 像素

2. **特征提取**
   - 使用预训练的 ResNet-50 (ImageNet)
   - 提取倒数第二层特征：4,096 维

#### 第 2 阶段：降维

1. **使用 PCA 降维**
   - 输入：$n = 5,000$ 张人脸图像，$D = 4,096$ 维
   - 目标：保留 99% 的方差

**计算过程**：

```
1. 中心化特征: X_c = X - μ
2. 协方差矩阵: Σ = (1/5000) X_c^T X_c
   - 大小: 4096×4096
   - 计算复杂度: O(5000 × 4096²)
3. 特征值分解: Σ = UΛU^T
4. 选择前 d 个特征值, 使得:
   ∑_{i=1}^d λ_i / ∑_{i=1}^D λ_i ≥ 0.99
```

**结果示例**：
| 维数 | 累积解释方差 | 压缩比 |
|-----|-----------|--------|
| 256 | 87.3% | 16× |
| 512 | 95.2% | 8× |
| 768 | 98.1% | 5.3× |
| 1024 | 99.0% | 4× |

选择 **d = 1024** 维。

2. **可选：使用 KPCA 进一步优化**

使用 RBF 核增强非线性特性：

```
k(x_i, x_j) = exp(-γ ||x_i - x_j||²)
γ = 1/(2σ²), σ² = median(||x_i - x_j||²)
```

可以进一步降至 512 维，同时保留 97% 方差。

#### 第 3 阶段：人脸识别

**验证**（1:1 匹配）：
$$\text{distance} = \|z_{\text{query}} - z_{\text{reference}}\|_2$$
$$\text{match} = \text{distance} < \text{threshold}$$

通常设置阈值为 0.6（根据 ROC 曲线调整）。

**识别**（1:N 搜索）：
```
Input: query_face
1. 提取特征：x_q
2. PCA 降维：z_q = (x_q - μ) W
3. 搜索：argmin_i ||z_q - z_i||²
4. Top-5 候选者
```

### 性能指标

| 指标 | 原始特征 | PCA 降维 | KPCA 降维 |
|-----|--------|---------|----------|
| 特征维数 | 4,096 | 1,024 | 512 |
| 内存占用 | 100% | 25% | 12.5% |
| 搜索时间 | 25 ms | 3 ms | 1.5 ms |
| 验证准确率 | 98.5% | 97.8% | 97.2% |
| 识别准确率 (@Top-1) | 94.3% | 92.1% | 88.5% |

**权衡分析**：
- PCA 维数选择 1,024 平衡准确率和效率
- 内存减少到 1/4，速度提升 8 倍
- 准确率下降不到 1.5%

### 进一步优化

1. **局部 PCA**：不同的脸部区域使用不同的投影矩阵
2. **非线性降维**：对困难案例使用 t-SNE 进行可视化和聚类
3. **核方法**：KPCA 用于微调最后 10% 困难样本

---

## 代码实现

### 1. PCA 从零实现

```python
import numpy as np
from numpy.linalg import svd, eig

class PCA:
    """主成分分析实现"""
    
    def __init__(self, n_components=None, explained_variance_ratio=None):
        """
        Parameters:
        -----------
        n_components : int, optional
            降维后的维数
        explained_variance_ratio : float, optional
            保留的累积方差比例 (0, 1)
            如果指定，则自动计算 n_components
        """
        self.n_components = n_components
        self.explained_variance_ratio_threshold = explained_variance_ratio
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        拟合 PCA 模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 方法：SVD
        U, S, Vt = svd(X_centered, full_matrices=False)
        
        # 方差（特征值）
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        
        # 方差比例
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # 确定降维维数
        if self.explained_variance_ratio_threshold is not None:
            cumsum = np.cumsum(self.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.explained_variance_ratio_threshold) + 1
        
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # 主成分（特征向量）
        self.components_ = Vt[:self.n_components, :].T
        
        return self
    
    def transform(self, X):
        """
        降维投影
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def inverse_transform(self, X_transformed):
        """
        重建原始维数
        
        Parameters:
        -----------
        X_transformed : array, shape (n_samples, n_components)
        
        Returns:
        --------
        X_original : array, shape (n_samples, n_features)
        """
        return X_transformed @ self.components_.T + self.mean_
    
    def fit_transform(self, X):
        """拟合并转换"""
        return self.fit(X).transform(X)
    
    def get_info(self):
        """获取 PCA 信息"""
        cumsum_ratio = np.cumsum(self.explained_variance_ratio_)
        print(f"选择的主成分数: {self.n_components}")
        print(f"累积解释方差: {cumsum_ratio[self.n_components-1]:.4f}")
        print(f"\n前 10 个主成分的解释方差:")
        for i in range(min(10, self.n_components)):
            print(f"  PC{i+1}: {self.explained_variance_ratio_[i]:.4f} "
                  f"(累积: {cumsum_ratio[i]:.4f})")
```

### 2. 核 PCA 实现

```python
class KernelPCA:
    """核主成分分析"""
    
    def __init__(self, n_components=2, kernel='rbf', gamma=None, c=None, degree=3):
        """
        Parameters:
        -----------
        kernel : str
            'linear', 'rbf', 'poly', 'sigmoid'
        gamma : float
            RBF 核的参数
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.degree = degree
        self.X_fit = None
        self.K_fit = None
        self.alphas_ = None
        self.lambdas_ = None
        
    def _kernel_matrix(self, X, Y=None):
        """计算核矩阵"""
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            K = X @ Y.T
        
        elif self.kernel == 'rbf':
            # ||x-y||² = ||x||² + ||y||² - 2<x,y>
            X_norm_sq = np.sum(X**2, axis=1, keepdims=True)
            Y_norm_sq = np.sum(Y**2, axis=1, keepdims=True).T
            K = X_norm_sq + Y_norm_sq - 2 * (X @ Y.T)
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            K = np.exp(-gamma * K)
        
        elif self.kernel == 'poly':
            K = (X @ Y.T + 1) ** self.degree
        
        elif self.kernel == 'sigmoid':
            K = np.tanh(self.c * (X @ Y.T) + 1)
        
        return K
    
    def fit(self, X):
        """拟合 KPCA"""
        n_samples = X.shape[0]
        self.X_fit = X
        
        # 计算核矩阵
        K = self._kernel_matrix(X)
        
        # 中心化核矩阵
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # 特征值分解
        eigenvalues, eigenvectors = eig(K_centered)
        
        # 排序（降序）
        idx = np.argsort(eigenvalues)[::-1]
        self.lambdas_ = eigenvalues[idx]
        self.alphas_ = eigenvectors[:, idx[:self.n_components]]
        
        # 正交化
        for i in range(self.n_components):
            self.alphas_[:, i] /= np.sqrt(self.lambdas_[i])
        
        self.K_fit = K_centered
        
        return self
    
    def transform(self, X):
        """降维投影"""
        K = self._kernel_matrix(X, self.X_fit)
        
        # 中心化
        n_samples_fit = self.X_fit.shape[0]
        one_m_n = np.ones((X.shape[0], n_samples_fit)) / n_samples_fit
        K_centered = K - one_m_n @ self.K_fit - K @ np.ones((n_samples_fit, n_samples_fit)) / n_samples_fit
        
        return K_centered @ self.alphas_
    
    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.K_fit @ self.alphas_
```

### 3. 人脸识别系统完整实现

```python
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path

class FaceRecognitionSystem:
    """人脸识别系统"""
    
    def __init__(self, pca_components=1024, use_kpca=False):
        self.pca = PCA(n_components=pca_components)
        self.kpca = KernelPCA(n_components=512, kernel='rbf') if use_kpca else None
        self.face_features = []
        self.face_names = []
        self.face_ids = []
        self.verification_threshold = 0.6
        
    def load_features(self, features_file):
        """
        加载预提取的特征
        
        Features 应该来自 ResNet-50 倒数第二层
        """
        features = np.load(features_file)  # shape: (n_faces, 4096)
        names = np.load(features_file.replace('features', 'names'))
        
        # 拟合 PCA
        self.pca.fit(features)
        reduced_features = self.pca.transform(features)
        
        # 可选：拟合 KPCA
        if self.kpca is not None:
            reduced_features = self.kpca.fit_transform(reduced_features)
        
        self.face_features = reduced_features
        self.face_names = names
        self.face_ids = np.arange(len(names))
        
        print(f"已加载 {len(names)} 个人脸")
        print(f"特征维数: {reduced_features.shape[1]}")
        
    def verify_face(self, query_features):
        """
        人脸验证 (1:1 匹配)
        
        Returns:
        --------
        (is_match, confidence, distance)
        """
        # 特征降维
        query_reduced = self.pca.transform(query_features[np.newaxis, :])[0]
        
        if self.kpca is not None:
            query_reduced = self.kpca.transform(query_reduced[np.newaxis, :])[0]
        
        # 计算距离
        distances = euclidean_distances(
            query_reduced.reshape(1, -1),
            self.face_features.reshape(len(self.face_features), -1)
        )[0]
        
        min_distance = np.min(distances)
        is_match = min_distance < self.verification_threshold
        confidence = 1.0 - (min_distance / 2.0)  # 转换为置信度
        
        return is_match, confidence, min_distance
    
    def identify_face(self, query_features, top_k=5):
        """
        人脸识别 (1:N 搜索)
        
        Returns:
        --------
        [(name, distance, confidence), ...]
        """
        # 特征降维
        query_reduced = self.pca.transform(query_features[np.newaxis, :])[0]
        
        if self.kpca is not None:
            query_reduced = self.kpca.transform(query_reduced[np.newaxis, :])[0]
        
        # 计算距离
        distances = euclidean_distances(
            query_reduced.reshape(1, -1),
            self.face_features.reshape(len(self.face_features), -1)
        )[0]
        
        # Top-K
        top_indices = np.argsort(distances)[:top_k]
        results = [
            (self.face_names[i], distances[i], 1.0 - distances[i] / 2.0)
            for i in top_indices
        ]
        
        return results
    
    def set_verification_threshold(self, threshold):
        """设置验证阈值"""
        self.verification_threshold = threshold
```

### 4. 使用示例

```python
# 示例：合成数据
np.random.seed(42)

# 生成高维数据（瑞士卷）
def swiss_roll(n_samples=1000, noise=0.1):
    t = 3 * np.pi * np.random.rand(n_samples)
    height = 21 * np.random.rand(n_samples)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    
    if noise:
        x += noise * np.random.randn(n_samples)
        y += noise * np.random.randn(n_samples)
        z += noise * np.random.randn(n_samples)
    
    return np.column_stack([x, y, z])

# 生成数据
X = swiss_roll(n_samples=1000, noise=0.1)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KPCA 降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)

# 可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# 原始数据
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap='viridis', alpha=0.6)
ax1.set_title('原始数据 (瑞士卷)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# PCA 结果
ax2 = fig.add_subplot(132)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 0], cmap='viridis', alpha=0.6)
ax2.set_title('PCA 降维结果')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

# KPCA 结果
ax3 = fig.add_subplot(133)
ax3.scatter(X_kpca[:, 0], X_kpca[:, 1], c=X[:, 0], cmap='viridis', alpha=0.6)
ax3.set_title('KPCA 降维结果 (RBF)')
ax3.set_xlabel('KPCA1')
ax3.set_ylabel('KPCA2')

plt.tight_layout()
plt.savefig('dimensionality_reduction_comparison.png', dpi=150)
plt.show()

# PCA 信息
pca_full = PCA(n_components=3)
pca_full.fit(X)
pca_full.get_info()
```

### 5. 性能基准测试

```python
import time

def benchmark_algorithms(X, dimensions=[2, 5, 10, 20]):
    """基准测试不同降维算法"""
    
    results = {}
    
    for d in dimensions:
        print(f"\n降维到 {d} 维:")
        
        # PCA
        start = time.time()
        pca = PCA(n_components=d)
        X_pca = pca.fit_transform(X)
        pca_time = time.time() - start
        print(f"  PCA: {pca_time:.4f}s")
        
        # KPCA (RBF)
        start = time.time()
        kpca = KernelPCA(n_components=d, kernel='rbf')
        X_kpca = kpca.fit_transform(X)
        kpca_time = time.time() - start
        print(f"  KPCA: {kpca_time:.4f}s")
        
        # LLE
        start = time.time()
        lle = LLE(n_components=d)
        X_lle = lle.fit_transform(X)
        lle_time = time.time() - start
        print(f"  LLE: {lle_time:.4f}s")
        
        results[d] = {
            'pca': pca_time,
            'kpca': kpca_time,
            'lle': lle_time
        }
    
    return results
```

---

## 总结与建议

### 算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|--------|------|
| 线性可分数据 | **PCA** | 计算高效，可解释性强 |
| 非线性流形 | **KPCA/t-SNE** | 能捕捉非线性结构 |
| 实时应用 | **PCA/增量PCA** | 速度快，内存高效 |
| 可视化 (2D/3D) | **t-SNE/UMAP** | 视觉效果好 |
| 独立成分分析 | **ICA** | 源信号恢复 |
| 大规模数据 | **增量PCA/UMAP** | 可处理海量数据 |

### 实践经验

1. **总是进行数据预处理**
   - 中心化和标准化
   - 处理缺失值

2. **选择合适的维数**
   - 使用累积解释方差 (通常 95%-99%)
   - 交叉验证评估下游任务性能
   - 考虑计算和存储成本

3. **超参数调优**
   - KPCA：选择合适的核函数和参数
   - t-SNE：学习率、困惑度(perplexity)
   - LLE：邻域大小 $k$

4. **可视化与验证**
   - 2D/3D 可视化检查结构
   - 计算重建误差
   - 评估下游任务性能

5. **组合方法**
   - PCA 预处理 + KPCA 精调
   - 分阶段降维
   - 集成多个降维方法

---

## 参考文献

1. Turk, M., & Pentland, A. (1991). "Eigenfaces for recognition". Journal of Cognitive Neuroscience.

2. Jolliffe, I. T. (2002). "Principal Component Analysis" (2nd ed.). Springer.

3. Schölkopf, B., Smola, A., & Müller, K. R. (2001). "Kernel Principal Component Analysis". Neural Computation.

4. Hinton, G. E., & Roweis, S. T. (2002). "Stochastic Neighbor Embedding". NIPS.

5. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE". JMLR.

6. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction". arXiv.

7. Roweis, S. T., & Saul, L. K. (2000). "Nonlinear dimensionality reduction by locally linear embedding". Science.

---

## 扩展阅读

- **自动编码器**：神经网络方法的降维
- **变分自编码器(VAE)**：概率生成模型
- **对比学习**：最新的深度降维方法
- **谱聚类**：基于图的方法

---

**最后更新：** 2026年1月24日

**作者：** DoraemonJack

**标签：** #机器学习 #无监督学习 #降维 #PCA #流形学习

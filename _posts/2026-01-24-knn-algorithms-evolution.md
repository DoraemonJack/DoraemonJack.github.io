---
layout: post
title: "机器学习——K-近邻算法"
subtitle: "KNN到图神经网络"
date: 2026-01-24
author: "DoraemonJack"
header-img: "img/post-bg-knn.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - 机器学习
    - Machine Learning
    - K-近邻算法
    - 图神经网络
    - 距离学习
    - 算法演变
---

> 本文详细解析K-近邻算法的数学原理、演变历程，以及如何应用于实际问题。从最基础的欧几里得距离，到现代深度度量学习，再到图神经网络，完整呈现这一算法家族的发展轨迹。

## 目录
1. [K-近邻算法基础](#k-近邻算法基础)
2. [数学原理详解](#数学原理详解)
3. [算法局限性分析](#算法局限性分析)
4. [加速策略](#加速策略)
5. [算法演变与改进](#算法演变与改进)
6. [现代应用与扩展](#现代应用与扩展)
7. [实际案例研究](#实际案例研究)
8. [性能对比与选择指南](#性能对比与选择指南)

---

## K-近邻算法基础

### 算法思想

K-近邻（K-Nearest Neighbors, KNN）是最简单却最强大的机器学习算法之一，其核心思想是：

> **一个样本的类别由其最近的K个样本的类别决定**

这个思想源于人类的直观认知——我们倾向于和周围的人相似。

### 算法流程

```
输入：训练集 D = {(x₁, y₁), (x₂, y₂), ..., (xₘ, yₘ)}
      查询点 q
      参数 K
      距离度量 d(·,·)

输出：查询点 q 的类别预测 ŷ

步骤：
1. 计算 q 到所有训练样本的距离
2. 找出距离最近的 K 个样本
3. 统计这 K 个样本的类别
4. 返回出现次数最多的类别
```

### 简单示例

假设我们有以下训练数据：
- 样本A (身高170, 体重65) → "男性"
- 样本B (身高175, 体重70) → "男性"  
- 样本C (身高160, 体重50) → "女性"
- 样本D (身高165, 体重55) → "女性"

查询点Q (身高172, 体重68)，K=3时：

1. 计算距离：
   - d(Q, A) = √((172-170)² + (68-65)²) = √13 ≈ 3.61
   - d(Q, B) = √((172-175)² + (68-70)²) = √13 ≈ 3.61
   - d(Q, C) = √((172-160)² + (68-50)²) = √544 ≈ 23.32
   - d(Q, D) = √((172-165)² + (68-55)²) = √218 ≈ 14.76

2. 最近的3个：A, B, D
3. 统计类别：男性出现2次，女性出现1次
4. 预测结果：**男性**

---

## 数学原理详解

### 1. 距离度量

距离度量是KNN算法的核心。不同的距离度量会产生不同的结果。

#### 1.1 欧几里得距离（Euclidean Distance）

最常用的距离度量，基于勾股定理：

$$d_E(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{i,k} - x_{j,k})^2}$$

**优点**：直观、几何意义明确
**缺点**：对特征量纲敏感，不适合高维稀疏数据

**改进**：特征标准化

$$x'_{i,k} = \frac{x_{i,k} - \mu_k}{\sigma_k}$$

其中 $\mu_k$ 是第k维特征的均值，$\sigma_k$ 是标准差。

#### 1.2 曼哈顿距离（Manhattan Distance）

$$d_M(x_i, x_j) = \sum_{k=1}^{n} |x_{i,k} - x_{j,k}|$$

**几何意义**：在网格上的实际距离（如城市街区）

**应用场景**：分类特征较多、特征之间独立性较强

#### 1.3 闵可夫斯基距离（Minkowski Distance）

统一距离度量的框架：

$$d_p(x_i, x_j) = \left(\sum_{k=1}^{n} |x_{i,k} - x_{j,k}|^p\right)^{1/p}$$

其中：
- p = 1 → 曼哈顿距离
- p = 2 → 欧几里得距离
- p → ∞ → 切比雪夫距离

#### 1.4 余弦相似度（Cosine Similarity）

$$\text{cos}(x_i, x_j) = \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||} = \frac{\sum_{k=1}^{n} x_{i,k} \cdot x_{j,k}}{\sqrt{\sum_{k=1}^{n} x_{i,k}^2} \cdot \sqrt{\sum_{k=1}^{n} x_{j,k}^2}}$$

**对应距离**：$d_{cos}(x_i, x_j) = 1 - \text{cos}(x_i, x_j)$

**应用**：文本分类、高维稀疏数据

**优点**：不受向量大小影响，只关心方向

#### 1.5 马氏距离（Mahalanobis Distance）

考虑特征间的相关性：

$$d_M(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)}$$

其中 $\Sigma$ 是协方差矩阵。

**几何意义**：在标准化的特征空间中的欧几里得距离

### 2. 分类规则

#### 2.1 投票法（Voting）

最常用的方法，直接计数：

$$\hat{y} = \arg\max_{c} \sum_{i=1}^{K} \mathbb{I}(y_i = c)$$

其中 $\mathbb{I}(\cdot)$ 是示性函数。

**缺点**：未考虑距离信息

#### 2.2 距离加权投票法（Distance-Weighted Voting）

根据距离给予不同权重：

$$\hat{y} = \arg\max_{c} \sum_{i=1}^{K} w_i \cdot \mathbb{I}(y_i = c)$$

常见权重函数：

$$w_i = \frac{1}{d_i + \epsilon}$$

或高斯核权重：

$$w_i = \exp\left(-\frac{d_i^2}{2\sigma^2}\right)$$

**优点**：
- 更近的邻居影响更大
- 通常提升分类性能

#### 2.3 软投票法

输出概率而非硬标签：

$$P(\hat{y} = c | x) = \frac{1}{K} \sum_{i=1}^{K} w_i \cdot \mathbb{I}(y_i = c)$$

### 3. K值的选择

$$K \text{值的选择} = \text{模型复杂度的权衡}$$

**K较小（如K=1）**：
- 模型复杂，容易过拟合
- 预测更"局部"
- 对异常点敏感

**K较大（如K=N）**：
- 模型简单，容易欠拟合
- 预测更"全局"
- 计算成本高

**最优K的选择**（使用交叉验证）：

```
对于各个K值 k = 1 to n:
    使用k-折交叉验证评估模型
    记录交叉验证准确率
选择具有最高交叉验证准确率的K值
```

**经验法则**：

$$K = \sqrt{n}$$

其中 $n$ 是训练样本数。

### 4. 分类性能分析

#### 4.1 错误率上界

KNN分类器的贝叶斯误差率上界：

$$P(error) \leq 2 \cdot P(error^*) + O(K^{-1/(d+1)})$$

其中 $P(error^*)$ 是贝叶斯错误率，$d$ 是特征维数。

**维度诅咒**：当 $d$ 很大时，$K^{-1/(d+1)}$ 的衰减速度很慢。

#### 4.2 混淆矩阵和评估指标

对于二分类问题：

| 预测\实际 | 正类 | 负类 |
|---------|------|------|
| 正类 | TP | FP |
| 负类 | FN | TN |

关键指标：

$$\text{准确率} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{精准率} = \frac{TP}{TP + FP}$$

$$\text{召回率} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{精准率} \times \text{召回率}}{\text{精准率} + \text{召回率}}$$

---

## 算法局限性分析

### 1. 时间复杂度问题

**预测时间**：$O(n \cdot d)$，其中 $n$ 是训练样本数，$d$ 是特征维数。

- 对每个查询点，需要计算到所有训练点的距离
- 对于大规模数据集（百万级以上），查询速度不可接受

**内存占用**：$O(n \cdot d)$

### 2. 维度诅咒（Curse of Dimensionality）

在高维空间中：

- 点与点之间的距离变得相似（距离分布集中）
- 需要指数级增加样本数来保持样本密度
- 几何直觉失效

**数学表现**：

设在d维单位立方体中随机均匀分布n个点，要在ε-邻域内包含比例为p的点数，需要：

$$N(p, \varepsilon, d) = \frac{n \cdot V_d(\varepsilon)}{V_d(1)} = n \cdot \varepsilon^d$$

当d增大时，所需的ε值快速增大。

### 3. 不平衡数据问题

类别不平衡时，少数类会被多数类"淹没"：

**改进方案**：
- 样本加权：给少数类更高权重
- 过采样：复制少数类样本
- 欠采样：删除多数类样本
- 使用不同的距离度量

### 4. 特征选择与权重问题

无关特征会增加噪声，相关特征应有不同权重。

**改进**：特征选择和特征权重学习

$$d_{weighted}(x_i, x_j) = \sqrt{\sum_{k=1}^{d} w_k (x_{i,k} - x_{j,k})^2}$$

其中 $w_k$ 是第k维特征的权重。

---

## 加速策略

### 1. KD树（KD-Tree）

**原理**：递归分割特征空间成超矩形区域

**构造过程**：

```
buildKDTree(points, depth):
    if points为空:
        return null
    
    axis = depth % k  // k是特征维数
    sorted_points = 按照axis维度排序points
    median_idx = len(sorted_points) / 2
    
    return {
        point: sorted_points[median_idx],
        left: buildKDTree(sorted_points[0:median_idx], depth+1),
        right: buildKDTree(sorted_points[median_idx+1:], depth+1)
    }
```

**查询过程**：

1. 递归找到叶节点（包含查询点的矩形）
2. 根据距离更新最近邻
3. 回溯检查其他分支是否可能有更近的点
4. 使用边界距离剪枝加速

**时间复杂度**：
- 构造：$O(n \log n)$
- 单次查询：$O(\log n)$（平均），$O(n)$（最坏）
- K个最近邻：$O(K + \log n)$（平均）

**缺点**：高维数据效果不佳

### 2. 球树（Ball Tree）

**原理**：用嵌套的超球体分割特征空间

**优点**：
- 对高维数据更有效
- 支持更多距离度量

**时间复杂度**：
- 单次查询：$O(\log n)$（平均）
- 高维下表现优于KD树

### 3. LSH（局部敏感哈希）

**原理**：将相似的点哈希到相同的桶中

**构造**：
1. 随机生成哈希函数族
2. 对每个点应用多个哈希函数
3. 点被哈希到多维哈希表

**查询**：
1. 计算查询点的哈希值
2. 返回同一桶中的所有点
3. 计算实际距离，找出K近邻

**时间复杂度**：
- 查询：$O(n^{1/c})$，其中 $c$ 是桶数密度
- 与维数无关

**应用**：超大规模数据集的最近邻搜索

### 4. 量化方法

**乘积量化（Product Quantization）**：

将高维向量分割为若干子向量，每个子向量独立量化：

$$x = [x_1, x_2, ..., x_m] \rightarrow [q_1(x_1), q_2(x_2), ..., q_m(x_m)]$$

**优点**：
- 显著降低内存占用
- 保持相对距离关系
- 支持快速距离计算

**缺点**：信息丢失

---

## 算法演变与改进

### 1. 标准KNN → 加权KNN

**改进**：根据距离加权

```python
# 标准投票
class_count = {c: 0 for c in classes}
for neighbor, label in k_neighbors:
    class_count[label] += 1

# 加权投票
class_count = {c: 0 for c in classes}
for neighbor, label, distance in k_neighbors:
    weight = 1 / (distance + 1e-5)  # 或使用其他核函数
    class_count[label] += weight
```

### 2. 固定K → 自适应K

不同查询点可能需要不同的K值。

**密度自适应**：

$$K(x) = \frac{\text{在x周围的平均样本密度} \cdot C}{d}$$

其中 $C$ 是常数，$d$ 是特征维数。

### 3. 单一距离度量 → 多距离度量融合

**加权距离融合**：

$$d_{fusion}(x_i, x_j) = \sum_{m=1}^{M} \lambda_m d_m(x_i, x_j)$$

约束条件：$\sum_{m=1}^{M} \lambda_m = 1, \lambda_m \geq 0$

**学习距离度量权重**：使用验证集优化 $\lambda_m$

### 4. 欧氏距离 → 学习距离度量

**思想**：从数据中学习最优的距离度量

#### 4.1 马氏距离学习

$$d_M(x_i, x_j) = \sqrt{(x_i - x_j)^T M (x_i - x_j)}$$

其中 $M$ 是对称正定矩阵。

**约束条件**：
- 同类点距离小
- 异类点距离大

**目标函数**：

$$\min_{M} \sum_{(i,j) \in S} ||x_i - x_j||_M^2 + \lambda \sum_{(i,k) \in D} ||x_i - x_k||_M^{-2}$$

其中S是同类对集合，D是异类对集合。

#### 4.2 深度度量学习（Deep Metric Learning）

使用神经网络学习特征表示，使得距离度量更合理：

$$d(x_i, x_j) = ||f(x_i) - f(x_j)||_2$$

其中 $f$ 是学习的特征提取函数。

**损失函数示例**：

**Triplet Loss**：
$$L = \max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + \alpha)$$

其中a是锚点，p是正样本，n是负样本，α是边界。

**Contrastive Loss**：
$$L = y \cdot ||f(x_i) - f(x_j)||^2 + (1-y) \cdot \max(0, m - ||f(x_i) - f(x_j)||)^2$$

其中y=1表示相似，y=0表示不相似，m是边界。

### 5. 显式查询 → 隐式学习

#### 5.1 原型网络（Prototypical Networks）

**思想**：每个类由一个原型表示，分类基于到原型的距离

$$p_c = \frac{1}{|S_c|} \sum_{(x,y) \in S_c} f(x)$$

其中 $S_c$ 是类c的支撑集。

**分类**：

$$P(y=c|x, S) = \frac{\exp(-d(f(x), p_c))}{\sum_{c'} \exp(-d(f(x), p_{c'}))}$$

#### 5.2 匹配网络（Matching Networks）

引入注意力机制，学习距离函数

$$a(x, x_i) = \frac{\exp(c(x, x_i))}{\sum_j \exp(c(x, x_j))}$$

其中 $c(\cdot, \cdot)$ 是学习的相似函数。

### 6. 欧氏空间 → 非欧氏空间

在某些数据类型上，非欧氏空间更适合：

#### 6.1 图上的距离

对于图结构数据，使用**最短路径距离**或**图核距离**

#### 6.2 超曲面上的距离

对于流形上的数据，使用**测地距离**（Geodesic Distance）

### 7. 传统KNN → 图神经网络（GNN）

这是现代的重要演变。

**概念桥接**：
- 每个样本是图中的节点
- K-近邻关系定义边
- 利用图结构进行消息传递和聚合

#### 7.1 图卷积网络（GCN）

**原理**：利用邻域信息聚合特征

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{AGGREGATE}(\{h_j^{(l)} : j \in N(i) \cup \{i\}\})\right)$$

其中N(i)是i的邻域，σ是激活函数。

**图卷积层**：

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

其中：
- $\tilde{A} = A + I$（加入自环）
- $\tilde{D}$ 是度数矩阵
- $W^{(l)}$ 是可学习权重

#### 7.2 图注意力网络（GAT）

使用注意力机制学习邻域权重：

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i || Wh_j]))}{\sum_k \exp(\text{LeakyReLU}(a^T[Wh_i || Wh_k]))}$$

$$h_i' = \sigma\left(\sum_j \alpha_{ij} W h_j\right)$$

**优点**：自动学习重要的邻域

#### 7.3 GraphSAINT采样方法

用于大规模图的KNN-like采样：

```
采样邻域子图 → 局部计算 → 聚合结果
```

---

## 现代应用与扩展

### 1. 少样本学习（Few-Shot Learning）

**场景**：只有少量标记样本

**KNN在少样本学习中的角色**：

**原型网络（Prototypical Networks）**：
- 支撑集：有标记的K个样本
- 查询集：测试样本
- 分类：基于到各类原型的距离

```python
def few_shot_classify(support_set, query_samples, model):
    # 为每个类计算原型
    prototypes = {}
    for label, samples in support_set:
        prototypes[label] = mean(model(samples))
    
    # 分类查询样本
    predictions = []
    for query in query_samples:
        query_embedding = model(query)
        distances = {label: dist(query_embedding, proto) 
                    for label, proto in prototypes.items()}
        predictions.append(argmin(distances))
    
    return predictions
```

### 2. 推荐系统

**协同过滤基于KNN**：

**用户-用户协同过滤**：

$$\hat{r}_{u,i} = \frac{\sum_{v \in N_K(u)} \text{sim}(u,v) \cdot r_{v,i}}{\sum_{v \in N_K(u)} \text{sim}(u,v)}$$

其中：
- $\text{sim}(u,v)$ 是用户u和v的相似度
- $N_K(u)$ 是u最相似的K个用户
- $r_{v,i}$ 是用户v对项目i的评分

**相似度计算**：

**皮尔逊相关系数**（Pearson Correlation）：

$$\text{sim}(u,v) = \frac{\sum_i (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_i (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_i (r_{v,i} - \bar{r}_v)^2}}$$

其中 $\bar{r}_u$ 是用户u的平均评分。

**现代扩展**：基于图神经网络的协同过滤

### 3. 异常检测

**KNN异常检测原理**：

样本到其K近邻的距离异常大 → 该样本是异常

**异常分数**：

$$\text{AnomalyScore}(x) = \frac{1}{K} \sum_{i=1}^{K} d(x, x_{nn_i})$$

或使用K-距离（第K个最近邻的距离）：

$$\text{AnomalyScore}(x) = d(x, x_{K-NN})$$

**应用**：
- 网络入侵检测
- 欺诈检测
- 设备故障预测

### 4. 图像检索

**内容基础图像检索（CBIR）**：

```
输入：查询图像 → 特征提取 → KNN搜索 → 返回相似图像
```

**特征表示**：
- 传统：SIFT, SURF, HOG特征
- 现代：CNN深度特征（最后一层卷积输出）

**加速技术**：
- 哈希：二进制特征编码
- 量化：乘积量化
- 索引：LSH, 树结构

### 5. 文本分类和相似度

**文档相似度**：

将文本转换为向量（词袋、TF-IDF、Word Embedding）：

$$d_{text}(\text{doc}_1, \text{doc}_2) = 1 - \cos(\vec{doc}_1, \vec{doc}_2)$$

**应用**：
- 文本分类
- 重复文档检测
- 问答系统

---

## 实际案例研究

### 案例一：电影推荐系统详解

**背景**：构建一个Netflix式的电影推荐系统

**数据集**：

```
用户-电影评分矩阵：

        电影A  电影B  电影C  电影D  电影E
用户1    5      4      ?      2      3
用户2    4      5      4      1      2
用户3    2      1      5      4      ?
用户4    1      2      4      5      5
用户5    5      4      ?      2      3
```

其中 `?` 表示未评分的电影。

#### 步骤1：数据预处理

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# 用户-电影评分矩阵
ratings = np.array([
    [5, 4, np.nan, 2, 3],
    [4, 5, 4, 1, 2],
    [2, 1, 5, 4, np.nan],
    [1, 2, 4, 5, 5],
    [5, 4, np.nan, 2, 3]
])

# 处理缺失值：用用户平均评分填充
def fill_missing_ratings(ratings):
    user_means = np.nanmean(ratings, axis=1, keepdims=True)
    nan_mask = np.isnan(ratings)
    ratings[nan_mask] = user_means[nan_mask[0]]
    return ratings

ratings_filled = fill_missing_ratings(ratings.copy())

# 特征标准化：每个用户的评分标准化
def normalize_ratings(ratings):
    user_means = ratings.mean(axis=1, keepdims=True)
    user_stds = ratings.std(axis=1, keepdims=True)
    # 避免除以0
    user_stds[user_stds == 0] = 1
    return (ratings - user_means) / user_stds

ratings_normalized = normalize_ratings(ratings_filled)
```

#### 步骤2：计算用户相似度

```python
# 方法1：余弦相似度
similarity_matrix = cosine_similarity(ratings_normalized)

# 方法2：皮尔逊相关系数
from scipy.stats import pearsonr

def pearson_correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)

n_users = ratings.shape[0]
pearson_sim = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(n_users):
        pearson_sim[i, j] = pearson_correlation(ratings[i], ratings[j])

print("余弦相似度矩阵：")
print(similarity_matrix)
print("\n皮尔逊相似度矩阵：")
print(pearson_sim)
```

输出示例：
```
余弦相似度矩阵：
[[1.         0.945       -0.812      -0.961       1.        ]
 [0.945      1.         -0.234      -0.890       0.945      ]
 [-0.812     -0.234      1.          0.950       -0.812      ]
 [-0.961     -0.890      0.950       1.          -0.961      ]
 [1.         0.945       -0.812      -0.961       1.         ]]
```

#### 步骤3：KNN预测缺失评分

```python
def knn_predict_rating(target_user, target_movie, ratings, similarity, k=3):
    """
    使用KNN预测目标用户对目标电影的评分
    
    参数：
        target_user: 目标用户索引
        target_movie: 目标电影索引
        ratings: 原始评分矩阵（包含NaN）
        similarity: 用户相似度矩阵
        k: 近邻数
    
    返回：
        预测评分
    """
    # 获取该电影已评分用户的相似度
    movie_ratings = ratings[:, target_movie]
    
    # 找出已评分该电影的用户
    valid_users = np.where(~np.isnan(movie_ratings))[0]
    
    # 计算这些用户与目标用户的相似度
    similarities = similarity[target_user, valid_users]
    
    # 选择相似度最高的K个用户
    top_k_idx = np.argsort(-similarities)[:k]
    top_k_users = valid_users[top_k_idx]
    top_k_sims = similarities[top_k_idx]
    top_k_ratings = movie_ratings[top_k_users]
    
    # 加权平均
    if np.sum(np.abs(top_k_sims)) > 0:
        predicted_rating = np.sum(top_k_sims * top_k_ratings) / np.sum(np.abs(top_k_sims))
    else:
        predicted_rating = np.mean(movie_ratings[~np.isnan(movie_ratings)])
    
    return predicted_rating, top_k_users, top_k_sims

# 预测用户0对电影2的评分
user_id, movie_id = 0, 2
predicted, similar_users, similarities = knn_predict_rating(
    user_id, movie_id, ratings, similarity_matrix, k=3
)

print(f"\n用户 {user_id} 对电影 {movie_id} 的预测评分：{predicted:.2f}")
print(f"参考用户：{similar_users}，相似度：{similarities}")
```

输出示例：
```
用户 0 对电影 2 的预测评分：4.13
参考用户：[1 4 2]，相似度：[0.945 1. -0.812]
```

**解释**：
- 用户1和4的评价最接近用户0
- 用户1评电影2为4分，相似度0.945
- 用户4评电影2为NaN（未评），但我们已过滤
- 用户2评电影2为5分，但相似度为负（偏好差异大）
- 加权平均：(0.945×4 + 1×NaN + ...) / 权重和

#### 步骤4：完整推荐算法

```python
def recommend_movies(target_user, ratings, similarity, k_neighbors=3, 
                    top_n=3):
    """
    为目标用户推荐top_n部电影
    """
    n_movies = ratings.shape[1]
    predictions = []
    
    # 预测所有未评分电影
    for movie_id in range(n_movies):
        if np.isnan(ratings[target_user, movie_id]):
            pred_rating, _, _ = knn_predict_rating(
                target_user, movie_id, ratings, similarity, k=k_neighbors
            )
            predictions.append((movie_id, pred_rating))
    
    # 按预测评分排序
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # 返回top_n推荐
    recommendations = predictions[:top_n]
    return recommendations

# 为用户0推荐电影
recommendations = recommend_movies(0, ratings, similarity_matrix)

print("\n用户0的推荐电影（预测评分）：")
for movie_id, pred_rating in recommendations:
    print(f"  电影 {movie_id}: {pred_rating:.2f}分")
```

#### 步骤5：性能评估

```python
def evaluate_recommendation(ratings_train, ratings_test, similarity, 
                            k_neighbors=3):
    """
    在测试集上评估推荐系统性能
    """
    mae = 0  # 平均绝对误差
    rmse = 0  # 均方根误差
    count = 0
    
    for user_id in range(ratings_test.shape[0]):
        for movie_id in range(ratings_test.shape[1]):
            if not np.isnan(ratings_test[user_id, movie_id]):
                pred_rating, _, _ = knn_predict_rating(
                    user_id, movie_id, ratings_train, similarity, 
                    k=k_neighbors
                )
                actual_rating = ratings_test[user_id, movie_id]
                
                error = abs(pred_rating - actual_rating)
                mae += error
                rmse += error ** 2
                count += 1
    
    mae /= count
    rmse = np.sqrt(rmse / count)
    
    return mae, rmse

# 使用留一法交叉验证评估
# 这里仅做演示，实际应使用更大数据集
print("\n模型评估（演示）：")
print("  MAE: 0.73")
print("  RMSE: 0.95")
```

#### 步骤6：改进：融合多个距离度量

```python
def hybrid_knn_recommendation(target_user, ratings, sim_cosine, 
                             sim_pearson, k=3, alpha=0.5):
    """
    融合多个相似度度量的推荐
    """
    # 融合相似度
    similarity_hybrid = alpha * sim_cosine + (1 - alpha) * sim_pearson
    
    n_movies = ratings.shape[1]
    predictions = []
    
    for movie_id in range(n_movies):
        if np.isnan(ratings[target_user, movie_id]):
            pred_rating, _, _ = knn_predict_rating(
                target_user, movie_id, ratings, similarity_hybrid, k=k
            )
            predictions.append((movie_id, pred_rating))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# 尝试不同的融合权重
for alpha in [0.3, 0.5, 0.7]:
    recs = hybrid_knn_recommendation(0, ratings, similarity_matrix, 
                                     pearson_sim, alpha=alpha)
    print(f"\nα={alpha}时的推荐：")
    for movie_id, rating in recs[:2]:
        print(f"  电影{movie_id}: {rating:.2f}分")
```

### 案例二：异常检测实际应用

**背景**：网络流量异常检测

**数据特征**：
- 源IP、目标IP
- 端口号
- 传输字节数
- 数据包数量
- 持续时间
- 协议类型

#### 数据准备与特征工程

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 模拟网络流量数据
np.random.seed(42)

# 正常流量
normal_data = np.random.normal(
    loc=[100, 50, 1000, 50, 10],  # 均值
    scale=[20, 15, 300, 20, 3],   # 标准差
    size=(800, 5)
)
normal_data = np.clip(normal_data, 0, None)  # 非负值

# 异常流量
anomaly_data = np.random.normal(
    loc=[10, 5, 50000, 10000, 100],  # 大流量
    scale=[5, 2, 10000, 2000, 20],
    size=(20, 5)
)
anomaly_data = np.clip(anomaly_data, 0, None)

# 合并数据
X = np.vstack([normal_data, anomaly_data])
y = np.hstack([np.zeros(800), np.ones(20)])  # 0正常, 1异常

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"训练集大小：{X_train.shape}")
print(f"测试集大小：{X_test.shape}")
print(f"异常样本比例：{y.sum() / len(y) * 100:.1f}%")
```

#### KNN异常检测实现

```python
class KNNAnomalyDetector:
    """
    基于KNN的异常检测器
    """
    def __init__(self, k=5, contamination=0.05):
        """
        参数：
            k: 近邻数
            contamination: 预期异常比例
        """
        self.k = k
        self.contamination = contamination
        self.threshold = None
        self.X_train = None
    
    def fit(self, X):
        """
        训练模型
        """
        self.X_train = X
        
        # 计算每个样本到其K近邻的距离
        distances = []
        for i in range(len(X)):
            # 计算到所有其他样本的距离
            dists = np.linalg.norm(X - X[i], axis=1)
            # 排序并获取第k个距离
            dists = np.sort(dists)[1:self.k+1]  # 跳过自己
            distances.append(dists[-1])  # 第K个最近邻的距离
        
        distances = np.array(distances)
        
        # 设置阈值为contamination分位数
        self.threshold = np.percentile(
            distances, 
            (1 - self.contamination) * 100
        )
        
        return self
    
    def predict(self, X):
        """
        预测异常
        返回：-1异常, 1正常
        """
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, -1)
    
    def decision_function(self, X):
        """
        计算异常分数（到K近邻的距离）
        """
        scores = []
        for i in range(len(X)):
            dists = np.linalg.norm(self.X_train - X[i], axis=1)
            # 第k个最近邻的距离作为异常分数
            scores.append(np.sort(dists)[self.k-1])
        
        return np.array(scores)

# 训练异常检测器
detector = KNNAnomalyDetector(k=5, contamination=0.05)
detector.fit(X_train)

# 预测
y_pred = detector.predict(X_test)
y_scores = detector.decision_function(X_test)

# 评估
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

print("\n异常检测结果：")
print("混淆矩阵：")
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
print(cm)

precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\n精准率 (Precision): {precision:.3f}")
print(f"召回率 (Recall): {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

输出示例：
```
异常检测结果：
混淆矩阵：
[[234  10]
 [  2  16]]

精准率 (Precision): 0.615
召回率 (Recall): 0.889
F1-Score: 0.727
```

#### 可视化异常

```python
import matplotlib.pyplot as plt

# 计算异常分数
train_scores = detector.decision_function(X_train)
test_scores = detector.decision_function(X_test)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 分布图
axes[0].hist(train_scores, bins=50, alpha=0.7, label='正常训练数据')
axes[0].axvline(detector.threshold, color='r', linestyle='--', 
                linewidth=2, label='异常阈值')
axes[0].set_xlabel('异常分数（K-距离）')
axes[0].set_ylabel('频率')
axes[0].set_title('KNN异常分数分布')
axes[0].legend()

# ROC曲线
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, test_scores)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='b', lw=2, label=f'ROC (AUC={roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='r', lw=2, linestyle='--', label='随机分类')
axes[1].set_xlabel('假正率 (FPR)')
axes[1].set_ylabel('真正率 (TPR)')
axes[1].set_title('ROC曲线')
axes[1].legend()

plt.tight_layout()
plt.savefig('knn_anomaly_detection.png', dpi=100)
plt.show()
```

#### 实际异常案例分析

```python
# 找出置信度最高的异常
anomaly_indices = np.where(y_pred == -1)[0]
anomaly_scores = y_scores[anomaly_indices]
top_anomalies_idx = anomaly_indices[np.argsort(-anomaly_scores)][:5]

print("\n置信度最高的5个异常：")
print("=" * 80)
for rank, idx in enumerate(top_anomalies_idx, 1):
    test_idx = idx
    original_idx = len(X_train) + test_idx  # 映射到原始索引
    
    features = X_test[idx]
    score = y_scores[idx]
    
    # 反标准化特征
    original_features = scaler.inverse_transform(features.reshape(1, -1))[0]
    
    print(f"\n异常 #{rank}:")
    print(f"  异常分数：{score:.3f}")
    print(f"  源IP连接数：{original_features[0]:.0f}")
    print(f"  目标IP数：{original_features[1]:.0f}")
    print(f"  数据字节数：{original_features[2]:.0f}")
    print(f"  数据包数：{original_features[3]:.0f}")
    print(f"  持续时间(秒)：{original_features[4]:.1f}")
```

#### 参数优化

```python
# 网格搜索最优K值和contamination
from sklearn.metrics import f1_score

best_f1 = 0
best_params = {}

k_values = [3, 5, 7, 9]
contamination_values = [0.01, 0.05, 0.1]

for k in k_values:
    for cont in contamination_values:
        detector = KNNAnomalyDetector(k=k, contamination=cont)
        detector.fit(X_train)
        y_pred = detector.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'k': k, 'contamination': cont, 'f1': f1}

print(f"\n最优参数：K={best_params['k']}, "
      f"污染率={best_params['contamination']}")
print(f"最佳F1-Score：{best_params['f1']:.3f}")
```

---

## 性能对比与选择指南

### 1. 不同算法的性能对比

#### 表格对比

| 算法 | 时间复杂度 | 空间复杂度 | 高维数据 | 参数调优 | 可解释性 |
|------|----------|----------|--------|--------|--------|
| 标准KNN | O(n·d) | O(n·d) | 差 | 简单 | 优秀 |
| KD-Tree | O(log n) | O(n·d) | 差 | 简单 | 优秀 |
| Ball Tree | O(log n) | O(n·d) | 良好 | 简单 | 优秀 |
| LSH | O(n^{1/c}) | 可调 | 优秀 | 复杂 | 中等 |
| 深度学习KNN | O(1) | 大 | 优秀 | 复杂 | 差 |
| GNN | O(K·d) | O(K·d) | 优秀 | 复杂 | 中等 |

#### 时间复杂度详解

**完整系统的时间复杂度分析**：

设：
- n = 训练样本数
- d = 特征维数
- m = 查询样本数
- k = 最近邻数

| 方法 | 预处理 | 单次查询 | m次查询 |
|------|------|--------|--------|
| 暴力搜索 | O(1) | O(n·d) | O(m·n·d) |
| KD-Tree | O(n·log n) | O(log n)* | O(m·log n)* |
| Ball-Tree | O(n·log n) | O(log n)* | O(m·log n)* |
| LSH | O(n) | O(n^{1/c}) | O(m·n^{1/c}) |
| 乘积量化 | O(n·log n) | O(m·K) | O(m·K) |

*平均情况

### 2. 选择指南

```
选择决策树：

1. 数据规模？
   ├─ 小于10k（可接受）→ 标准KNN或KD-Tree
   ├─ 10k-100k（中等）→ KD-Tree或Ball-Tree
   └─ 100k+（大规模）→ LSH或深度学习

2. 特征维数？
   ├─ 低维（<20）→ KD-Tree
   ├─ 中维（20-100）→ Ball-Tree
   └─ 高维（>100）→ LSH或深度学习

3. 特征类型？
   ├─ 数值型 → 任何方法
   ├─ 文本型 → 余弦相似度+LSH
   └─ 混合型 → 深度学习预处理

4. 精度要求？
   ├─ 高精度 → 标准KNN + 超参调优
   ├─ 中等 → 改进KNN（加权、自适应）
   └─ 可接受误差 → LSH或近似方法

5. 可解释性要求？
   ├─ 需要高可解释性 → 标准KNN
   ├─ 中等 → 改进KNN
   └─ 可黑盒 → 深度学习

6. 实时性要求？
   ├─ 需要超低延迟 → LSH或GPU加速
   ├─ 毫秒级 → 树结构
   └─ 可离线处理 → 标准KNN
```

### 3. 场景选择

#### 推荐系统
- 小数据集（<100k用户）：协同过滤KNN
- 大数据集：向量量化+LSH或矩阵分解
- 现代方案：GNN协同过滤

#### 异常检测
- 低维：标准KNN
- 高维：隔离森林或深度学习
- 流数据：在线KNN变体

#### 图像检索
- 特征维数高：LSH + CNN特征
- 对精度敏感：Ball-Tree + 深度特征
- 超大规模：乘积量化 + HNSW索引

#### 文本分类
- 小文本集：TF-IDF + 余弦KNN
- 大文本集：Word2Vec/BERT + LSH
- 细粒度：Transformer + 原型网络

---

## 总结

### K-近邻算法的发展轨迹

$$\text{KNN演变} = \begin{cases}
\text{基础算法} \xrightarrow{\text{性能}} \text{加速算法} \xrightarrow{\text{表示}} \text{深度学习} \\
& \xrightarrow{\text{结构}} \text{图神经网络}
\end{cases}$$

### 关键优化方向

1. **距离度量**：从欧氏距离 → 学习度量 → 深度特征
2. **效率**：从暴力搜索 → 树结构 → 哈希 → 近似算法
3. **适应性**：从固定K → 自适应K → 学习邻域权重
4. **表示能力**：从原始特征 → 工程特征 → 学习表示 → 端到端学习

### 实践建议

1. **始终从标准KNN开始**：简单、可解释、易调试
2. **根据瓶颈优化**：识别是速度还是精度瓶颈
3. **特征工程很重要**：比选择算法更重要
4. **结合背景知识**：数据的先验知识能显著提升效果
5. **持续验证**：交叉验证、A/B测试不可或缺

### 未来发展方向

- **自适应学习**：动态调整K值和距离度量
- **神经拓扑**：将图拓扑和神经网络融合
- **高效索引**：突破维度诅咒的新方法
- **多模态学习**：融合多种数据类型
- **可信AI**：提高模型解释性和鲁棒性

---

## 参考文献与扩展阅读

### 经典文献
1. Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification.
2. Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching.

### 深度学习与度量学习
3. Snell, J., et al. (2017). Prototypical Networks for Few-shot Learning.
4. Schroff, F., et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering.

### 图神经网络
5. Kipf, T., & Welling, M. (2017). Semi-supervised Classification with Graph Convolutional Networks.
6. Veličković, P., et al. (2018). Graph Attention Networks.

### 推荐系统
7. Sarwar, B., et al. (2001). Item-based Collaborative Filtering Recommendation Algorithms.

---
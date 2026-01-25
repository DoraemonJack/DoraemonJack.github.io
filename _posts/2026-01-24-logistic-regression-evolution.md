---
layout: post
title: "机器学习——逻辑回归"
subtitle: "深度解析逻辑回归的数学原理、算法演变与实战应用"
date: 2026-01-24
author: "DoraemonJack"
header-img: "img/article_bg/2026-01-24-logistic-regression-evolution.webp"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - 机器学习
    - Machine Learning
    - 逻辑回归
    - Logistic Regression
    - 分类算法
    - 数学原理
    - 实战案例
---

# 逻辑回归完全指南：从基础到现代算法演变

---

## 第零部分：前置基础知识完全指南

在深入学习逻辑回归之前，我们需要掌握几个核心概念，这些是理解逻辑回归的基础。本部分将从零开始，详细讲解每个概念背后的数学原理。

### 0.1 线性回归：逻辑回归的起点

#### 0.1.1 问题定义

**线性回归**是机器学习中最基础的有监督学习算法，它用于预测**连续值**。

给定数据集 $\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$，其中：
- $\mathbf{x}_i = [x_{i1}, x_{i2}, ..., x_{id}]^T \in \mathbb{R}^{d}$ 是第 $i$ 个样本的特征向量（$d$ 维）
- $y_i \in \mathbb{R}$ 是对应的连续值标签（价格、温度、身高等）
- $n$ 是样本总数

**目标**：找到一个线性函数来预测 $y$：

$$\hat{y}_i = \mathbf{w}^T\mathbf{x}_i + b$$

其中：
- $\mathbf{w} = [w_1, w_2, ..., w_d]^T \in \mathbb{R}^{d}$ 是权重向量
- $b \in \mathbb{R}$ 是偏置项（截距）

#### 0.1.2 平方损失函数（Mean Squared Error）

为了衡量预测的好坏，我们定义**均方误差（MSE）**：

$$J(\mathbf{w}, b) = \frac{1}{2n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{2n}\sum_{i=1}^{n} (\mathbf{w}^T\mathbf{x}_i + b - y_i)^2$$

**系数 $\frac{1}{2n}$ 的解释**：
- $\frac{1}{n}$ 是为了使损失与样本数量无关（求平均）
- $\frac{1}{2}$ 是为了在求导时消除平方的导数因子，便于计算

**损失函数的性质**：
- 当预测值接近真实值时，$J$ 接近 0
- 当预测值偏离真实值时，$J$ 增大
- 二次函数形式保证了凸性（全局最优解唯一）

#### 0.1.3 几何意义

线性回归在空间中找到一条最佳拟合线（在高维中是超平面），使得所有样本点到这条线的距离平方和最小。

**在二维情况的可视化**：
```
y │     × (实际值)
  │    ╱╱ (预测线)
  │   ╱╱ 
  │  ╱╱
  │ ╱╱
  │╱─────────── x
  └─────────────────
```

#### 0.1.4 闭式解（Closed-form Solution）

对于线性回归，我们可以通过求导找到最优参数的**闭式解**。

**推导过程**：

对 $J$ 对 $\mathbf{w}$ 求偏导：

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{w}^T\mathbf{x}_i + b - y_i)\mathbf{x}_i$$

令导数为 0：

$$\sum_{i=1}^{n} (\mathbf{w}^T\mathbf{x}_i + b - y_i)\mathbf{x}_i = 0$$

**矩阵形式**：

定义设计矩阵：
$$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_n^T \end{bmatrix} \in \mathbb{R}^{n \times d}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^{n}$$

目标函数可写成：
$$J(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2$$

其中 $\mathbf{1} = [1, 1, ..., 1]^T$

**最优解**（忽略偏置项的严格推导）：

$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

这被称为**Normal Equation（正规方程）**。

**时间复杂度分析**：
- 矩阵乘法：$O(nd^2)$
- 矩阵求逆：$O(d^3)$
- 总复杂度：$O(d^3)$

**适用场景**：
- 当 $d$ 较小（$d < 10000$）时非常高效
- 当 $d$ 很大或 $n$ 极大时，计算逆矩阵很困难甚至不稳定

#### 0.1.5 为什么需要梯度下降？

虽然线性回归有闭式解，但：

1. **计算矩阵逆很困难**：当 $\mathbf{X}^T\mathbf{X}$ 接近奇异矩阵时，求逆数值不稳定
2. **高维问题**：$d$ 非常大时，$O(d^3)$ 的复杂度不可接受
3. **在线学习**：无法处理流式数据，只能处理批量数据
4. **非凸问题推广**：逻辑回归的损失函数虽然凸，但没有闭式解

因此，我们需要**梯度下降**这样的迭代优化算法。

---

### 0.2 梯度下降：通用优化算法

#### 0.2.1 核心思想

**梯度下降（Gradient Descent）** 是最常用的优化算法，其核心思想是：

> 在参数空间中，沿着**损失函数下降最快的方向**（即负梯度方向）逐步移动参数，直到达到最优点。

**直观比喻**：想象一个盲人在山脉上，要找到最低点。他无法看全局，但可以感受当前位置周围的坡度，然后沿着最陡的下坡方向走。

#### 0.2.2 梯度的定义与计算

**梯度（Gradient）** 是多元函数偏导数的向量：

$$\nabla J(\mathbf{w}) = \begin{bmatrix} \frac{\partial J}{\partial w_1} \\ \frac{\partial J}{\partial w_2} \\ \vdots \\ \frac{\partial J}{\partial w_d} \end{bmatrix}$$

**梯度的几何意义**：
- 梯度的方向是函数值增长最快的方向
- 梯度的大小（模）表示增长的速率
- 负梯度方向 $-\nabla J(\mathbf{w})$ 是函数值下降最快的方向

**对于线性回归**，梯度为：

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\mathbf{X}^T(\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y})$$

这是一个关于 $\mathbf{w}$ 的 $d$ 维向量。

#### 0.2.3 参数更新规则

梯度下降的**参数更新公式**为：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla J(\mathbf{w}_t)$$

其中：
- $t$ 是迭代次数（轮次）
- $\alpha$ 是**学习率**（Learning Rate），控制步长大小
- $\nabla J(\mathbf{w}_t)$ 是第 $t$ 次迭代时的梯度

**更新过程的可视化**（一维情况）：

```
J(w)
  │
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲
  │  ╱      ╲ (实际函数)
  │ ╱        ╲╲
  │╱          ╲╲___
  └─────────────────── w
      ↑   ↑   ↑
     w₀  w₁  w₂  (收敛到最小值)
```

每次迭代中，$\mathbf{w}$ 都沿着梯度反方向移动。

#### 0.2.4 梯度下降的收敛性分析

**关键问题**：梯度下降是否一定能找到最优解？

对于**凸函数**（Convex Function），梯度下降有以下保证：

**定义**：函数 $f$ 是凸函数，当且仅当：
$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$
其中 $\lambda \in [0,1]$

**Hessian 矩阵判别法**（充要条件）：

函数 $f$ 在区域内凸，当且仅当其 Hessian 矩阵半正定：
$$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial w_1^2} & \frac{\partial^2 f}{\partial w_1 \partial w_2} & \cdots \\ \frac{\partial^2 f}{\partial w_2 \partial w_1} & \frac{\partial^2 f}{\partial w_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} \succeq 0$$

**对于线性回归**：

$$\frac{\partial^2 J}{\partial \mathbf{w}^2} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$$

由于 $\mathbf{X}^T\mathbf{X}$ 总是半正定的（特征值 $\geq 0$），所以 MSE 是凸函数。

**收敛定理**：

对于凸函数，如果学习率 $\alpha$ 满足：
$$\alpha < \frac{2}{L}$$

其中 $L$ 是函数的**Lipschitz 常数**（对于线性回归，$L = \frac{1}{n}\lambda_{max}(\mathbf{X}^T\mathbf{X})$），则梯度下降收敛到全局最优解。

#### 0.2.5 学习率的影响

学习率 $\alpha$ 对梯度下降的表现至关重要：

| 学习率大小 | 特征 | 问题 |
|-----------|------|------|
| $\alpha$ 太小（如 0.0001） | 稳定，方向正确 | 收敛极慢，需要很多迭代次数 |
| $\alpha$ 合适（如 0.01） | 收敛快且稳定 | 需要调参找到最优值 |
| $\alpha$ 太大（如 1.0） | 更新快 | 可能发散，损失函数值越来越大 |
| $\alpha$ 很大（如 10） | 可能跳过最小值 | 震荡或发散 |

**可视化**：

```
损失
  │  α太大：震荡╱╲╱╱╱╱╱╱
  │          ╱  ╲╱
  │  α合适： ╱
  │        ╱───────────
  │  α太小：╱╱╱╱╱╱╱╱╱╱
  └────────────────────── 迭代次数
```

#### 0.2.6 学习率选择的启发式方法

**方法 1：固定学习率**

通常在 $[0.001, 0.1]$ 范围内选择，根据不同问题调整。

**方法 2：学习率衰减（Learning Rate Decay）**

随着迭代进行，逐步减小学习率：

$$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$$

或：

$$\alpha_t = \frac{\alpha_0}{1 + \lambda t}$$

**优点**：早期快速收敛，后期精细调整

**方法 3：自适应学习率**

根据历史梯度信息动态调整学习率，详见后续的 AdaGrad、Adam 等算法。

#### 0.2.7 梯度下降的变体

**1. 批量梯度下降（Batch GD）**

使用所有样本计算梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{1}{n}\sum_{i=1}^{n} \nabla_\mathbf{w} J(\mathbf{w}_t; \mathbf{x}_i, y_i)$$

**优点**：
- 梯度估计准确，收敛稳定
- 可以利用向量化操作（GPU加速）

**缺点**：
- 计算量大，当 $n$ 巨大时内存压力大
- 不支持在线学习
- 难以逃离局部最小值（虽然线性回归无此问题）

**2. 随机梯度下降（SGD）**

每次只使用一个样本计算梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_\mathbf{w} J(\mathbf{w}_t; \mathbf{x}_t, y_t)$$

**优点**：
- 计算快速，内存占用少
- 支持在线学习（流式数据）
- 随机性有助于逃离局部最小值

**缺点**：
- 梯度估计有噪声，收敛不稳定
- 在最小值附近会不停振荡

**3. 小批量梯度下降（Mini-batch GD）**

在批量和随机之间取平衡，使用 $m$ 个样本（通常 $m = 32, 64, 128$）：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{1}{m}\sum_{i \in B_t} \nabla_\mathbf{w} J(\mathbf{w}_t; \mathbf{x}_i, y_i)$$

其中 $B_t$ 是第 $t$ 轮的小批量。

**优点**：兼具批量和随机的优点
- 相对稳定的梯度估计
- 训练速度快
- 利用现代 GPU 的并行计算能力

**缺点**：需要调整批大小这个超参数

---

### 0.3 特征工程：从原始数据到有用特征

#### 0.3.1 特征工程的重要性

> "垃圾数据进，垃圾模型出。"——Andrew Ng

机器学习中有一条黄金法则：

$$\text{好的模型} = \text{算法} + \text{特征工程} + \text{参数调优}$$

其中**特征工程**往往占 60-80% 的重要性！

#### 0.3.2 什么是特征？

**特征（Feature）** 是用来描述数据对象的属性或变量。

**示例**：预测房价时
- 房屋面积（square feet）→ 特征
- 卧室数量 → 特征
- 位置（坐标）→ 特征
- 建造年份 → 特征

这些都是**原始特征**或**一阶特征**。

#### 0.3.3 特征工程的主要任务

**1. 特征选择（Feature Selection）**

从众多原始特征中选出最相关、最有预测力的特征。

**为什么需要**：
- 减少特征维度，降低计算复杂度
- 移除无关特征可以改善泛化性能
- 提高模型可解释性

**方法 1：相关系数分析**

计算每个特征与目标变量的皮尔逊相关系数：

$$\rho(X, y) = \frac{\text{Cov}(X, y)}{\sigma_X \sigma_y}$$

其中：
- $\text{Cov}(X, y)$ 是协方差
- $\sigma_X, \sigma_y$ 分别是标准差

**相关系数的解释**：
- $|\rho| > 0.7$：强相关
- $0.3 < |\rho| < 0.7$：中等相关
- $|\rho| < 0.3$：弱相关

**方法 2：统计显著性检验**

使用 t-检验、卡方检验等判断特征是否对目标有显著影响。

**方法 3：特征重要性评分**

- 对于线性模型：权重的绝对值 $|w_j|$
- 对于树模型：基尼系数下降或熵下降
- 对于深度学习：特征归因方法

**2. 特征转换（Feature Transformation）**

将原始特征转换为更适合模型的形式。

**常见转换**：

a) **对数变换**

当特征呈指数分布或长尾分布时使用：

$$x' = \log(x + 1)$$

（加 1 是为了避免 $\log(0)$）

**示例**：房价通常呈现长尾分布，使用 $\log(\text{price})$ 可以使其更接近正态分布。

b) **平方根变换**

对中等歪斜的数据：

$$x' = \sqrt{x}$$

c) **Box-Cox 变换**

一般化的幂变换：

$$x'(\lambda) = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log(x) & \lambda = 0 \end{cases}$$

其中 $\lambda$ 通过最大似然估计确定。

**3. 特征交互（Feature Interaction）**

创建新特征作为原始特征的组合。

**示例**：

房价预测中：
- 原始特征：面积 ($A$)、每平方英尺价格 ($P/A$)
- 交互特征：总价 = $A \times (P/A)$

**多项式特征**：

对特征 $[x_1, x_2]$，可以构造多项式特征：

$$\phi(\mathbf{x}) = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$$

一般地，对于 $d$ 维输入和 $p$ 阶多项式，特征数为 $\binom{d+p}{p}$。

**警告**：高阶多项式会导致特征爆炸和过拟合！

**4. 特征编码（Feature Encoding）**

将分类变量转换为数值形式。

**方法 1：标签编码（Label Encoding）**

直接映射：
- 红色 → 0
- 绿色 → 1
- 蓝色 → 2

**问题**：模型可能误解为有序关系（红 < 绿 < 蓝）

**方法 2：独热编码（One-Hot Encoding）**

创建虚拟变量：

| 颜色 | 是红色 | 是绿色 | 是蓝色 |
|-----|-------|-------|-------|
| 红色 | 1 | 0 | 0 |
| 绿色 | 0 | 1 | 0 |
| 蓝色 | 0 | 0 | 1 |

**优点**：无序相关的类别处理更正确

**缺点**：特征数增加（如果有 $k$ 个类别，则增加 $k$ 维）

---

### 0.4 特征缩放：数值稳定性的保证

#### 0.4.1 为什么需要特征缩放？

**问题示例**：

假设预测房价，有两个特征：
- 特征 1：房屋面积（范围 500-5000 平方英尺）
- 特征 2：房间数（范围 1-10）

不进行缩放时：
- 面积对应的权重可能是 0.0001
- 房间数对应的权重可能是 100

这会导致：

1. **数值不稳定**：在计算梯度或矩阵求逆时，数值误差大
2. **优化困难**：梯度下降时，损失函数的等高线呈现高度拉伸的椭圆，收敛困难
3. **可解释性差**：权重大小难以比较，因为它们的量纲不同

#### 0.4.2 常见的缩放方法

**1. 标准化（Standardization）- Z-score Normalization**

将特征转换为均值 0、标准差 1 的分布：

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

其中：
- $\mu_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}$ 是第 $j$ 个特征的均值
- $\sigma_j = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij} - \mu_j)^2}$ 是标准差

**数学性质**：

转换后的特征满足：
$$\text{E}[x'_j] = 0, \quad \text{Var}(x'_j) = 1$$

**何时使用**：
- 特征近似服从正态分布
- 使用对特征缩放敏感的算法（如线性回归、逻辑回归、SVM）
- 需要权重可解释性时

**2. 归一化（Normalization）- Min-Max Scaling**

将特征缩放到 $[0, 1]$ 区间：

$$x'_j = \frac{x_j - \min_j}{\max_j - \min_j}$$

其中 $\min_j, \max_j$ 分别是第 $j$ 个特征的最小值和最大值。

**优点**：
- 保留了原始分布的形状
- 特征范围清晰（在 [0, 1] 内）

**缺点**：
- 对异常值很敏感
- 如果新数据超出训练集范围，会产生 < 0 或 > 1 的值

**3. 鲁棒缩放（Robust Scaling）**

使用中位数和四分位数范围（IQR）处理异常值：

$$x'_j = \frac{x_j - \text{median}_j}{\text{IQR}_j}$$

其中 $\text{IQR}_j = Q_3 - Q_1$（第三四分位数减第一四分位数）

**优点**：对异常值不敏感

**4. 均值删除、单位方差缩放**

更为一般的标准化：

$$x'_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$

其中 $\epsilon$ 是小的正数（通常 $10^{-8}$），用于数值稳定性。

#### 0.4.3 数学原理：为什么缩放帮助梯度下降？

**情景分析**：

不缩放时，假设损失函数为：

$$J(w_1, w_2) = (w_1 - 1)^2 + 100(w_2 - 1)^2$$

这对应特征 1 的量纲是 [1, 1]，特征 2 的量纲是 [1, 100]。

**等高线图**：

```
  w₂
   │     椭圆（高度拉伸）
   │    ╱╱╱╱╱╱╱
   │   ╱╱╱╱╱╱╱╱
   │  ╱╱╱╱╱╱╱╱╱
   │ ╱╱╱╱╱╱╱╱╱╱
   └────────────── w₁
```

梯度：

$$\nabla J = \begin{bmatrix} 2(w_1 - 1) \\ 200(w_2 - 1) \end{bmatrix}$$

当 $(w_1, w_2) = (0, 1)$ 时：

$$\nabla J = \begin{bmatrix} -2 \\ 0 \end{bmatrix}$$

梯度方向主要指向 $w_1$ 轴。为了有效收敛，需要：
- 在 $w_1$ 方向上的步长很小（避免在 $w_2$ 方向上振荡）
- 但这样 $w_2$ 的收敛极慢

**缩放后**：

转换后，损失变为：

$$J'(w'_1, w'_2) = w'^2_1 + w'^2_2$$

等高线变成圆形，梯度下降可以直线收敛到最优点。

#### 0.4.4 缩放的注意事项

**1. 分开计算训练集和测试集的统计量**

❌ 错误做法：
```python
# 用整个数据集的均值和标准差缩放
X_all_scaled = (X_all - X_all.mean()) / X_all.std()
```

✅ 正确做法：
```python
# 只用训练集的统计量
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 学习 μ 和 σ
X_test_scaled = scaler.transform(X_test)        # 用学到的参数缩放测试集
```

**原因**：防止数据泄露。测试集应该代表未来的新数据，不应该用其统计信息。

**2. 对 one-hot 编码的特征无需缩放**

因为这些特征已经是 [0, 1] 范围内的二进制值。

**3. 对目标变量是否缩放？**

- **回归问题**：通常需要缩放，便于数值稳定性
- **分类问题**：目标是类别标签，不缩放（或用 one-hot 编码）

---

### 0.5 从线性回归到逻辑回归：为什么需要非线性？

#### 0.5.1 线性回归的局限

虽然线性回归在回归问题中强大，但对于**分类问题**有本质限制。

**问题 1：输出范围不合理**

线性回归输出 $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ 可以是任意实数，但分类问题需要概率输出（在 [0, 1] 范围）。

**示例**：

预测患者是否患病，训练集中：
- 健康人的线性回归输出范围 [-2, 0.5]
- 患病人的线性回归输出范围 [0.3, 3]

虽然有重叠，但无法解释为概率。特别地：
- 如果 $\hat{y} = 5$，无法说"患病概率是 5"（概率不能>1）
- 如果 $\hat{y} = -10$，无法说"患病概率是 -10"（概率不能<0）

**问题 2：决策边界形状固定**

线性回归的决策边界（在二分类中）是线性的。但现实问题中，类边界可能不是线性的。

**示例**：

```
线性数据（线性回归可以）：      非线性数据（线性回归无法）：
  y                             y
  │ ●●●                         │    ●●●
  │ ●●●  ╱╱╱ 决策边界           │  ●● ●●●  ↻ 真实边界
  │      ╱╱╱                    │ ●  ●●  ●
  │ ○○○╱╱╱                      │ ○    ○ ○
  │ ○○○                         │  ○ ○
  └──────────── x              └──────────── x
```

#### 0.5.2 逻辑回归的优雅解决方案

逻辑回归通过引入 **Sigmoid 函数**来解决这些问题：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**Sigmoid 函数的性质**：

1. **值域**：$\sigma(z) \in (0, 1)$，可以解释为概率
2. **单调性**：严格单调递增，保留了线性模型的单调性
3. **光滑性**：处处可微，便于优化
4. **数值稳定性**：无论 $z$ 多大或多小，输出总在 (0, 1) 内

**Sigmoid 函数的数学性质**：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

导数：

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

对称性：

$$\sigma(-z) = 1 - \sigma(z)$$

**推导导数**：

$$\frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$$

#### 0.5.3 Sigmoid 函数的几何理解

```
概率
  1.0 │            ╱─────────────
  0.8 │          ╱╱
  0.6 │        ╱╱
  0.5 │      ╱╱  (决策点：P=0.5)
  0.4 │    ╱╱
  0.2 │  ╱╱
    0 │─────────────
  -5  -2  0  2  5  z
```

**关键特点**：
- $z = 0$ 时，$\sigma(0) = 0.5$（决策边界）
- $z > 0$ 时，$\sigma(z) > 0.5$，倾向于正类
- $z < 0$ 时，$\sigma(z) < 0.5$，倾向于负类
- 中间（$z \in [-2, 2]$）变化最快
- 两端（$|z| > 5$）趋于平坦

#### 0.5.4 概率的完整建模

逻辑回归完整地建模了两个类的概率：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

$$P(y=0|\mathbf{x}) = 1 - \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{e^{-(\mathbf{w}^T\mathbf{x} + b)}}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

**验证**：

$$P(y=1|\mathbf{x}) + P(y=0|\mathbf{x}) = \sigma(z) + (1-\sigma(z)) = 1 \quad \checkmark$$

#### 0.5.5 Odds 和 Log Odds

逻辑回归与**比值对数（Log Odds）**有深刻的数学联系。

**定义**：

**Odds**（比值）：

$$\text{Odds} = \frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \frac{\sigma(z)}{1-\sigma(z)}$$

**展开**：

$$\frac{\sigma(z)}{1-\sigma(z)} = \frac{\frac{1}{1+e^{-z}}}{\frac{e^{-z}}{1+e^{-z}}} = \frac{1}{e^{-z}} = e^z$$

**Log Odds**（对数比值）：

$$\log\left(\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})}\right) = \log(e^z) = z = \mathbf{w}^T\mathbf{x} + b$$

**解释**：

逻辑回归的参数 $\mathbf{w}$ 直接影响比值的对数。这意味着：

- 如果 $w_j > 0$，增加特征 $x_j$ 会增加事件发生的比值（更可能是正类）
- 如果 $w_j < 0$，增加特征 $x_j$ 会减少事件发生的比值（更可能是负类）

**定量关系**：

$$\log(\text{Odds})$ 增加 $\beta$ \Rightarrow \text{Odds}$ 乘以 $e^\beta$$

这提供了直观的可解释性。

---

## 第一部分：逻辑回归基础理论

### 1.1 问题背景与动机

逻辑回归（Logistic Regression）是机器学习中最经典的**监督学习分类算法**之一。尽管名字中带有"回归"，但它本质上是一个**分类模型**。

**为什么需要逻辑回归？**

线性回归用于预测连续值，而许多实际问题需要预测离散的类别标签：
- 邮件是否为垃圾邮件（是/否）
- 患者是否患有某种疾病（是/否）
- 客户是否会购买产品（是/否）
- 贷款申请是否通过（通过/不通过）

这些都是**二分类问题**。如果使用线性回归直接预测，输出值会超出 $[0,1]$ 范围，无法解释为概率。

### 1.2 核心概念：Sigmoid 函数

#### 1.2.1 Sigmoid 函数定义

Sigmoid 函数是逻辑回归的核心，它将任意实数映射到 $(0,1)$ 区间：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**性质分析：**

1. **值域**: $\sigma(z) \in (0, 1)$
2. **单调性**: 严格单调递增
3. **对称性**: $\sigma(-z) = 1 - \sigma(z)$
4. **导数**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

#### 1.2.2 Sigmoid 函数的几何意义

```
σ(z) 的曲线特征：
     1 |                    ╱─────────
       |                 ╱╱
     0.5|             ╱╱
       |          ╱╱
     0 |  ─────╱
       |________________________________
       -5  -2.5   0   2.5   5      z
```

- 当 $z \to +\infty$ 时，$\sigma(z) \to 1$
- 当 $z \to -\infty$ 时，$\sigma(z) \to 0$
- 当 $z = 0$ 时，$\sigma(z) = 0.5$（决策边界）

### 1.3 逻辑回归模型

#### 1.3.1 模型假设

给定样本 $\mathbf{x} \in \mathbb{R}^{d}$，逻辑回归假设：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

其中：
- $\mathbf{w} \in \mathbb{R}^{d}$ 是权重向量
- $b \in \mathbb{R}$ 是偏置项
- $\mathbf{w}^T\mathbf{x} + b$ 称为 **logit** 或 **log odds**

相应地：
$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = \frac{e^{-(\mathbf{w}^T\mathbf{x} + b)}}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

#### 1.3.2 决策函数

预测类别为：
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

等价于：
$$\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w}^T\mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

**几何解释**：$\mathbf{w}^T\mathbf{x} + b = 0$ 定义的超平面是决策边界。

### 1.4 损失函数与极大似然估计

#### 1.4.1 从概率角度推导

给定单个样本 $(x_i, y_i)$，其中 $y_i \in \{0, 1\}$，我们可以写成：

$$P(y_i|\mathbf{x}_i) = p_i^{y_i} (1-p_i)^{1-y_i}$$

其中 $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$

对于 $n$ 个独立同分布的样本，**似然函数**为：

$$L(\mathbf{w}, b) = \prod_{i=1}^{n} p_i^{y_i} (1-p_i)^{1-y_i}$$

#### 1.4.2 对数似然函数

取对数以便于优化：

$$\ell(\mathbf{w}, b) = \sum_{i=1}^{n} [y_i \log p_i + (1-y_i) \log(1-p_i)]$$

展开 $p_i$ 的定义：

$$\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \left[ y_i \log\frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}_i + b)}} + (1-y_i) \log\frac{e^{-(\mathbf{w}^T\mathbf{x}_i + b)}}{1+e^{-(\mathbf{w}^T\mathbf{x}_i + b)}} \right]$$

简化后得到：

$$\ell(\mathbf{w}, b) = \sum_{i=1}^{n} [y_i(\mathbf{w}^T\mathbf{x}_i + b) - \log(1 + e^{\mathbf{w}^T\mathbf{x}_i + b})]$$

#### 1.4.3 交叉熵损失函数

为了最大化对数似然，等价于最小化**交叉熵损失**：

$$J(\mathbf{w}, b) = -\frac{1}{n}\ell(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n} [y_i(\mathbf{w}^T\mathbf{x}_i + b) - \log(1 + e^{\mathbf{w}^T\mathbf{x}_i + b})]$$

或等价形式：

$$J(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} [-y_i \log p_i - (1-y_i) \log(1-p_i)]$$

### 1.5 参数优化：梯度下降

#### 1.5.1 梯度计算

计算 $J$ 对 $\mathbf{w}$ 的梯度：

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)\mathbf{x}_i$$

对 $b$ 的梯度：

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)$$

**推导过程**（链式法则）：

记 $z_i = \mathbf{w}^T\mathbf{x}_i + b$，则：

$$\frac{\partial J}{\partial \mathbf{w}} = -\frac{1}{n}\sum_{i=1}^{n} \frac{\partial}{\partial \mathbf{w}}[y_i z_i - \log(1 + e^{z_i})]$$

$$= -\frac{1}{n}\sum_{i=1}^{n} \left[y_i \mathbf{x}_i - \frac{e^{z_i}}{1+e^{z_i}}\mathbf{x}_i\right]$$

$$= -\frac{1}{n}\sum_{i=1}^{n} (y_i - p_i)\mathbf{x}_i = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)\mathbf{x}_i$$

#### 1.5.2 梯度下降算法

```
输入：训练集 {(x₁,y₁),...,(xₙ,yₙ)}，学习率 α，迭代次数 T
初始化：w ← 0，b ← 0

for t = 1 to T:
    for i = 1 to n:
        pᵢ ← σ(w^T xᵢ + b)
    
    // 计算梯度
    ∇w ← (1/n) Σᵢ(pᵢ - yᵢ)xᵢ
    ∇b ← (1/n) Σᵢ(pᵢ - yᵢ)
    
    // 更新参数
    w ← w - α∇w
    b ← b - α∇b
    
    if 收敛 or t % 检查周期 == 0:
        计算损失函数值，检查收敛条件

返回：w, b
```

#### 1.5.3 收敛性分析

逻辑回归的损失函数 $J(\mathbf{w}, b)$ 是**凸函数**，证明如下：

Hessian 矩阵为：
$$\mathbf{H} = \frac{1}{n}\sum_{i=1}^{n} p_i(1-p_i)\mathbf{x}_i\mathbf{x}_i^T$$

由于 $p_i(1-p_i) \geq 0$ 且 $\mathbf{x}_i\mathbf{x}_i^T$ 是半正定的，因此 $\mathbf{H}$ 是半正定的，证明了凸性。

**结论**：任何局部最优解都是全局最优解，梯度下降必然收敛。

---

## 第二部分：逻辑回归的现代演变

### 2.1 正则化逻辑回归

#### 2.1.1 为什么需要正则化？

在实际应用中，特别是当特征数量很多时，模型容易产生**过拟合**：
- 参数学习过度
- 在训练集表现好，测试集表现差
- 模型对噪声敏感

#### 2.1.2 L2 正则化（岭回归）

在损失函数中添加 L2 惩罚项：

$$J_{L2}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} [-y_i \log p_i - (1-y_i) \log(1-p_i)] + \frac{\lambda}{2n}\|\mathbf{w}\|_2^2$$

其中 $\lambda > 0$ 是**正则化参数**，控制正则化强度。

**梯度更新**：
$$\frac{\partial J_{L2}}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n} (p_i - y_i)\mathbf{x}_i + \frac{\lambda}{n}\mathbf{w}$$

**效果**：
- 较小的 $\lambda$ → 弱约束，模型复杂度高
- 较大的 $\lambda$ → 强约束，模型更平滑，参数更接近 0

#### 2.1.3 L1 正则化（Lasso）

使用 L1 范数惩罚：

$$J_{L1}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} [-y_i \log p_i - (1-y_i) \log(1-p_i)] + \frac{\lambda}{n}\|\mathbf{w}\|_1$$

其中 $\|\mathbf{w}\|_1 = \sum_{j=1}^{d} |w_j|$

**L1 vs L2 对比**：

| 特性 | L1 正则化 | L2 正则化 |
|------|---------|---------|
| 惩罚方式 | $\sum \|w_j\|$ | $\sum w_j^2$ |
| 特征选择 | 能产生稀疏解 | 不产生稀疏解 |
| 计算效率 | 较复杂 | 简单 |
| 应用场景 | 特征筛选 | 一般使用 |

#### 2.1.4 Elastic Net（弹性网络）

结合 L1 和 L2 正则化：

$$J_{EN}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} [-y_i \log p_i - (1-y_i) \log(1-p_i)] + \frac{\lambda_1}{n}\|\mathbf{w}\|_1 + \frac{\lambda_2}{2n}\|\mathbf{w}\|_2^2$$

**优势**：
- 兼具 L1 的特征选择能力和 L2 的稳定性
- 当特征高度相关时表现更好

### 2.2 多分类逻辑回归

#### 2.2.1 多类别问题

当 $y \in \{1, 2, ..., K\}$（$K > 2$）时，需要推广逻辑回归到多分类。

#### 2.2.2 Softmax 回归（多项逻辑回归）

模型假设：
$$P(y=k|\mathbf{x}) = \text{softmax}_k(\mathbf{w}) = \frac{e^{\mathbf{w}_k^T\mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x} + b_j}}$$

其中每个类 $k$ 都有自己的参数向量 $\mathbf{w}_k$ 和偏置 $b_k$。

**性质**：
- $\sum_{k=1}^{K} P(y=k|\mathbf{x}) = 1$
- 是 sigmoid 在多分类情况下的推广

#### 2.2.3 交叉熵损失

对于多分类，交叉熵损失为：

$$J(\mathbf{W}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log p_{ik}$$

其中 $y_{ik}$ 是 one-hot 编码，$p_{ik} = P(y_i = k | \mathbf{x}_i)$

#### 2.2.4 一对多（One-vs-Rest）策略

当无法直接使用 softmax 时，可以训练 $K$ 个二分类器：
- 第 $k$ 个分类器：类别 $k$ vs 其他类别
- 预测时选择置信度最高的分类器

### 2.3 概率校准与不确定性估计

#### 2.3.1 为什么需要校准？

原始的逻辑回归输出概率可能不能准确反映真实概率。例如：
- 模型预测某事件概率为 0.7，但统计上该置信度下事件发生概率只有 0.5

#### 2.3.2 Platt 缩放

一种简单的校准方法。将模型输出 $\hat{p}$ 通过 sigmoid 进一步变换：

$$p_{calib} = \sigma(A\hat{p} + B)$$

其中 $A, B$ 通过验证集上的最大似然估计学习。

#### 2.3.3 同位素回归

使用 pool adjacent violators 算法（PAVA）进行单调校准。

### 2.4 分布式与在线学习

#### 2.4.1 随机梯度下降（SGD）

用单个样本的梯度估计全体梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha_t \nabla J(\mathbf{w}_t; \mathbf{x}_{t}, y_t)$$

**优势**：
- 内存需求少
- 适合大规模数据
- 天然支持在线学习

#### 2.4.2 小批量梯度下降（Mini-batch GD）

结合批量梯度下降和 SGD：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha_t \frac{1}{m}\sum_{i \in B_t} \nabla J(\mathbf{w}_t; \mathbf{x}_i, y_i)$$

其中 $B_t$ 是大小为 $m$ 的小批量。

#### 2.4.3 自适应学习率方法

**AdaGrad** - 动态调整学习率：
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \mathbf{g}_t$$

其中 $\mathbf{G}_t = \sum_{\tau=1}^{t} \mathbf{g}_\tau \odot \mathbf{g}_\tau$

**Adam** - 结合动量和自适应学习率：
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha\frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}$$

### 2.5 对不平衡数据的处理

#### 2.5.1 问题定义

当正负样本比例严重不均时（如欺诈检测中 99% 都是正常交易），标准逻辑回归会偏向多数类。

#### 2.5.2 类权重调整

修改损失函数：

$$J_{weighted}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} w_{y_i}[-y_i \log p_i - (1-y_i) \log(1-p_i)]$$

其中：
$$w_1 = \frac{n}{2n_1}, \quad w_0 = \frac{n}{2n_0}$$

$n_0, n_1$ 分别是负、正样本数量。

#### 2.5.3 重采样策略

**过采样（Oversampling）**：复制少数类样本
$$\text{多数类样本数} = \text{少数类样本数} \times r$$

**欠采样（Undersampling）**：删除多数类样本
$$\text{多数类样本数} = \text{少数类样本数} \times r$$

**SMOTE（合成少数类过采样技术）**：
为每个少数类样本生成合成邻居
$$\mathbf{x}_{new} = \mathbf{x}_i + \text{rand}(0,1) \times (\mathbf{x}_{neighbor} - \mathbf{x}_i)$$

---

## 第三部分：代码实现

### 3.1 从零实现逻辑回归

#### 3.1.1 Python 基础实现

```python
import numpy as np
from scipy.special import expit  # sigmoid 函数
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LogisticRegression:
    """基础逻辑回归实现"""
    
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_reg=0.0):
        """
        初始化逻辑回归模型
        
        参数：
            learning_rate: 学习率
            iterations: 最大迭代次数
            lambda_reg: L2正则化参数
        """
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.losses = []
        
    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        return expit(z)  # 数值稳定的实现
    
    def compute_loss(self, X, y):
        """计算交叉熵损失 + L2正则化"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        p = self.sigmoid(z)
        
        # 避免 log(0)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        
        # 交叉熵损失
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        # 添加 L2 正则化
        reg_loss = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        
        return loss + reg_loss
    
    def compute_gradients(self, X, y):
        """计算梯度"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        p = self.sigmoid(z)
        
        # 梯度
        dw = np.dot(X.T, (p - y)) / m + (self.lambda_reg / m) * self.weights
        db = np.sum(p - y) / m
        
        return dw, db
    
    def fit(self, X, y, verbose=True):
        """
        训练模型
        
        参数：
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            verbose: 是否打印进度
        """
        m, n = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n)
        self.bias = 0
        self.losses = []
        
        # 梯度下降
        for iteration in range(self.iterations):
            # 计算梯度
            dw, db = self.compute_gradients(X, y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 记录损失
            loss = self.compute_loss(X, y)
            self.losses.append(loss)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class LogisticRegressionWithOptimize:
    """使用 scipy.optimize 优化的实现"""
    
    def __init__(self, lambda_reg=0.0):
        self.weights = None
        self.bias = None
        self.lambda_reg = lambda_reg
    
    def sigmoid(self, z):
        return expit(z)
    
    def objective(self, params, X, y):
        """目标函数：损失函数"""
        m, n = X.shape
        w = params[:n]
        b = params[n]
        
        z = np.dot(X, w) + b
        p = self.sigmoid(z)
        
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = (self.lambda_reg / (2 * m)) * np.sum(w ** 2)
        
        return loss + reg
    
    def gradient(self, params, X, y):
        """目标函数的梯度"""
        m, n = X.shape
        w = params[:n]
        b = params[n]
        
        z = np.dot(X, w) + b
        p = self.sigmoid(z)
        
        dw = np.dot(X.T, (p - y)) / m + (self.lambda_reg / m) * w
        db = np.sum(p - y) / m
        
        return np.concatenate([dw, [db]])
    
    def fit(self, X, y):
        """使用 L-BFGS 优化器训练"""
        m, n = X.shape
        params0 = np.zeros(n + 1)
        
        result = minimize(
            self.objective,
            params0,
            args=(X, y),
            method='L-BFGS-B',
            jac=self.gradient,
            options={'disp': False}
        )
        
        self.weights = result.x[:n]
        self.bias = result.x[n]
        
        return self
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# 多分类逻辑回归 (Softmax)
class SoftmaxRegression:
    """多分类逻辑回归"""
    
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_reg=0.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.weights = None  # shape: (n_features, n_classes)
        self.bias = None     # shape: (n_classes,)
        self.classes = None
        self.losses = []
    
    def softmax(self, z):
        """Softmax 函数"""
        # 数值稳定：减去最大值
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """训练多分类模型"""
        m, n = X.shape
        self.classes = np.unique(y)
        K = len(self.classes)
        
        # 将标签转换为 one-hot 编码
        y_onehot = np.zeros((m, K))
        for i, label in enumerate(self.classes):
            y_onehot[y == label, i] = 1
        
        # 初始化参数
        self.weights = np.random.randn(n, K) * 0.01
        self.bias = np.zeros(K)
        
        # 梯度下降
        for iteration in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            p = self.softmax(z)
            
            # 计算梯度
            dw = np.dot(X.T, (p - y_onehot)) / m + (self.lambda_reg / m) * self.weights
            db = np.sum(p - y_onehot, axis=0) / m
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 计算损失
            loss = -np.mean(y_onehot * np.log(np.clip(p, 1e-15, 1)))
            self.losses.append(loss)
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes[pred_indices]
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
```

#### 3.1.2 使用 scikit-learn 的高级实现

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# 1. 基础使用
model = LogisticRegression(
    penalty='l2',           # 正则化类型：'l1', 'l2', 'elasticnet'
    C=1.0,                  # 正则化强度的倒数（C越小，正则化越强）
    solver='lbfgs',         # 优化器：'lbfgs', 'liblinear', 'saga', 'newton-cg'
    max_iter=1000,
    random_state=42
)

# 2. 多分类
model_multiclass = LogisticRegression(
    multi_class='multinomial',  # 多分类策略
    solver='lbfgs'
)

# 3. 处理不平衡数据
model_balanced = LogisticRegression(
    class_weight='balanced'  # 自动调整权重
)

# 4. 概率预测
y_proba = model.predict_proba(X_test)  # 返回每个类的概率
```

### 3.2 实战案例：乳腺癌诊断

详见第四部分。

---

## 第四部分：实战案例 - 乳腺癌诊断系统

### 4.1 问题描述

**背景**：乳腺癌是全球女性最常见的癌症之一。早期诊断对提高生存率至关重要。

**目标**：基于临床测量特征（如肿块硬度、大小等），构建分类模型预测肿块是否为恶性。

**数据集**：使用经典的 Breast Cancer Wisconsin 数据集
- 569 个样本
- 30 个特征（细胞核特性）
- 2 个类别：恶性（Malignant） = 1，良性（Benign） = 0

### 4.2 详细实现步骤

#### 步骤 1：数据加载与探索

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"数据集大小: {X.shape}")
print(f"类别分布: {np.bincount(y)}")
print(f"特征名称:\n{feature_names}")

# 创建 DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 基本统计
print("\n数据统计信息:")
print(df.describe())

# 可视化类别分布
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.countplot(x=df['target'])
plt.xlabel('诊断结果')
plt.ylabel('样本数')
plt.title('样本类别分布')
plt.xticks([0, 1], ['良性', '恶性'])

plt.subplot(1, 2, 2)
# 计算正负样本比例
n_negative = np.sum(y == 0)
n_positive = np.sum(y == 1)
plt.pie([n_negative, n_positive], labels=['良性', '恶性'], autopct='%1.1f%%')
plt.title('类别比例')

plt.tight_layout()
plt.show()

# 特征相关性分析
plt.figure(figsize=(15, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('特征相关性矩阵')
plt.show()

# 前 10 个特征与目标的相关性
plt.figure(figsize=(10, 6))
target_corr = correlation_matrix['target'].drop('target').sort_values(ascending=False)
plt.barh(range(len(target_corr[:10])), target_corr[:10].values)
plt.yticks(range(len(target_corr[:10])), target_corr[:10].index)
plt.xlabel('与诊断结果的相关系数')
plt.title('前 10 个最相关的特征')
plt.tight_layout()
plt.show()
```

#### 步骤 2：数据预处理

```python
# 特征标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.3,        # 70% 训练，30% 测试
    random_state=42,
    stratify=y            # 保持类别比例
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集正负比例: {np.bincount(y_train)}")
print(f"测试集正负比例: {np.bincount(y_test)}")
```

#### 步骤 3：模型训练与评估

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

# 训练基础模型
print("="*60)
print("1. 基础逻辑回归模型")
print("="*60)

model_basic = LogisticRegression(
    max_iter=10000,
    random_state=42
)
model_basic.fit(X_train, y_train)

y_pred_basic = model_basic.predict(X_test)
y_proba_basic = model_basic.predict_proba(X_test)[:, 1]

print(f"训练集准确率: {model_basic.score(X_train, y_train):.4f}")
print(f"测试集准确率: {model_basic.score(X_test, y_test):.4f}")

# 详细评估指标
print("\n测试集评估指标:")
print(f"准确率 (Accuracy):  {accuracy_score(y_test, y_pred_basic):.4f}")
print(f"精准率 (Precision): {precision_score(y_test, y_pred_basic):.4f}")
print(f"召回率 (Recall):    {recall_score(y_test, y_pred_basic):.4f}")
print(f"F1 分数:            {f1_score(y_test, y_pred_basic):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_basic)
print("\n混淆矩阵:")
print(f"[[真负数 {cm[0,0]:3d}  假正数 {cm[0,1]:3d}]")
print(f" [假负数 {cm[1,0]:3d}  真正数 {cm[1,1]:3d}]]")

# 医学意义
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # 召回率（敏感度）
specificity = tn / (tn + fp)  # 特异性
print(f"\n医学指标:")
print(f"敏感度 (Sensitivity): {sensitivity:.4f}  (找到患者的比例)")
print(f"特异性 (Specificity): {specificity:.4f}  (正确识别健康人的比例)")
```

#### 步骤 4：ROC 曲线分析

```python
# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba_basic)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"AUC 分数: {roc_auc:.4f}")
```

#### 步骤 5：正则化对比

```python
print("\n" + "="*60)
print("2. 不同正则化强度的比较")
print("="*60)

# 不同的正则化参数
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
results = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    results.append({
        'C': C,
        'train_acc': train_acc,
        'test_acc': test_acc
    })
    
    print(f"C={C:8.3f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

# 可视化
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
plt.semilogx(results_df['C'], results_df['train_acc'], 'o-', label='训练集', linewidth=2)
plt.semilogx(results_df['C'], results_df['test_acc'], 's-', label='测试集', linewidth=2)
plt.xlabel('正则化参数 C (越小正则化越强)')
plt.ylabel('准确率')
plt.title('正则化强度对模型性能的影响')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### 步骤 6：L1 vs L2 正则化

```python
print("\n" + "="*60)
print("3. L1 vs L2 正则化对比")
print("="*60)

# L1 正则化
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=10000)
model_l1.fit(X_train, y_train)

# L2 正则化
model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=10000)
model_l2.fit(X_train, y_train)

print(f"L1 正则化 - 训练准确率: {model_l1.score(X_train, y_train):.4f}, 测试准确率: {model_l1.score(X_test, y_test):.4f}")
print(f"L2 正则化 - 训练准确率: {model_l2.score(X_train, y_train):.4f}, 测试准确率: {model_l2.score(X_test, y_test):.4f}")

# 特征权重稀疏性
n_zero_l1 = np.sum(model_l1.coef_[0] == 0)
n_zero_l2 = np.sum(model_l2.coef_[0] == 0)

print(f"\nL1 正则化 - 零权重特征数: {n_zero_l1}/{len(model_l1.coef_[0])}")
print(f"L2 正则化 - 零权重特征数: {n_zero_l2}/{len(model_l2.coef_[0])}")

# 可视化权重
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# L1
coef_l1_sorted_idx = np.argsort(np.abs(model_l1.coef_[0]))[-15:]
axes[0].barh(range(len(coef_l1_sorted_idx)), model_l1.coef_[0][coef_l1_sorted_idx])
axes[0].set_yticks(range(len(coef_l1_sorted_idx)))
axes[0].set_yticklabels([feature_names[i] for i in coef_l1_sorted_idx])
axes[0].set_xlabel('权重值')
axes[0].set_title('L1 正则化 - 权重最大的 15 个特征')
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# L2
coef_l2_sorted_idx = np.argsort(np.abs(model_l2.coef_[0]))[-15:]
axes[1].barh(range(len(coef_l2_sorted_idx)), model_l2.coef_[0][coef_l2_sorted_idx])
axes[1].set_yticks(range(len(coef_l2_sorted_idx)))
axes[1].set_yticklabels([feature_names[i] for i in coef_l2_sorted_idx])
axes[1].set_xlabel('权重值')
axes[1].set_title('L2 正则化 - 权重最大的 15 个特征')
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 步骤 7：交叉验证

```python
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import make_scorer

# 5 折交叉验证
model_cv = LogisticRegression(max_iter=10000, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(model_cv, X_train, y_train, cv=5, scoring=scoring)

print("="*60)
print("4. 5 折交叉验证结果")
print("="*60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    scores = cv_results[f'test_{metric}']
    print(f"{metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 交叉验证预测
y_cv_pred = cross_val_predict(model_cv, X_train, y_train, cv=5)
print("\n交叉验证混淆矩阵:")
print(confusion_matrix(y_train, y_cv_pred))
```

#### 步骤 8：概率校准

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# 原始模型的校准曲线
prob_true_raw, prob_pred_raw = calibration_curve(
    y_test, y_proba_basic, n_bins=10
)

# 使用 Platt 缩放进行校准
calibrated_model = CalibratedClassifierCV(
    model_basic, 
    method='sigmoid',  # Platt 缩放
    cv=5
)
calibrated_model.fit(X_train, y_train)
y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

prob_true_cal, prob_pred_cal = calibration_curve(
    y_test, y_proba_calibrated, n_bins=10
)

# 绘制校准曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, prob_true, prob_pred, title in [
    (axes[0], prob_true_raw, prob_pred_raw, '原始模型'),
    (axes[1], prob_true_cal, prob_pred_cal, '校准后模型')
]:
    ax.plot([0, 1], [0, 1], 'k--', label='理想校准')
    ax.plot(prob_pred, prob_true, 's-', label='实际校准')
    ax.set_xlabel('预测概率')
    ax.set_ylabel('实际概率')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()

print("校准前后 AUC 对比:")
print(f"原始模型: {auc(fpr, tpr):.4f}")
fpr_cal, tpr_cal, _ = roc_curve(y_test, y_proba_calibrated)
print(f"校准模型: {auc(fpr_cal, tpr_cal):.4f}")
```

#### 步骤 9：处理不平衡数据

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

print("\n" + "="*60)
print("5. 不平衡数据处理")
print("="*60)

# 原始数据不平衡情况
print(f"原始训练集类别分布: {np.bincount(y_train)}")

# 方法 1：类权重
model_weighted = LogisticRegression(
    class_weight='balanced',
    max_iter=10000,
    random_state=42
)
model_weighted.fit(X_train, y_train)

print(f"\n方法 1: 类权重调整")
print(f"训练准确率: {model_weighted.score(X_train, y_train):.4f}")
print(f"测试准确率: {model_weighted.score(X_test, y_test):.4f}")

# 方法 2：SMOTE 过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(max_iter=10000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

print(f"\n方法 2: SMOTE 过采样")
print(f"采样后训练集类别分布: {np.bincount(y_train_smote)}")
print(f"测试准确率: {model_smote.score(X_test, y_test):.4f}")

# 方法 3：组合过采样和欠采样
pipeline_smote = ImbPipeline([
    ('over', SMOTE(random_state=42)),
    ('under', RandomUnderSampler(random_state=42))
])
X_train_resampled, y_train_resampled = pipeline_smote.fit_resample(X_train, y_train)

model_combined = LogisticRegression(max_iter=10000, random_state=42)
model_combined.fit(X_train_resampled, y_train_resampled)

print(f"\n方法 3: SMOTE + 欠采样")
print(f"采样后训练集类别分布: {np.bincount(y_train_resampled)}")
print(f"测试准确率: {model_combined.score(X_test, y_test):.4f}")

# 对比三种方法
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models_compare = [
    (model_weighted, "类权重", axes[0]),
    (model_smote, "SMOTE", axes[1]),
    (model_combined, "SMOTE+欠采样", axes[2])
]

for model, title, ax in models_compare:
    y_pred_compare = model.predict(X_test)
    cm_compare = confusion_matrix(y_test, y_pred_compare)
    
    sns.heatmap(cm_compare, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(f'{title}\nAcc: {accuracy_score(y_test, y_pred_compare):.3f}')
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')

plt.tight_layout()
plt.show()
```

#### 步骤 10：特征重要性分析

```python
# 最终模型权重分析
model_final = LogisticRegression(max_iter=10000, random_state=42)
model_final.fit(X_train, y_train)

# 获取权重
weights = model_final.coef_[0]
feature_importance = np.abs(weights)

# 排序
sorted_idx = np.argsort(feature_importance)[::-1]

# 绘制特征重要性
fig, ax = plt.subplots(figsize=(10, 8))
top_features_idx = sorted_idx[:15]
ax.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
ax.set_yticks(range(len(top_features_idx)))
ax.set_yticklabels([feature_names[i] for i in top_features_idx])
ax.set_xlabel('特征权重绝对值')
ax.set_title('逻辑回归特征重要性（Top 15）')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

print("最重要的 10 个特征:")
for i, idx in enumerate(sorted_idx[:10]):
    print(f"{i+1:2d}. {feature_names[idx]:30s} - 权重: {weights[idx]:7.4f}")
```

### 4.3 实际应用指导

#### 临床使用场景

假设医院想要使用这个模型来辅助诊断：

```python
# 新患者的检查数据（需要标准化）
new_patient_raw = np.array([[17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 
                              0.1471, 0.2419, 0.07871, 1.095, 1.105, 8.589, 153.4, 
                              0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                              25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 
                              0.2654, 0.4601, 0.11890]])

# 标准化
new_patient = scaler.transform(new_patient_raw)

# 预测
pred_class = model_final.predict(new_patient)[0]
pred_prob = model_final.predict_proba(new_patient)[0]

print("="*60)
print("患者诊断结果")
print("="*60)
print(f"预测类别: {'恶性' if pred_class == 1 else '良性'}")
print(f"良性概率: {pred_prob[0]:.1%}")
print(f"恶性概率: {pred_prob[1]:.1%}")

# 临床建议
if pred_prob[1] >= 0.8:
    print("\n临床建议: 强烈建议进一步深入检查和治疗")
elif pred_prob[1] >= 0.5:
    print("\n临床建议: 建议进一步医学检查确认")
else:
    print("\n临床建议: 暂无异常发现，建议定期复查")
```

#### 模型部署

```python
import pickle
import json

# 保存模型
with open('logistic_regression_cancer_model.pkl', 'wb') as f:
    pickle.dump(model_final, f)

# 保存标准化器
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 创建模型卡
model_card = {
    "model_name": "乳腺癌诊断逻辑回归模型",
    "version": "1.0",
    "performance": {
        "accuracy": float(model_final.score(X_test, y_test)),
        "auc": float(roc_auc),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp))
    },
    "features_used": len(feature_names),
    "training_data_size": len(X_train),
    "class_labels": ["良性", "恶性"]
}

with open('model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)

print("模型已保存！")
print(json.dumps(model_card, indent=2, ensure_ascii=False))
```

---

## 第五部分：逻辑回归的局限与未来方向

### 5.1 逻辑回归的局限性

1. **假设线性可分**
   - 只能处理线性可分的问题
   - 对于复杂的非线性关系表现不佳

2. **对异常值敏感**
   - 极端值会对决策边界产生显著影响

3. **特征工程依赖**
   - 需要手动构造交互特征和非线性变换

4. **不支持自动特征学习**
   - 与深度学习方法相比，无法自动学习特征表示

### 5.2 现代替代方案

**1. 核逻辑回归（Kernel Logistic Regression）**
- 通过核技巧处理非线性问题
- 保留逻辑回归的可解释性

**2. 支持向量机（SVM）**
- 通过最大间隔思想改进
- 更好处理高维数据

**3. 决策树与随机森林**
- 自动进行特征交互
- 更强的非线性建模能力

**4. 神经网络**
- 单层神经网络等价于逻辑回归
- 多层网络可学习复杂非线性关系

**5. 梯度提升模型（XGBoost, LightGBM）**
- 集成多个弱分类器
- 状态最强的表现性能

### 5.3 何时使用逻辑回归

✅ **适合场景**：
- 特征与目标基本线性相关
- 需要高可解释性（医疗、金融等）
- 数据量不足以训练复杂模型
- 对模型复杂度有严格限制
- 需要概率输出和置信度估计

❌ **不适合场景**：
- 特征间存在复杂非线性交互
- 数据量充足且特征维度高
- 模型黑箱性可接受
- 对性能要求极高

---

## 附录：常见问题与解答

### Q1：如何选择合适的学习率？

**A**：
- 初始尝试：0.01 - 0.1 之间
- 过大会导致发散，过小收敛缓慢
- 可以使用学习率衰减策略：$\alpha_t = \alpha_0 \cdot e^{-t/\tau}$

### Q2：正则化参数 λ 如何选择？

**A**：
- 网格搜索：$\lambda \in \{0, 0.001, 0.01, 0.1, 1, 10\}$
- 交叉验证选择最优值
- sklearn 中用 C（$C = 1/\lambda$）表示

### Q3：样本不平衡如何处理？

**A**：
1. 类权重调整：`class_weight='balanced'`
2. SMOTE 过采样
3. 阈值移动
4. 集成方法

### Q4：如何解释模型权重？

**A**：
- 正权重：特征增大 → 正类概率增大
- 负权重：特征增大 → 正类概率减小
- 权重绝对值：特征重要性

---

## 总结

逻辑回归从最初的简单模型，发展到今天包括：
- 各种正则化方法（L1, L2, Elastic Net）
- 多分类扩展（Softmax）
- 概率校准技术
- 不平衡数据处理方法
- 自适应优化器

尽管有现代深度学习方法，但逻辑回归因其**可解释性强**、**计算效率高**、**鲁棒性好**等特点，在医学诊断、金融风控等关键领域仍然被广泛使用。

掌握逻辑回归的理论与实践，不仅能解决实际问题，还能为学习更复杂的模型打下坚实基础。

---

**参考文献**：
1. Bishop, C. M. (2006). Pattern recognition and machine learning.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning.
4. Ng, A. (2020). Machine Learning Yearning.

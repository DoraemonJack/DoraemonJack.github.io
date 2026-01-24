---
layout:       post
title:        "最优化理论——牛顿法和拟牛顿法"
subtitle:     "从理论推导到算法实现的完整解析"
date:         2025-09-13 10:00:00
author:       "DoraemonJack"
header-style: text
header-img: "img/post-bg-math.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - Optimization
    - 最优化算法
    - Newton Method
    - Quasi-Newton
    - Numerical Methods
---

本文深入探讨牛顿法和拟牛顿法的理论基础、算法推导和实际应用。从经典的牛顿法出发，逐步介绍BFGS、DFP等拟牛顿算法，并通过详细的例题演示这些方法的计算过程和收敛特性。

<div class="alert alert-info" role="alert">
  <strong>前置阅读：</strong>
  <a href="/2025/09/11/steepest-descent-and-conjugate-gradient/" target="_blank">最优化方法——最速下降法与共轭梯度法</a>
  <span style="margin-left:8px">（建议先掌握基本优化方法）</span>
</div>

## 一、牛顿法的理论基础与核心原理

### 1.1 基本思想与几何直观

牛顿法是求解无约束优化问题的经典方法，其核心思想是**利用目标函数的二阶信息（Hessian矩阵）来构造二次近似模型**，从而获得比一阶方法更快的收敛速度。

**几何直观**：
- **一阶方法**（如最速下降法）：只使用梯度信息，沿着最陡下降方向前进
- **牛顿法**：同时使用梯度和曲率信息，能够"预测"函数的弯曲程度，选择更智能的搜索方向

对于优化问题：
$$\min_{x \in \mathbb{R}^n} f(x)$$

**核心思想**：在当前点 $$x_k$$ 附近用二次函数近似原函数，然后求解这个二次函数的最优解作为下一个迭代点。

### 1.2 Taylor展开与二次近似

在点 $$x_k$$ 处，将 $$f(x)$$ 进行**二阶Taylor展开**：
$$f(x) \approx f(x_k) + \nabla f(x_k)^T (x - x_k) + \frac{1}{2}(x - x_k)^T \nabla^2 f(x_k) (x - x_k)$$

其中：
- $$\nabla f(x_k)$$：梯度向量（一阶偏导数）
- $$\nabla^2 f(x_k)$$：Hessian矩阵（二阶偏导数矩阵）

**Hessian矩阵的定义**：
$$H_{ij}(x) = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}$$

$$\nabla^2 f(x) = H(x) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

### 1.3 牛顿法迭代公式的详细推导

设 $$g_k = \nabla f(x_k)$$，$$H_k = \nabla^2 f(x_k)$$，则**二次近似函数**为：
$$q_k(x) = f(x_k) + g_k^T (x - x_k) + \frac{1}{2}(x - x_k)^T H_k (x - x_k)$$

**步骤1**：求二次近似函数的最优解

要找到 $$q_k(x)$$ 的最小值，对 $$x$$ 求导：
$$\nabla_x q_k(x) = g_k + H_k (x - x_k)$$

令梯度为零（最优性条件）：
$$\nabla_x q_k(x) = g_k + H_k (x - x_k) = 0$$

**步骤2**：解线性方程组

从上式可得：
$$H_k (x - x_k) = -g_k$$

如果 $$H_k$$ 可逆，则：
$$x - x_k = -H_k^{-1} g_k$$

**步骤3**：得到牛顿方向和迭代公式

定义**牛顿方向**：
$$d_k = x - x_k = -H_k^{-1} g_k$$

因此，**牛顿法迭代公式**为：
$$\boxed{x_{k+1} = x_k + d_k = x_k - H_k^{-1} g_k}$$

### 1.4 牛顿方向的几何意义

**牛顿方向** $$d_k = -H_k^{-1} g_k$$ 具有重要的几何意义：

1. **与梯度的关系**：
   - 当 $$H_k = I$$（单位矩阵）时，$$d_k = -g_k$$，退化为最速下降方向
   - 当 $$H_k \neq I$$ 时，牛顿方向考虑了函数的曲率信息

2. **椭圆等高线的情况**：
   - 对于二次函数，等高线是椭圆
   - 最速下降法沿椭圆的法线方向（可能不是最优方向）
   - 牛顿法直接指向椭圆的中心（最优点）

3. **曲率修正**：
   - $$H_k^{-1}$$ 起到"曲率修正"的作用
   - 在曲率大的方向上步长较小，在曲率小的方向上步长较大

### 1.5 牛顿法的矩阵形式推导

**替代推导方式**：直接从最优性条件出发

对于无约束优化问题，最优解 $$x^*$$ 满足：
$$\nabla f(x^*) = 0$$

在点 $$x_k$$ 处对 $$\nabla f(x)$$ 进行一阶Taylor展开：
$$\nabla f(x) \approx \nabla f(x_k) + \nabla^2 f(x_k)(x - x_k)$$

令 $$\nabla f(x) = 0$$：
$$\nabla f(x_k) + \nabla^2 f(x_k)(x - x_k) = 0$$

解得：
$$x = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

这就是牛顿法的迭代公式。

### 1.6 牛顿法的实际计算步骤

**重要提醒**：在实际实现中，我们**不直接计算** $$H_k^{-1}$$，而是**求解线性方程组**：

$$H_k d_k = -g_k$$

**原因**：
1. **计算效率**：直接求逆矩阵需要 $$O(n^3)$$ 操作，而求解线性方程组可以利用矩阵的特殊结构
2. **数值稳定性**：求解线性方程组比直接求逆更稳定
3. **存储需求**：不需要显式存储逆矩阵

**常用求解方法**：
1. **Cholesky分解**（当 $$H_k$$ 正定时）：$$H_k = LL^T$$
2. **LU分解**：$$H_k = LU$$
3. **QR分解**：$$H_k = QR$$

### 1.7 牛顿法与其他方法的对比


| 特性 | 最速下降法 | 牛顿法 | 拟牛顿法 |
|------|------------|--------|----------|
| **使用信息** | 一阶（梯度） | 二阶（梯度+Hessian） | 一阶（近似二阶） |
| **迭代公式** | $$x_{k+1} = x_k - \alpha_k g_k$$ | $$x_{k+1} = x_k - H_k^{-1} g_k$$ | $$x_{k+1} = x_k - B_k^{-1} g_k$$ |
| **收敛速度** | 线性 | 二次 | 超线性 |
| **每步计算量** | $$O(n)$$ | $$O(n^3)$$ | $$O(n^2)$$ |
| **存储需求** | $$O(n)$$ | $$O(n^2)$$ | $$O(n^2)$$ |
| **全局收敛性** | 好 | 差 | 好 |

### 1.8 算法框架

**算法1：牛顿法**

**输入**：初始点 $$x_0$$，容差 $$\epsilon > 0$$
**输出**：最优解 $$x^*$$

1. **初始化**：$$k = 0$$
2. **while** $$\|\nabla f(x_k)\| > \epsilon$$ **do**
3. 　　计算梯度 $$g_k = \nabla f(x_k)$$
4. 　　计算Hessian矩阵 $$H_k = \nabla^2 f(x_k)$$
5. 　　求解线性方程组：$$H_k d_k = -g_k$$
6. 　　更新：$$x_{k+1} = x_k + d_k$$
7. 　　$$k = k + 1$$
8. **end while**

### 1.9 牛顿法的数学原理深入分析

#### 1.9.1 二次模型的精确性

牛顿法的核心在于**二次近似模型**的质量。对于二次函数：
$$f(x) = \frac{1}{2}x^T A x + b^T x + c$$

其中 $$A$$ 为正定矩阵，梯度和Hessian矩阵为：
- $$\nabla f(x) = Ax + b$$
- $$\nabla^2 f(x) = A$$（常数矩阵）

在这种情况下，Taylor展开是**精确的**，牛顿法**一步收敛**到最优解：
$$x^* = -A^{-1}b$$

#### 1.9.2 牛顿方向的下降性质

**定理**：如果 $$H_k$$ 正定，则牛顿方向 $$d_k = -H_k^{-1} g_k$$ 是下降方向。

**证明**：
$$g_k^T d_k = g_k^T (-H_k^{-1} g_k) = -g_k^T H_k^{-1} g_k$$

由于 $$H_k$$ 正定，$$H_k^{-1}$$ 也正定，因此：
$$g_k^T H_k^{-1} g_k > 0 \quad \text{（当 } g_k \neq 0 \text{ 时）}$$

所以 $$g_k^T d_k < 0$$，$$d_k$$ 是下降方向。

#### 1.9.3 牛顿法的不变性质

牛顿法具有**仿射不变性**：如果对变量进行线性变换 $$y = Tx$$（$$T$$ 可逆），牛顿法在新坐标系下的迭代路径与原坐标系等价。

这意味着牛顿法不受坐标系选择的影响，这是其相对于最速下降法的一个重要优势。

#### 1.9.4 条件数的影响

对于二次函数 $$f(x) = \frac{1}{2}x^T A x + b^T x + c$$：
- **最速下降法**的收敛率受条件数 $$\kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}$$ 影响
- **牛顿法**不受条件数影响，总是一步收敛

这解释了为什么牛顿法在处理病态问题时表现更好。

### 1.10 收敛性分析

**定理1（牛顿法局部收敛性）**：
设 $$f \in C^2$$，$$x^*$$ 为 $$f$$ 的极小点，且 $$\nabla^2 f(x^*)$$ 正定。若初始点 $$x_0$$ 充分接近 $$x^*$$，则牛顿法二次收敛：
$$\|x_{k+1} - x^*\| \leq M \|x_k - x^*\|^2$$

其中 $$M$$ 为正常数。

**收敛率比较**：
- **最速下降法**：线性收敛 $$O(\rho^k)$$，$$0 < \rho < 1$$
- **共轭梯度法**：超线性收敛（二次函数有限步收敛）
- **牛顿法**：二次收敛 $$O(\|x_k - x^*\|^2)$$

---

## 二、牛顿法的实际问题

### 2.1 主要困难

1. **Hessian矩阵计算昂贵**：每次迭代需要计算 $$n^2$$ 个二阶偏导数
2. **线性方程组求解复杂度高**：$$O(n^3)$$ 的计算量
3. **Hessian可能不正定**：导致搜索方向不是下降方向
4. **全局收敛性差**：对初始点要求严格

### 2.2 修正策略

#### 2.2.1 阻尼牛顿法

引入步长因子 $$\alpha_k$$：
$$x_{k+1} = x_k + \alpha_k d_k$$

其中 $$\alpha_k$$ 通过线搜索确定，常用Armijo准则：
$$f(x_k + \alpha_k d_k) \leq f(x_k) + c_1 \alpha_k g_k^T d_k$$

#### 2.2.2 修正Hessian矩阵

当 $$H_k$$ 不正定时，使用修正矩阵：
$$\tilde{H}_k = H_k + \mu_k I$$

其中 $$\mu_k \geq 0$$ 使得 $$\tilde{H}_k$$ 正定。

---

## 三、拟牛顿法的理论基础与核心原理

### 3.1 拟牛顿法的动机与基本思想

拟牛顿法是为了解决牛顿法的实际困难而提出的，其核心思想是：

**主要动机**：
1. **避免计算Hessian矩阵**：计算二阶偏导数在高维问题中代价昂贵（$$O(n^2)$$ 个元素）
2. **避免求解线性方程组**：每次迭代需要 $$O(n^3)$$ 的计算复杂度
3. **改善全局收敛性**：牛顿法对初始点要求严格，容易发散

**基本策略**：
- **用近似矩阵** $$B_k$$ **代替真实的Hessian矩阵** $$H_k$$
- **仅使用梯度信息**来逐步改进 $$B_k$$
- **保持超线性收敛**：在理想条件下达到与牛顿法相近的收敛速度

### 3.2 割线条件的推导与意义

#### 3.2.1 割线条件的来源

考虑函数 $$f(x)$$ 在两个相邻迭代点 $$x_k$$ 和 $$x_{k+1}$$ 处的梯度：

设 $$s_k = x_{k+1} - x_k$$（步长向量），$$y_k = g_{k+1} - g_k$$（梯度变化）

根据**微分中值定理**，存在 $$\xi_k \in [x_k, x_{k+1}]$$ 使得：
$$y_k = g_{k+1} - g_k = \nabla f(x_{k+1}) - \nabla f(x_k) = \nabla^2 f(\xi_k) s_k$$

这启发我们要求近似Hessian矩阵 $$B_{k+1}$$ 满足**割线条件**（Secant Condition）：
$$\boxed{B_{k+1} s_k = y_k}$$

#### 3.2.2 割线条件的几何意义

- **一维情况**：割线条件对应于用割线斜率近似导数
- **多维情况**：割线条件要求近似Hessian能够"记住"最近一步的曲率信息
- **物理意义**：$$B_{k+1}$$ 应该能够正确反映函数在方向 $$s_k$$ 上的曲率变化

#### 3.2.3 割线条件的限制

割线条件只给出了 $$n$$ 个约束（$$B_{k+1} s_k = y_k$$），但 $$B_{k+1}$$ 有 $$\frac{n(n+1)}{2}$$ 个独立元素（对称矩阵）。

当 $$n > 1$$ 时，约束不足，需要额外的条件来唯一确定 $$B_{k+1}$$。

### 3.3 拟牛顿条件的完整体系

一个理想的拟牛顿近似 $$B_{k+1}$$ 应该满足：

1. **割线条件**：$$B_{k+1} s_k = y_k$$
   - 保证对最近的梯度变化信息的正确反映

2. **对称性**：$$B_{k+1} = B_{k+1}^T$$
   - 保持与真实Hessian矩阵相同的对称性质

3. **正定性**：$$B_{k+1} \succ 0$$
   - 确保搜索方向是下降方向

4. **最小变化原则**：$$B_{k+1}$$ 与 $$B_k$$ 尽可能接近
   - 保持已有信息，只在必要时修正

### 3.4 拟牛顿方程的数学表述

#### 3.4.1 基本拟牛顿方程

对于近似Hessian矩阵 $$B_k$$，拟牛顿法的迭代公式为：
$$x_{k+1} = x_k - \alpha_k B_k^{-1} g_k$$

其中 $$\alpha_k$$ 是步长参数（通常通过线搜索确定）。

#### 3.4.2 逆Hessian近似

实际实现中，我们通常直接维护 $$H_k = B_k^{-1}$$（逆Hessian近似），迭代公式变为：
$$x_{k+1} = x_k - \alpha_k H_k g_k$$

相应的割线条件变为：
$$H_{k+1} y_k = s_k$$

### 3.5 拟牛顿法的理论挑战

#### 3.5.1 曲率条件

为保证算法的良好性质，需要满足**曲率条件**：
$$y_k^T s_k > 0$$

**物理意义**：这确保了函数在步长方向上是"上凸的"，即具有正曲率。

**数学意义**：这是保证 $$B_{k+1}$$ 正定的必要条件。

#### 3.5.2 更新公式的唯一性

给定割线条件和其他约束，如何唯一确定 $$B_{k+1}$$？

**解决方案**：使用**最小化范数变化**的原则：
$$\min_{B} \|B - B_k\|_F \quad \text{s.t.} \quad B s_k = y_k, \quad B = B^T$$

其中 $$\|\cdot\|_F$$ 是Frobenius范数。

### 3.6 拟牛顿法的分类

根据不同的更新策略，拟牛顿法主要分为：

1. **DFP（Davidon-Fletcher-Powell）**：第一个成功的拟牛顿算法
2. **BFGS（Broyden-Fletcher-Goldfarb-Shanno）**：目前最成功的拟牛顿算法
3. **L-BFGS（Limited-memory BFGS）**：大规模问题的首选
4. **SR1（Symmetric Rank-1）**：特殊的对称秩1更新

每种方法对应不同的 $$B_{k+1}$$ 更新公式，但都满足基本的拟牛顿条件。

---

## 四、BFGS算法的深入分析

### 4.1 BFGS算法的推导

**Broyden-Fletcher-Goldfarb-Shanno (BFGS)** 算法是目前最成功的拟牛顿算法。其推导基于**秩2更新**的思想。

#### 4.1.1 BFGS更新公式的推导

**问题设定**：给定当前近似 $$B_k$$，寻找 $$B_{k+1}$$ 使得：
1. 满足割线条件：$$B_{k+1} s_k = y_k$$
2. 对称性：$$B_{k+1} = B_{k+1}^T$$
3. 最小化 $$\|B_{k+1} - B_k\|_F$$（Frobenius范数）

**解决方案**：使用**秩2更新**的形式：
$$B_{k+1} = B_k + \alpha u u^T + \beta v v^T$$

其中 $$u, v$$ 是待定向量，$$\alpha, \beta$$ 是待定标量。

**推导过程**：
1. 令 $$u = y_k$$，$$v = B_k s_k$$
2. 利用割线条件 $$B_{k+1} s_k = y_k$$
3. 通过对称性和最小范数变化原则确定 $$\alpha, \beta$$

**最终得到BFGS更新公式**：
$$\boxed{B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}}$$

#### 4.1.2 逆Hessian的BFGS更新

实际应用中，我们直接更新 $$H_k = B_k^{-1}$$。使用**Sherman-Morrison-Woodbury公式**可以得到：

$$\boxed{H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}}$$

这个公式可以分解为三个部分：
1. **第一项**：$$\left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right)$$ - 对旧信息的修正
2. **第二项**：$$\frac{s_k s_k^T}{y_k^T s_k}$$ - 新信息的添加

### 4.2 BFGS公式的几何解释

#### 4.2.1 秩2更新的含义

BFGS更新是**秩2更新**，即：
$$B_{k+1} - B_k = \text{rank-2 matrix}$$

这意味着我们只修改 $$B_k$$ 在两个特定方向上的"行为"：
- **减去**：$$\frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$ - 移除旧的曲率信息
- **加上**：$$\frac{y_k y_k^T}{y_k^T s_k}$$ - 添加新的曲率信息

#### 4.2.2 投影解释

BFGS更新可以理解为：
1. **投影移除**：将 $$B_k s_k$$ 方向的信息移除
2. **投影添加**：将 $$y_k$$ 方向的信息添加

这确保了 $$B_{k+1} s_k = y_k$$，同时保持其他方向的信息不变。

### 4.3 BFGS算法的重要性质

#### 4.3.1 正定性保持

**定理**：如果 $$B_k \succ 0$$ 且 $$y_k^T s_k > 0$$，则 $$B_{k+1} \succ 0$$。

**证明思路**：
- BFGS更新保持正定性
- 曲率条件 $$y_k^T s_k > 0$$ 是关键

#### 4.3.2 自修正性质

BFGS算法具有**自修正性质**：即使 $$B_0$$ 不是好的近似，经过足够多次迭代后，$$B_k$$ 会自动收敛到真实的Hessian矩阵。

#### 4.3.3 有限步超线性收敛

**定理**：对于二次函数，BFGS算法在至多 $$n$$ 步内生成 $$A$$-共轭方向组，从而有限步收敛。

### 4.4 BFGS更新公式的数值实现

#### 4.4.1 直接实现

```python
def bfgs_update_H(H_k, s_k, y_k):
    """BFGS更新逆Hessian矩阵"""
    rho = 1.0 / (y_k.T @ s_k)
    
    # 计算 I - rho * s_k * y_k^T
    I_minus_rho_sk_yk = np.eye(len(s_k)) - rho * np.outer(s_k, y_k)
    
    # 计算 I - rho * y_k * s_k^T  
    I_minus_rho_yk_sk = np.eye(len(s_k)) - rho * np.outer(y_k, s_k)
    
    # BFGS更新
    H_new = I_minus_rho_sk_yk @ H_k @ I_minus_rho_yk_sk + rho * np.outer(s_k, s_k)
    
    return H_new
```

#### 4.4.2 数值稳定性考虑

1. **跳过更新条件**：当 $$y_k^T s_k \leq \delta$$ 时跳过更新（$$\delta$$ 是小正数）
2. **缩放策略**：使用适当的初始缩放 $$H_0 = \gamma I$$
3. **重启策略**：定期重置 $$H_k = I$$

### 4.2 BFGS算法框架

**算法2：BFGS算法**

**输入**：初始点 $$x_0$$，初始矩阵 $$H_0 = I$$
**输出**：最优解 $$x^*$$

1. **初始化**：$$k = 0$$，计算 $$g_0 = \nabla f(x_0)$$
2. **while** $$\|g_k\| > \epsilon$$ **do**
3. 　　计算搜索方向：$$d_k = -H_k g_k$$
4. 　　线搜索确定步长 $$\alpha_k$$
5. 　　更新：$$x_{k+1} = x_k + \alpha_k d_k$$
6. 　　计算：$$g_{k+1} = \nabla f(x_{k+1})$$
7. 　　设置：$$s_k = x_{k+1} - x_k$$，$$y_k = g_{k+1} - g_k$$
8. 　　**if** $$y_k^T s_k > 0$$ **then**
9. 　　　　更新 $$H_{k+1}$$ 使用BFGS公式
10. 　　**else**
11. 　　　　$$H_{k+1} = H_k$$（跳过更新）
12. 　　**end if**
13. 　　$$k = k + 1$$
14. **end while**

### 4.3 BFGS的理论性质

**定理2（BFGS收敛性）**：
设 $$f$$ 二次连续可微，$$\nabla^2 f$$ Lipschitz连续，$$x^*$$ 为严格局部极小点。若线搜索满足Wolfe条件，则BFGS算法超线性收敛。

**Wolfe条件**：
1. **充分下降条件**：$$f(x_k + \alpha_k d_k) \leq f(x_k) + c_1 \alpha_k g_k^T d_k$$
2. **曲率条件**：$$g(x_k + \alpha_k d_k)^T d_k \geq c_2 g_k^T d_k$$

其中 $$0 < c_1 < c_2 < 1$$，典型取值 $$c_1 = 10^{-4}$$，$$c_2 = 0.9$$。

---

## 五、DFP算法

### 5.1 DFP更新公式

**Davidon-Fletcher-Powell (DFP)** 算法是第一个拟牛顿算法：

$$H_{k+1} = H_k - \frac{H_k y_k y_k^T H_k}{y_k^T H_k y_k} + \frac{s_k s_k^T}{y_k^T s_k}$$

### 5.2 DFP与BFGS的关系

DFP和BFGS是**对偶**关系：
- DFP更新 $$H_k$$ 的公式等价于BFGS更新 $$B_k$$ 的公式
- BFGS更新 $$H_k$$ 的公式等价于DFP更新 $$B_k$$ 的公式

实践中，**BFGS通常比DFP更稳定**。

---

## 六、L-BFGS算法的理论与实现

### 6.1 大规模问题的挑战与动机

#### 6.1.1 BFGS算法的局限性

当问题规模 $$n$$ 很大时，标准BFGS算法面临严重挑战：

1. **存储需求**：$$n \times n$$ 的矩阵 $$H_k$$ 需要 $$O(n^2)$$ 存储空间
   - 当 $$n = 10^6$$ 时，需要约8TB内存（双精度浮点）
   
2. **计算复杂度**：每次更新 $$H_k$$ 需要 $$O(n^2)$$ 操作

3. **实际应用限制**：在机器学习、图像处理等领域，$$n$$ 经常达到百万甚至更大

#### 6.1.2 L-BFGS的核心思想

**Limited-memory BFGS (L-BFGS)** 的关键洞察：
- **不显式存储** $$H_k$$ 矩阵
- **隐式表示** $$H_k$$ 通过有限的历史信息
- **仅计算** $$H_k g_k$$（我们真正需要的量）

### 6.2 L-BFGS的数学基础

#### 6.2.1 BFGS递推关系

从BFGS更新公式出发，我们可以建立递推关系。设：
- $$\{s_i, y_i\}_{i=k-m}^{k-1}$$：最近 $$m$$ 个向量对
- $$H_k^0$$：初始近似（通常取 $$\gamma I$$）

则 $$H_k$$ 可以表示为：
$$H_k = V_{k-1}^T V_{k-2}^T \cdots V_{k-m}^T H_k^0 V_{k-m} \cdots V_{k-2} V_{k-1} + \sum_{i=k-m}^{k-1} \rho_i V_{k-1}^T \cdots V_{i+1}^T s_i s_i^T V_{i+1} \cdots V_{k-1}$$

其中：
$$V_i = I - \rho_i y_i s_i^T, \quad \rho_i = \frac{1}{y_i^T s_i}$$

#### 6.2.2 两循环递推的数学推导

**目标**：计算 $$d_k = -H_k g_k$$ 而不显式构造 $$H_k$$

**关键观察**：$$H_k$$ 的结构允许我们通过两个循环高效计算 $$H_k g_k$$：

1. **第一循环（向后）**：逐步"剥离"BFGS更新的影响
2. **第二循环（向前）**：重新"应用"这些更新

### 6.3 L-BFGS两循环递推算法详解

#### 6.3.1 算法的数学表述

**输入**：梯度 $$g_k$$，历史信息 $$\{s_i, y_i\}_{i=k-m}^{k-1}$$，初始矩阵 $$H_k^0$$
**输出**：搜索方向 $$d_k = -H_k g_k$$

**第一循环（向后递推）**：
```
q = g_k
for i = k-1, k-2, ..., k-m do
    ρ_i = 1 / (y_i^T s_i)
    α_i = ρ_i s_i^T q
    q = q - α_i y_i
end for
```

**中间步骤**：
```
r = H_k^0 q
```

**第二循环（向前递推）**：
```
for i = k-m, k-m+1, ..., k-1 do
    β = ρ_i y_i^T r
    r = r + s_i (α_i - β)
end for
return d_k = -r
```

#### 6.3.2 算法的几何直观

**第一循环的作用**：
- 逐步"撤销"每个BFGS更新对梯度的影响
- 得到"原始"梯度在初始度量下的表示

**第二循环的作用**：
- 重新"应用"这些更新，但现在是在正确的顺序下
- 得到最终的搜索方向

### 6.4 L-BFGS的实现细节

#### 6.4.1 存储管理

L-BFGS只需存储：
- $$m$$ 个 $$s_i$$ 向量（每个长度 $$n$$）
- $$m$$ 个 $$y_i$$ 向量（每个长度 $$n$$）
- $$m$$ 个标量 $$\rho_i$$

**总存储需求**：$$O(mn)$$，其中 $$m$$ 通常为5-20

#### 6.4.2 初始Hessian近似的选择

常用的 $$H_k^0$$ 选择：
1. **单位矩阵**：$$H_k^0 = I$$
2. **缩放单位矩阵**：$$H_k^0 = \gamma_k I$$，其中
   $$\gamma_k = \frac{y_{k-1}^T s_{k-1}}{y_{k-1}^T y_{k-1}}$$

第二种选择通常效果更好，因为它考虑了最近的曲率信息。

### 6.5 L-BFGS的理论性质

#### 6.5.1 收敛性

**定理**：在适当条件下，L-BFGS算法具有与BFGS相同的超线性收敛性质。

**关键条件**：
- 目标函数满足适当的光滑性条件
- 线搜索满足Wolfe条件
- 存储参数 $$m$$ 足够大

#### 6.5.2 计算复杂度

每次迭代的计算复杂度：
- **两循环递推**：$$O(mn)$$
- **线搜索**：$$O(n)$$ 每次函数/梯度计算
- **总计算量**：$$O(mn + \text{线搜索开销})$$

相比BFGS的 $$O(n^2)$$，在大规模问题中有显著优势。

### 6.3 两循环递推算法

**算法3：L-BFGS两循环递推**

**输入**：$$g_k$$，$$\{s_i, y_i\}_{i=k-m}^{k-1}$$，初始矩阵 $$H_k^0$$
**输出**：$$d_k = -H_k g_k$$

**第一循环**（向后）：
1. $$q = g_k$$
2. **for** $$i = k-1, k-2, \ldots, k-m$$ **do**
3. 　　$$\rho_i = \frac{1}{y_i^T s_i}$$
4. 　　$$\alpha_i = \rho_i s_i^T q$$
5. 　　$$q = q - \alpha_i y_i$$
6. **end for**

**中间步骤**：
7. $$r = H_k^0 q$$

**第二循环**（向前）：
8. **for** $$i = k-m, k-m+1, \ldots, k-1$$ **do**
9. 　　$$\beta = \rho_i y_i^T r$$
10. 　　$$r = r + s_i (\alpha_i - \beta)$$
11. **end for**
12. **return** $$d_k = -r$$


---

## 七、详细例题解析

### 例题1：二次函数的牛顿法

**问题**：使用牛顿法求解
$$\min f(x) = \frac{1}{2}x^T \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} x - \begin{pmatrix} 1 \\ 2 \end{pmatrix}^T x$$

初始点 $$x_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$。

**解析**：

**步骤1**：计算梯度和Hessian矩阵
$$\nabla f(x) = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} x - \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$\nabla^2 f(x) = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$$

**步骤2**：第0次迭代
$$g_0 = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 0 \\ 0 \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} -1 \\ -2 \end{pmatrix}$$

$$H_0 = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}$$

求解 $$H_0 d_0 = -g_0$$：
$$\begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} d_0 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

$$d_0 = H_0^{-1} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \frac{1}{7}\begin{pmatrix} 0 \\ 7 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

$$x_1 = x_0 + d_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**步骤3**：第1次迭代
$$g_1 = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

由于 $$g_1 = 0$$，算法收敛到最优解 $$x^* = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$。

**验证**：理论最优解为 $$x^* = A^{-1}b = \frac{1}{7}\begin{pmatrix} 2 & -1 \\ -1 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**结论**：对于二次函数，牛顿法一步收敛到精确解。

### 例题2：非线性函数的BFGS算法

**问题**：使用BFGS算法求解
$$\min f(x_1, x_2) = (x_1 - 2)^4 + (x_1 - 2x_2)^2$$

初始点 $$x_0 = \begin{pmatrix} 0 \\ 3 \end{pmatrix}$$。

**解析**：

**步骤1**：计算梯度
$$\frac{\partial f}{\partial x_1} = 4(x_1 - 2)^3 + 2(x_1 - 2x_2)$$
$$\frac{\partial f}{\partial x_2} = -4(x_1 - 2x_2)$$

**步骤2**：第0次迭代
$$g_0 = \begin{pmatrix} 4(-2)^3 + 2(-6) \\ -4(-6) \end{pmatrix} = \begin{pmatrix} -44 \\ 24 \end{pmatrix}$$

$$H_0 = I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

$$d_0 = -H_0 g_0 = -\begin{pmatrix} -44 \\ 24 \end{pmatrix} = \begin{pmatrix} 44 \\ -24 \end{pmatrix}$$

使用Armijo线搜索，假设得到 $$\alpha_0 = 0.01$$：
$$x_1 = x_0 + \alpha_0 d_0 = \begin{pmatrix} 0 \\ 3 \end{pmatrix} + 0.01 \begin{pmatrix} 44 \\ -24 \end{pmatrix} = \begin{pmatrix} 0.44 \\ 2.76 \end{pmatrix}$$

**步骤3**：更新BFGS矩阵
$$s_0 = x_1 - x_0 = \begin{pmatrix} 0.44 \\ -0.24 \end{pmatrix}$$

$$g_1 = \begin{pmatrix} 4(0.44-2)^3 + 2(0.44-2 \times 2.76) \\ -4(0.44-2 \times 2.76) \end{pmatrix} = \begin{pmatrix} -25.02 \\ 20.16 \end{pmatrix}$$

$$y_0 = g_1 - g_0 = \begin{pmatrix} -25.02 + 44 \\ 20.16 - 24 \end{pmatrix} = \begin{pmatrix} 18.98 \\ -3.84 \end{pmatrix}$$

检查 $$y_0^T s_0 = 18.98 \times 0.44 + (-3.84) \times (-0.24) = 9.27 > 0$$，满足更新条件。

使用BFGS公式更新 $$H_1$$：
$$H_1 = \left(I - \frac{s_0 y_0^T}{y_0^T s_0}\right) H_0 \left(I - \frac{y_0 s_0^T}{y_0^T s_0}\right) + \frac{s_0 s_0^T}{y_0^T s_0}$$

继续迭代直到收敛...

### 例题3：L-BFGS算法实例

**问题**：使用L-BFGS算法求解Rosenbrock函数
$$f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$$

对于 $$n = 4$$的情况。

**解析**：

**步骤1**：计算梯度
对于Rosenbrock函数，梯度为：
$$\frac{\partial f}{\partial x_1} = -400x_1(x_2 - x_1^2) - 2(1 - x_1)$$
$$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400x_i(x_{i+1} - x_i^2) - 2(1 - x_i), \quad i = 2, \ldots, n-1$$
$$\frac{\partial f}{\partial x_n} = 200(x_n - x_{n-1}^2)$$

**步骤2**：L-BFGS迭代
由于存储限制，我们只保存最近 $$m = 5$$ 个 $$\{s_i, y_i\}$$ 对，使用两循环递推算法计算搜索方向。

初始点取 $$x_0 = (-1.2, 1, -1.2, 1)^T$$，经过多次迭代后收敛到全局最优解 $$x^* = (1, 1, 1, 1)^T$$。

---

## 八、算法比较与选择

### 8.1 性能比较


| 算法 | 收敛速度 | 每步计算量 | 存储需求 | 全局收敛性 | 适用问题规模 |
|------|----------|------------|----------|------------|--------------|
| 牛顿法 | 二次收敛 | $$O(n^3)$$ | $$O(n^2)$$ | 差 | 小规模 |
| BFGS | 超线性收敛 | $$O(n^2)$$ | $$O(n^2)$$ | 好 | 中等规模 |
| L-BFGS | 超线性收敛 | $$O(mn)$$ | $$O(mn)$$ | 好 | 大规模 |
| DFP | 超线性收敛 | $$O(n^2)$$ | $$O(n^2)$$ | 一般 | 中等规模 |


### 8.2 算法选择指南

1. **小规模问题**（$$n < 100$$）：
   - 如果能容易计算Hessian矩阵，使用牛顿法
   - 否则使用BFGS

2. **中等规模问题**（$$100 \leq n \leq 1000$$）：
   - 首选BFGS
   - 考虑使用修正的牛顿法

3. **大规模问题**（$$n > 1000$$）：
   - 首选L-BFGS
   - 考虑使用截断牛顿法

4. **特殊结构问题**：
   - 稀疏问题：使用稀疏牛顿法
   - 非光滑问题：使用子梯度方法或束方法

### 8.3 实现技巧

#### 8.3.1 线搜索策略

**Wolfe条件**是拟牛顿法中最常用的线搜索准则：
1. $$f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f(x_k)^T d_k$$
2. $$\nabla f(x_k + \alpha d_k)^T d_k \geq c_2 \nabla f(x_k)^T d_k$$

**推荐参数**：$$c_1 = 10^{-4}$$，$$c_2 = 0.9$$（BFGS），$$c_2 = 0.1$$（共轭梯度法）

#### 8.3.2 数值稳定性

1. **跳过更新**：当 $$y_k^T s_k \leq 0$$ 时跳过BFGS更新
2. **重启策略**：定期将 $$H_k$$ 重置为单位矩阵
3. **缩放技巧**：使用适当的初始缩放 $$H_0 = \gamma I$$

---

## 九、高级主题

### 9.1 约束优化中的拟牛顿法

对于等式约束问题：
$$\min f(x) \quad \text{s.t.} \quad h(x) = 0$$

可以使用**序列二次规划（SQP）**方法，其中拟牛顿法用于近似拉格朗日函数的Hessian矩阵。

### 9.2 非凸优化中的应用

在机器学习中，拟牛顿法广泛应用于：
- **逻辑回归**：L-BFGS是标准求解器
- **神经网络训练**：L-BFGS用于小批量优化
- **支持向量机**：SMO算法的变种

### 9.3 并行化实现

L-BFGS的并行化主要体现在：
1. **梯度计算并行化**：分布式计算 $$\nabla f(x)$$
2. **线搜索并行化**：同时尝试多个步长
3. **向量操作并行化**：利用BLAS库加速

---

## 十、数值实验

### 10.1 测试函数

我们使用经典的测试函数比较各算法性能：

1. **Rosenbrock函数**：$$f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$$
2. **Powell函数**：$$f(x) = \sum_{i=1}^{n/4} [(x_{4i-3} + 10x_{4i-2})^2 + 5(x_{4i-1} - x_{4i})^2 + (x_{4i-2} - 2x_{4i-1})^4 + 10(x_{4i-3} - x_{4i})^4]$$
3. **Beale函数**：$$f(x_1, x_2) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 + (2.625 - x_1 + x_1 x_2^3)^2$$

### 10.2 实验结果


| 测试函数 | 维数 | 牛顿法 | BFGS | L-BFGS | 最速下降法 |
|----------|------|--------|------|--------|------------|
| Rosenbrock | 100 | - | 52步 | 48步 | >1000步 |
| Powell | 100 | - | 31步 | 35步 | >1000步 |
| Beale | 2 | 6步 | 12步 | 13步 | 156步 |


**说明**："-"表示Hessian矩阵病态或不正定，牛顿法失效。

---

## 总结

本文系统地介绍了牛顿法和拟牛顿法的理论基础与实际应用：

### 主要内容回顾

1. **牛顿法**：
   - 利用二阶信息实现快速收敛
   - 局部二次收敛，但全局收敛性差
   - 计算代价高，适合小规模问题

2. **拟牛顿法**：
   - BFGS：最成功的拟牛顿算法，超线性收敛
   - L-BFGS：大规模问题的首选，存储效率高
   - DFP：历史意义重大，但稳定性不如BFGS

3. **实际应用**：
   - 算法选择依赖于问题规模和结构
   - 线搜索和数值稳定性是实现关键
   - 在机器学习中有广泛应用

### 实践建议

1. **对于初学者**：先掌握BFGS的基本原理和实现
2. **对于实践者**：根据问题规模选择合适算法
3. **对于研究者**：关注约束优化和非凸优化中的新发展

拟牛顿法在现代优化中仍然是核心方法，理解其原理对于解决实际优化问题具有重要意义。

---


## 相关例题

如果您想查看本文所有的相关例题，请点击下方链接：

<div style="text-align: center; margin: 30px 0;">
  <a href="/2025/09/14/newton-quasi-newton-methods-examples/" class="btn btn-primary btn-lg" style="display: inline-block; padding: 15px 30px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; border: none;">
    <i class="fa fa-code" style="margin-right: 10px;"></i>查看相关例题
  </a>
</div>

> **相关文章**:
> - [《最优化方法——最速下降法与共轭梯度法》](/2025/09/11/steepest-descent-and-conjugate-gradient/)
> - [《最速下降法与共轭梯度法例题详解》](/2025/09/12/steepest-descent-conjugate-gradient-examples/)
> - [《最优化理论基础》](/2025/09/08/optimization-fundamentals/)
> - [《KKT条件详解》](/2025/09/10/kkt-conditions/)

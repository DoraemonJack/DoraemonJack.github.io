---
layout:       post
title:        "最优化理论——基础_例题集"
subtitle:     "凸函数判断、凸规划问题与二次型函数"
date:         2025-09-09 10:00:00
author:       "DoraemonJack"
header-style: text
header-img: "img/post-bg-math.jpg"
catalog:      true
mathjax:      true
hidden:       true
mermaid:      true
tags:
    - Optimization
    - 最优化理论
    - Convex Analysis
    - Examples
    - Mathematics
---

本文收集了最优化理论基础中的经典例题，涵盖凸函数判断、凸规划问题识别以及二次型函数的性质分析。通过详细的解题过程，帮助读者深入理解最优化理论的核心概念。

## 一、凸函数判断例题

### 例1：判断二元函数的凸性

**题目**：判断 $f(x_1, x_2) = x_1 x_2$ 在集合 $S = \{x \mid x_1 \geq 0, x_2 \geq 0\}$ 上是否为凸函数。

**解**：

设 $x^{(1)} = (1,2)$，$x^{(2)} = (2,1) \in S$，取 $\lambda = \frac{1}{2}$。

**步骤1**：计算 $f(\lambda x^{(1)} + (1-\lambda) x^{(2)})$

$$f(\lambda x^{(1)} + (1-\lambda) x^{(2)}) = f\left(\frac{1}{2}(1,2) + \frac{1}{2}(2,1)\right) = f\left(\frac{3}{2}, \frac{3}{2}\right) = \frac{3}{2} \cdot \frac{3}{2} = \frac{9}{4}$$

**步骤2**：计算 $\lambda f(x^{(1)}) + (1-\lambda) f(x^{(2)})$

$$\lambda f(x^{(1)}) + (1-\lambda) f(x^{(2)}) = \frac{1}{2} \cdot f(1,2) + \frac{1}{2} \cdot f(2,1) = \frac{1}{2} \cdot 2 + \frac{1}{2} \cdot 2 = 2$$

**步骤3**：比较结果

由于 $\frac{9}{4} = 2.25 > 2$，不满足凸函数定义：
$$f(\lambda x^{(1)} + (1-\lambda) x^{(2)}) \leq \lambda f(x^{(1)}) + (1-\lambda) f(x^{(2)})$$

**结论**：$f(x_1, x_2) = x_1 x_2$ 在 $S$ 上不是凸函数。

### 例2：二次型函数的凸性条件

**题目**：设 $f(x) = x^T Q x$，其中 $Q \in \mathbb{R}^{n \times n}$，$Q = Q^T$。证明 $f$ 为凸函数当且仅当 $(x-y)^T Q (x-y) \geq 0$。

**证明**：

**必要性**：设 $f$ 是凸函数，则对任意 $x, y \in \mathbb{R}^n$ 和 $\lambda \in [0,1]$，有：
$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$$

即：
$$\lambda f(x) + (1-\lambda) f(y) - f(\lambda x + (1-\lambda) y) \geq 0$$

将 $f(x) = x^T Q x$ 代入：
$$\lambda x^T Q x + (1-\lambda) y^T Q y - (\lambda x + (1-\lambda) y)^T Q (\lambda x + (1-\lambda) y) \geq 0$$

展开右边：
$$\lambda x^T Q x + (1-\lambda) y^T Q y - [\lambda^2 x^T Q x + \lambda(1-\lambda) x^T Q y + \lambda(1-\lambda) y^T Q x + (1-\lambda)^2 y^T Q y] \geq 0$$

整理得：
$$\lambda(1-\lambda)[x^T Q x + y^T Q y - x^T Q y - y^T Q x] \geq 0$$

由于 $\lambda(1-\lambda) \geq 0$，所以：
$$x^T Q x + y^T Q y - x^T Q y - y^T Q x \geq 0$$

即：
$$(x-y)^T Q (x-y) \geq 0$$

**充分性**：反之，如果 $(x-y)^T Q (x-y) \geq 0$ 对所有 $x, y$ 成立，则上述推导过程可逆，得到 $f$ 是凸函数。

**结论**：$f(x) = x^T Q x$ 为凸函数当且仅当 $Q$ 是半正定矩阵。

## 二、凸规划问题识别

### 例3：判断是否为凸规划问题

**题目**：判断以下问题是否为凸规划问题：

$$\begin{align}
\min \quad & f(x) = x_1^2 + x_2^2 - 4x_1 + 4 \\
\text{s.t.} \quad & g_1(x) = -x_1 + x_2 - 2 \leq 0 \\
& g_2(x) = x_1^2 - x_2 + 1 \leq 0 \\
& x_1, x_2 \geq 0
\end{align}$$

**解**：

**步骤1**：分析目标函数 $f(x)$ 的凸性

计算 $f(x)$ 的Hessian矩阵：
$$\frac{\partial f}{\partial x_1} = 2x_1 - 4, \quad \frac{\partial f}{\partial x_2} = 2x_2$$

$$\frac{\partial^2 f}{\partial x_1^2} = 2, \quad \frac{\partial^2 f}{\partial x_1 \partial x_2} = 0, \quad \frac{\partial^2 f}{\partial x_2^2} = 2$$

因此：
$$H_f = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

由于 $H_f$ 是正定矩阵（特征值均为正），所以 $f(x)$ 是凸函数。

**步骤2**：分析约束函数的凸性

对于 $g_1(x) = -x_1 + x_2 - 2$：
$$H_{g_1} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$

$H_{g_1}$ 是半正定矩阵，所以 $g_1(x)$ 是凸函数。

对于 $g_2(x) = x_1^2 - x_2 + 1$：
$$\frac{\partial^2 g_2}{\partial x_1^2} = 2, \quad \frac{\partial^2 g_2}{\partial x_1 \partial x_2} = 0, \quad \frac{\partial^2 g_2}{\partial x_2^2} = 0$$

$$H_{g_2} = \begin{bmatrix} 2 & 0 \\ 0 & 0 \end{bmatrix}$$

$H_{g_2}$ 是半正定矩阵，所以 $g_2(x)$ 是凸函数。

**步骤3**：分析非负约束

$x_1 \geq 0$ 和 $x_2 \geq 0$ 都是线性约束，因此是凸约束。

**结论**：该问题的目标函数和所有约束函数都是凸函数，因此这是一个凸规划问题。

## 三、二次型函数的梯度与Hessian矩阵

### 例4：计算二次型函数的梯度与Hessian矩阵

**题目**：求 $f(x) = \frac{1}{2} \|Ax - b\|^2$ 的梯度和Hessian矩阵。

**解**：

**步骤1**：展开目标函数

由于 $Ax - b$ 是向量，有：
$$\|Ax - b\|^2 = (Ax - b)^T (Ax - b)$$

因此：
$$f(x) = \frac{1}{2} (Ax - b)^T (Ax - b) = \frac{1}{2} (x^T A^T - b^T)(Ax - b)$$

展开得：
$$f(x) = \frac{1}{2} [x^T A^T A x - x^T A^T b - b^T A x + b^T b]$$

由于 $b^T A x = x^T A^T b$，所以：
$$f(x) = \frac{1}{2} x^T A^T A x - x^T A^T b + \frac{1}{2} b^T b$$

**步骤2**：计算梯度

对 $f(x)$ 关于 $x$ 求导：
$$\nabla f(x) = \frac{1}{2} \cdot 2 A^T A x - A^T b = A^T A x - A^T b = A^T (Ax - b)$$

**步骤3**：计算Hessian矩阵

对梯度再次求导：
$$\nabla^2 f(x) = A^T A$$

**步骤4**：验证凸性

由于 $A^T A$ 是半正定矩阵（对任意向量 $v$，有 $v^T A^T A v = \|Av\|^2 \geq 0$），所以 $f(x)$ 是凸函数。

**结论**：
- 梯度：$\nabla f(x) = A^T (Ax - b)$
- Hessian矩阵：$\nabla^2 f(x) = A^T A$

## 四、无约束优化问题例题

### 例5：求二元函数的驻点并判断性质

**题目**：考虑函数 $f(x_1, x_2) = 2x_1^3 + 3x_2^2 + 3x_1^2x_2 - 24x_2$，求其驻点，并判断是局部极小点、局部极大点还是鞍点。

**解**：

**步骤1：计算梯度（一阶偏导数）**

$$\frac{\partial f}{\partial x_1} = 6x_1^2 + 6x_1x_2$$
$$\frac{\partial f}{\partial x_2} = 6x_2 + 3x_1^2 - 24$$

因此梯度为：
$$\nabla f(x) = \begin{pmatrix} 6x_1^2 + 6x_1x_2 \\ 6x_2 + 3x_1^2 - 24 \end{pmatrix}$$

**步骤2：令梯度为零，求解驻点**

令 $\nabla f(x) = 0$，即：
$$\begin{cases} 6x_1^2 + 6x_1x_2 = 0 \quad (1) \\ 6x_2 + 3x_1^2 - 24 = 0 \quad (2) \end{cases}$$

从方程 (1) 得：$6x_1(x_1 + x_2) = 0$，即 $x_1 = 0$ 或 $x_1 + x_2 = 0$

**情况1**：$x_1 = 0$

将 $x_1 = 0$ 代入方程 (2)：
$$6x_2 + 3(0)^2 - 24 = 0$$
$$6x_2 = 24$$
$$x_2 = 4$$

因此驻点为：$x^{(1)} = (0, 4)$

**情况2**：$x_1 + x_2 = 0$，即 $x_2 = -x_1$

将 $x_2 = -x_1$ 代入方程 (2)：
$$6(-x_1) + 3x_1^2 - 24 = 0$$
$$-6x_1 + 3x_1^2 - 24 = 0$$
$$3x_1^2 - 6x_1 - 24 = 0$$
$$x_1^2 - 2x_1 - 8 = 0$$
$$(x_1 - 4)(x_1 + 2) = 0$$

因此 $x_1 = 4$ 或 $x_1 = -2$

- 当 $x_1 = 4$ 时，$x_2 = -4$，驻点为：$x^{(2)} = (4, -4)$
- 当 $x_1 = -2$ 时，$x_2 = 2$，驻点为：$x^{(3)} = (-2, 2)$

因此所有驻点为：
- $x^{(1)} = (0, 4)$
- $x^{(2)} = (4, -4)$
- $x^{(3)} = (-2, 2)$

**步骤3：计算Hessian矩阵（二阶偏导数矩阵）**

$$\frac{\partial^2 f}{\partial x_1^2} = 12x_1 + 6x_2$$
$$\frac{\partial^2 f}{\partial x_1 \partial x_2} = 6x_1$$
$$\frac{\partial^2 f}{\partial x_2^2} = 6$$

因此Hessian矩阵为：
$$\nabla^2 f(x) = \begin{pmatrix} 12x_1 + 6x_2 & 6x_1 \\ 6x_1 & 6 \end{pmatrix}$$

**步骤4：在驻点处评估Hessian矩阵的性质**

**在驻点 $x^{(1)} = (0, 4)$ 处**：
$$\nabla^2 f(x^{(1)}) = \begin{pmatrix} 12(0) + 6(4) & 6(0) \\ 6(0) & 6 \end{pmatrix} = \begin{pmatrix} 24 & 0 \\ 0 & 6 \end{pmatrix}$$

计算主子式：
- 一阶主子式：$D_1 = 24 > 0$
- 二阶主子式：$D_2 = \det \begin{pmatrix} 24 & 0 \\ 0 & 6 \end{pmatrix} = 24 \cdot 6 = 144 > 0$

由于所有主子式都大于0，Hessian矩阵为正定矩阵。

**在驻点 $x^{(2)} = (4, -4)$ 处**：
$$\nabla^2 f(x^{(2)}) = \begin{pmatrix} 12(4) + 6(-4) & 6(4) \\ 6(4) & 6 \end{pmatrix} = \begin{pmatrix} 48 - 24 & 24 \\ 24 & 6 \end{pmatrix} = \begin{pmatrix} 24 & 24 \\ 24 & 6 \end{pmatrix}$$

计算主子式：
- 一阶主子式：$D_1 = 24 > 0$
- 二阶主子式：$D_2 = \det \begin{pmatrix} 24 & 24 \\ 24 & 6 \end{pmatrix} = 24 \cdot 6 - 24 \cdot 24 = 144 - 576 = -432 < 0$

由于 $D_2 < 0$，Hessian矩阵为不定矩阵。

**在驻点 $x^{(3)} = (-2, 2)$ 处**：
$$\nabla^2 f(x^{(3)}) = \begin{pmatrix} 12(-2) + 6(2) & 6(-2) \\ 6(-2) & 6 \end{pmatrix} = \begin{pmatrix} -24 + 12 & -12 \\ -12 & 6 \end{pmatrix} = \begin{pmatrix} -12 & -12 \\ -12 & 6 \end{pmatrix}$$

计算主子式：
- 一阶主子式：$D_1 = -12 < 0$
- 二阶主子式：$D_2 = \det \begin{pmatrix} -12 & -12 \\ -12 & 6 \end{pmatrix} = (-12) \cdot 6 - (-12) \cdot (-12) = -72 - 144 = -216 < 0$

由于 $D_1 < 0$ 且 $D_2 < 0$，Hessian矩阵为不定矩阵。

**结论**：
- $x^{(1)} = (0, 4)$ 是局部极小点（Hessian矩阵正定）
- $x^{(2)} = (4, -4)$ 是鞍点（Hessian矩阵不定）
- $x^{(3)} = (-2, 2)$ 是鞍点（Hessian矩阵不定）

### 例6：判断函数的全局性质

**题目**：对于函数 $f(x_1, x_2) = 2x_1^3 + 3x_2^2 + 3x_1^2x_2 - 24x_2$，判断是否存在全局最优解。

**解**：

**分析函数的全局性质**：

考虑当 $x_1 = 0$ 时：
$$f(0, x_2) = 2(0)^3 + 3x_2^2 + 3(0)^2x_2 - 24x_2 = 3x_2^2 - 24x_2$$

这是一个关于 $x_2$ 的二次函数，当 $x_2 \to \pm\infty$ 时，$f(0, x_2) \to +\infty$。

考虑当 $x_2 = 0$ 时：
$$f(x_1, 0) = 2x_1^3 + 3(0)^2 + 3x_1^2(0) - 24(0) = 2x_1^3$$

当 $x_1 \to +\infty$ 时，$f(x_1, 0) \to +\infty$
当 $x_1 \to -\infty$ 时，$f(x_1, 0) \to -\infty$

这说明函数在 $x_1 \to -\infty$ 时趋向于 $-\infty$，因此：
- 函数没有全局最小值（可以趋向于 $-\infty$）
- 函数没有全局最大值（在 $x_1 \to +\infty$ 或 $x_2 \to \pm\infty$ 时趋向于 $+\infty$）

**结论**：该函数既没有全局最小值也没有全局最大值，只存在局部驻点（包括一个局部极小点和两个鞍点）。

### 例7：修正的凸函数例题

**题目**：考虑函数 $f(x_1, x_2) = x_1^2 + x_2^2 - 4x_1 + 4$，求其驻点并判断性质。

**解**：

**步骤1：计算梯度**

$$\frac{\partial f}{\partial x_1} = 2x_1 - 4$$
$$\frac{\partial f}{\partial x_2} = 2x_2$$

梯度为：
$$\nabla f(x) = \begin{pmatrix} 2x_1 - 4 \\ 2x_2 \end{pmatrix}$$

**步骤2：求解驻点**

令 $\nabla f(x) = 0$：
$$\begin{cases} 2x_1 - 4 = 0 \\ 2x_2 = 0 \end{cases}$$

解得：$x_1 = 2$，$x_2 = 0$

因此驻点为：$x^* = (2, 0)$

**步骤3：计算Hessian矩阵**

$$\nabla^2 f(x) = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

**步骤4：判断性质**

Hessian矩阵是正定矩阵（特征值均为正），因此：
- $x^* = (2, 0)$ 是严格局部极小点
- 由于Hessian矩阵正定，函数是严格凸函数
- 严格凸函数只有一个全局最小值，因此 $x^* = (2, 0)$ 也是全局最小值

**验证**：$f(2, 0) = 4 + 0 - 8 + 4 = 0$

### 例8：拉格朗日对偶问题

### 问题描述

考虑非线性规划问题（原问题）：

$$\begin{align}
\min \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \geq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, l \\
& x \in D
\end{align}$$

其中约束分为两种：一种是等式约束和不等式约束，另一种写成集约束的形式，即 $x \in D$。如果将问题写成只有前一种约束的情形，就认为 $D = \mathbb{R}^n$。

**问题**：
1. 写出上述原问题的对偶问题
2. 写出并证明弱对偶定理

### 解答

#### (1) 对偶问题的构造

**步骤1**：构造拉格朗日函数
$$L(x, \lambda, \mu) = f(x) - \sum_{i=1}^m \lambda_i g_i(x) - \sum_{j=1}^l \mu_j h_j(x)$$

其中 $\lambda_i \geq 0$ 为不等式约束的拉格朗日乘子，$\mu_j$ 为等式约束的拉格朗日乘子。

**步骤2**：定义对偶函数
$$g(\lambda, \mu) = \inf_{x \in D} L(x, \lambda, \mu) = \inf_{x \in D} \left[ f(x) - \sum_{i=1}^m \lambda_i g_i(x) - \sum_{j=1}^l \mu_j h_j(x) \right]$$

**步骤3**：构造对偶问题
$$\begin{align}
\max \quad & g(\lambda, \mu) \\
\text{s.t.} \quad & \lambda_i \geq 0, \quad i = 1, \ldots, m
\end{align}$$

#### (2) 弱对偶定理的证明

**定理**：设 $x$ 是原问题的可行解，$(\lambda, \mu)$ 是对偶问题的可行解，则
$$f(x) \geq g(\lambda, \mu)$$

**证明**：

由于 $x$ 是原问题的可行解，有：
- $g_i(x) \geq 0, \quad i = 1, \ldots, m$
- $h_j(x) = 0, \quad j = 1, \ldots, l$
- $x \in D$

由于 $(\lambda, \mu)$ 是对偶问题的可行解，有：
- $\lambda_i \geq 0, \quad i = 1, \ldots, m$

因此：
$$\sum_{i=1}^m \lambda_i g_i(x) \geq 0 \quad \text{（因为 } \lambda_i \geq 0 \text{ 且 } g_i(x) \geq 0\text{）}$$

$$\sum_{j=1}^l \mu_j h_j(x) = 0 \quad \text{（因为 } h_j(x) = 0\text{）}$$

于是：
$$L(x, \lambda, \mu) = f(x) - \sum_{i=1}^m \lambda_i g_i(x) - \sum_{j=1}^l \mu_j h_j(x) \leq f(x)$$

由于 $g(\lambda, \mu) = \inf_{x \in D} L(x, \lambda, \mu)$，而 $x \in D$，所以：
$$g(\lambda, \mu) \leq L(x, \lambda, \mu) \leq f(x)$$

因此：<g>$f(x) \geq g(\lambda, \mu)$</g>

**证毕**。

### 对偶间隙

**定义**：对偶间隙（Duality Gap）为：
$$\text{Gap} = f(x^*) - g(\lambda^*, \mu^*)$$

其中 $$x^*$$ 是原问题的最优解，$$(\lambda^*, \mu^*)$$ 是对偶问题的最优解。

**性质**：
- 弱对偶定理保证：<i>Gap ≥ 0</i>
- 当 Gap = 0 时，称为<k>强对偶</k>
- 强对偶成立的条件：<k>Slater条件</k>或<k>KKT条件</k>

---


## 例9：邻近算子的计算

### 问题描述

计算以下函数的邻近算子：

1. $f(x) = \|x\|$（绝对值函数）
2. $f(x) = \frac{1}{2}\|x\|_2^2$（二次函数）
3. $f(x) = \max(0, x)$（ReLU函数）

其中邻近算子定义为：
$$\text{prox}_f(y) = \arg\min_{x} \left\{ f(x) + \frac{1}{2}\|x - y\|_2^2 \right\}$$

### 解答

#### (1) $f(x) = \|x\|$ 的邻近算子

**问题**：$$\text{prox}_f(y) = \arg\min_{x} \left\{ \|x\| + \frac{1}{2}(x - y)^2 \right\}$$

**求解过程**：

设 $g(x) = \|x\| + \frac{1}{2}(x - y)^2$

**情况1**：$x \geq 0$
$$g(x) = x + \frac{1}{2}(x - y)^2$$

求导：$g'(x) = 1 + (x - y) = x - y + 1$

令 $g'(x) = 0$：$x = y - 1$

当 $y - 1 \geq 0$，即 $y \geq 1$ 时，$x^* = y - 1$

**情况2**：$x < 0$
$$g(x) = -x + \frac{1}{2}(x - y)^2$$

求导：$g'(x) = -1 + (x - y) = x - y - 1$

令 $g'(x) = 0$：$x = y + 1$

当 $y + 1 < 0$，即 $y < -1$ 时，$x^* = y + 1$

**情况3**：$-1 \leq y \leq 1$

在 $x = 0$ 处，$g(0) = \frac{1}{2}y^2$

比较边界值：
- $g(0) = \frac{1}{2}y^2$
- $g(y-1) = \|y-1\| + \frac{1}{2} = 1 - y + \frac{1}{2} = \frac{3}{2} - y$（当 $y \geq 1$）
- $g(y+1) = \|y+1\| + \frac{1}{2} = -1 - y + \frac{1}{2} = -\frac{1}{2} - y$（当 $y \leq -1$）

当 $-1 \leq y \leq 1$ 时，$x^* = 0$ 是最优解。

**结论**：
$$\text{prox}_f(y) = \begin{cases}
y - 1, & \text{if } y > 1 \\
0, & \text{if } -1 \leq y \leq 1 \\
y + 1, & \text{if } y < -1
\end{cases}$$

这就是<k>软阈值函数</k>（Soft Thresholding Function）。

#### (2) $f(x) = \frac{1}{2}\|x\|_2^2$ 的邻近算子

**问题**：$$\text{prox}_f(y) = \arg\min_{x} \left\{ \frac{1}{2}\|x\|_2^2 + \frac{1}{2}\|x - y\|_2^2 \right\}$$

**求解过程**：

设 $g(x) = \frac{1}{2}\|x\|_2^2 + \frac{1}{2}\|x - y\|_2^2 = \frac{1}{2}\|x\|_2^2 + \frac{1}{2}\|x\|_2^2 - x^T y + \frac{1}{2}\|y\|_2^2$

$$= \|x\|_2^2 - x^T y + \frac{1}{2}\|y\|_2^2$$

求梯度：$\nabla g(x) = 2x - y$

令 $\nabla g(x) = 0$：$2x - y = 0$，即 $x = \frac{y}{2}$

**结论**：<g>$\text{prox}_f(y) = \frac{y}{2}$</g>

#### (3) $f(x) = \max(0, x)$ 的邻近算子

**问题**：$$\text{prox}_f(y) = \arg\min_{x} \left\{ \max(0, x) + \frac{1}{2}(x - y)^2 \right\}$$

**求解过程**：

设 $g(x) = \max(0, x) + \frac{1}{2}(x - y)^2$

**情况1**：$x \geq 0$
$$g(x) = x + \frac{1}{2}(x - y)^2$$

求导：$g'(x) = 1 + (x - y) = x - y + 1$

令 $g'(x) = 0$：$x = y - 1$

当 $y - 1 \geq 0$，即 $y \geq 1$ 时，$x^* = y - 1$

**情况2**：$x < 0$
$$g(x) = 0 + \frac{1}{2}(x - y)^2$$

求导：$g'(x) = x - y$

令 $g'(x) = 0$：$x = y$

当 $y < 0$ 时，$x^* = y$

**情况3**：$0 \leq y < 1$

在 $x = 0$ 处，$g(0) = \frac{1}{2}y^2$

比较 $g(0)$ 和 $g(y-1)$：
- $g(0) = \frac{1}{2}y^2$
- $g(y-1) = (y-1) + \frac{1}{2} = y - \frac{1}{2}$

当 $0 \leq y < 1$ 时，$g(0) < g(y-1)$，所以 $x^* = 0$

**结论**：
$$\text{prox}_f(y) = \begin{cases}
y - 1, & \text{if } y > 1 \\
0, & \text{if } y \leq 1
\end{cases}$$

### 例10：共轭函数的计算

计算以下函数的共轭函数：

1. $f(x) = \frac{1}{2}\|x\|_2^2$  
2. $f(x)=\|x\|$
3. $f(x) = \max(0, x)$

其中共轭函数定义为：
$$f^*(y) = \sup_{x} \{ \langle x, y \rangle - f(x) \}$$

### 解答

#### (1) $f(x) = \frac{1}{2}\|x\|_2^2$ 的共轭函数

**问题**：$$ f^*(y) = \sup_{x} \{ x^T y - \frac{1}{2}\|x\|_2^2 \} $$

**求解过程**：

设 $$g(x) = x^T y - \frac{1}{2}\|x\|_2^2$$

求梯度：$\nabla g(x) = y - x$

令 $\nabla g(x) = 0$：$y - x = 0$，即 $x = y$

因此：
$$f^*(y) = g(y) = y^T y - \frac{1}{2}\|y\|_2^2 = \frac{1}{2}\|y\|_2^2$$

**结论**：<g>$f^*(y) = \frac{1}{2}\|y\|_2^2$</g>

**验证**：$f^{**}(x) = \frac{1}{2}\|x\|_2^2 = f(x)$，满足共轭函数的性质。

#### (2) $f(x) = \|x\|$ 的共轭函数

**问题**：$f^*(y) = \sup_{x} \{ xy - \|x\| \}$

**求解过程**：

设 $g(x) = xy - \|x\|$

**情况1**：$x \geq 0$
$$g(x) = xy - x = x(y - 1)$$

- 当 $y > 1$ 时，$g(x) \to +\infty$ 当 $x \to +\infty$
- 当 $y = 1$ 时，$g(x) = 0$ 对所有 $x \geq 0$
- 当 $y < 1$ 时，$g(x) \to -\infty$ 当 $x \to +\infty$，最大值在 $x = 0$ 处

**情况2**：$x < 0$
$$g(x) = xy - (-x) = x(y + 1)$$

- 当 $y < -1$ 时，$g(x) \to +\infty$ 当 $x \to -\infty$
- 当 $y = -1$ 时，$g(x) = 0$ 对所有 $x < 0$
- 当 $y > -1$ 时，$g(x) \to -\infty$ 当 $x \to -\infty$，最大值在 $x = 0$ 处

**综合分析**：
- 当 $\|y\| > 1$ 时，$f^*(y) = +\infty$
- 当 $\|y\| \leq 1$ 时，$f^*(y) = 0$

**结论**：
$$f^*(y) = \begin{cases}
0, & \text{if } |y| \leq 1 \\
+\infty, & \text{if } |y| > 1
\end{cases}$$

这就是<k>单位球的指示函数</k>。

#### (3) $f(x) = \max(0, x)$ 的共轭函数

**问题**：$f^*(y) = \sup_{x} \{ xy - \max(0, x) \}$

**求解过程**：

设 $g(x) = xy - \max(0, x)$

**情况1**：$x \geq 0$
$$g(x) = xy - x = x(y - 1)$$

- 当 $y > 1$ 时，$g(x) \to +\infty$ 当 $x \to +\infty$
- 当 $y = 1$ 时，$g(x) = 0$ 对所有 $x \geq 0$
- 当 $y < 1$ 时，$g(x) \to -\infty$ 当 $x \to +\infty$，最大值在 $x = 0$ 处

**情况2**：$x < 0$
$$g(x) = xy - 0 = xy$$

- 当 $y > 0$ 时，$g(x) \to -\infty$ 当 $x \to -\infty$，最大值在 $x = 0$ 处
- 当 $y = 0$ 时，$g(x) = 0$ 对所有 $x < 0$
- 当 $y < 0$ 时，$g(x) \to +\infty$ 当 $x \to -\infty$

**综合分析**：
- 当 $y > 1$ 时，$f^*(y) = +\infty$
- 当 $y \leq 1$ 时，$f^*(y) = 0$

**结论**：
$$f^*(y) = \begin{cases}
0, & \text{if } y \leq 1 \\
+\infty, & \text{if } y > 1
\end{cases}$$

### 例11：对偶问题的求解

考虑以下优化问题：

$$\begin{align}
\min \quad & f(x) = x_1^2 + x_2^2 \\
\text{s.t.} \quad & g_1(x) = x_1 + x_2 - 1 \geq 0 \\
& g_2(x) = x_1 - x_2 \geq 0
\end{align}$$

1. 写出对偶问题
2. 求解对偶问题
3. 验证强对偶是否成立

### 解答

#### (1) 构造对偶问题

**步骤1**：构造拉格朗日函数
$$L(x, \lambda) = x_1^2 + x_2^2 - \lambda_1(x_1 + x_2 - 1) - \lambda_2(x_1 - x_2)$$

**步骤2**：定义对偶函数
$$g(\lambda) = \inf_{x} L(x, \lambda) = \inf_{x} \{ x_1^2 + x_2^2 - \lambda_1(x_1 + x_2 - 1) - \lambda_2(x_1 - x_2) \}$$

**步骤3**：求解对偶函数

对 $x_1$ 求偏导：
$$\frac{\partial L}{\partial x_1} = 2x_1 - \lambda_1 - \lambda_2 = 0 \Rightarrow x_1 = \frac{\lambda_1 + \lambda_2}{2}$$

对 $x_2$ 求偏导：
$$\frac{\partial L}{\partial x_2} = 2x_2 - \lambda_1 + \lambda_2 = 0 \Rightarrow x_2 = \frac{\lambda_1 - \lambda_2}{2}$$

代入拉格朗日函数：
$$g(\lambda) = \left(\frac{\lambda_1 + \lambda_2}{2}\right)^2 + \left(\frac{\lambda_1 - \lambda_2}{2}\right)^2 - \lambda_1\left(\frac{\lambda_1 + \lambda_2}{2} + \frac{\lambda_1 - \lambda_2}{2} - 1\right) - \lambda_2\left(\frac{\lambda_1 + \lambda_2}{2} - \frac{\lambda_1 - \lambda_2}{2}\right)$$

$$= \frac{(\lambda_1 + \lambda_2)^2 + (\lambda_1 - \lambda_2)^2}{4} - \lambda_1(\lambda_1 - 1) - \lambda_2 \lambda_2$$

$$= \frac{2\lambda_1^2 + 2\lambda_2^2}{4} - \lambda_1^2 + \lambda_1 - \lambda_2^2$$

$$= \frac{\lambda_1^2 + \lambda_2^2}{2} - \lambda_1^2 + \lambda_1 - \lambda_2^2$$

$$= -\frac{\lambda_1^2 + \lambda_2^2}{2} + \lambda_1$$

**步骤4**：构造对偶问题
$$\begin{align}
\max \quad & g(\lambda) = -\frac{\lambda_1^2 + \lambda_2^2}{2} + \lambda_1 \\
\text{s.t.} \quad & \lambda_1 \geq 0, \quad \lambda_2 \geq 0
\end{align}$$

#### (2) 求解对偶问题

**对偶问题的KKT条件**：

对 $\lambda_1$ 求偏导：
$$\frac{\partial g}{\partial \lambda_1} = -\lambda_1 + 1 = 0 \Rightarrow \lambda_1 = 1$$

对 $\lambda_2$ 求偏导：
$$\frac{\partial g}{\partial \lambda_2} = -\lambda_2 = 0 \Rightarrow \lambda_2 = 0$$

**对偶问题的最优解**：$\lambda_1^* = 1, \lambda_2^* = 0$

**对偶问题的最优值**：
$$g(\lambda^*) = -\frac{1^2 + 0^2}{2} + 1 = -\frac{1}{2} + 1 = \frac{1}{2}$$

#### (3) 验证强对偶

**原问题的最优解**：

从对偶解可以反推原问题的最优解：
$$x_1^* = \frac{\lambda_1^* + \lambda_2^*}{2} = \frac{1 + 0}{2} = \frac{1}{2}$$

$$x_2^* = \frac{\lambda_1^* - \lambda_2^*}{2} = \frac{1 - 0}{2} = \frac{1}{2}$$

**验证可行性**：
- $g_1(x^*) = \frac{1}{2} + \frac{1}{2} - 1 = 0 \geq 0$ ✓
- $g_2(x^*) = \frac{1}{2} - \frac{1}{2} = 0 \geq 0$ ✓

**原问题的最优值**：
$$f(x^*) = \left(\frac{1}{2}\right)^2 + \left(\frac{1}{2}\right)^2 = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$$

**强对偶验证**：
$$f(x^*) = \frac{1}{2} = g(\lambda^*)$$

因此，<g>强对偶成立</g>！

## 五、总结

通过以上例题，我们学习了：

1. **凸函数判断**：通过定义验证函数是否满足凸性条件
2. **二次型函数性质**：二次型函数 $x^T Q x$ 的凸性与矩阵 $Q$ 的正定性密切相关
3. **凸规划问题识别**：需要同时检查目标函数和约束函数的凸性
4. **梯度与Hessian计算**：对于二次型函数，可以系统地计算其梯度和Hessian矩阵
5. **无约束优化问题**：通过梯度和Hessian矩阵判断驻点的性质（局部极小点、局部极大点、鞍点）
6. **全局性质分析**：判断函数是否存在全局最优解，区分局部和全局最优性
7. **Hessian矩阵判别法**：利用主子式的符号判断矩阵的正定性、负定性或不定性
8. **拉格朗日对偶理论**：构造对偶问题的方法和弱对偶定理的证明
9. **对偶间隙概念**：理解强对偶和弱对偶的区别，掌握对偶理论的实际应用
10. **邻近算子计算**：掌握邻近算子的定义和计算方法，理解软阈值函数等经典结果
11. **共轭函数理论**：学会计算函数的共轭函数，理解其对偶性质
12. **对偶问题求解**：通过具体例题掌握对偶问题的构造、求解和强对偶验证

---

> **返回**: [《最优化理论基础》](/2025/09/08/optimization-fundamentals/)

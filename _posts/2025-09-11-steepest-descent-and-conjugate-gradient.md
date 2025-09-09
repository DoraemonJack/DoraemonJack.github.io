---
layout:       post
title:        "最优化方法——最速下降法与共轭梯度法"
subtitle:     "迭代公式、几何直观与中等难度例题"
date:         2025-09-11 10:00:00
author:       "DoraemonJack"
header-style: text
header-img: "img/post-bg-math.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
  - Optimization
  - 最优化算法
  - Steepest Descent
  - Conjugate Gradient
  - Numerical Optimization
---

最速下降法（Steepest Descent, SD）和共轭梯度法（Conjugate Gradient, CG）是经典的一阶迭代优化方法。SD以“最陡方向”作为下降方向，配合（一维）线搜索求步长；CG在二次型与对称正定（SPD）情形下能在至多 n 步内精确收敛，且在大规模稀疏问题中极为高效。本文给出两法的迭代公式、收敛要点与中等难度例题推导，并在末尾做对比。

<div class="alert alert-warning" role="alert">
  <strong style="font-size:1.2em">推荐阅读：</strong>
  <a href="https://flat2010.github.io/2018/10/26/%E5%85%B1%E8%BD%AD%E6%A2%AF%E5%BA%A6%E6%B3%95%E9%80%9A%E4%BF%97%E8%AE%B2%E4%B9%89/#4-%E6%9C%80%E9%80%9F%E4%B8%8B%E9%99%8D%E6%B3%95" target="_blank" rel="noopener">共轭梯度法通俗讲义</a>
  <span style="margin-left:12px">（入门与直观极佳）</span>
  <span style="float:right">🔗</span>
</div>


## 一、最速下降法（Steepest Descent, SD）

### 1.1 问题与思路

给定可微目标 $f: \mathbb{R}^n \to \mathbb{R}$，在点 $x_k$ 处的最速下降方向是使得单位步前向下降最快的方向，即负梯度方向：$p_k = -\nabla f(x_k)$。

### 1.2 迭代公式（含精确线搜索）

记 $g_k = \nabla f(x_k)$，则最速下降法为：

$$
\begin{aligned}
&\text{方向：}\quad p_k = -g_k \\
&\text{步长：}\quad \alpha_k = \underset{\alpha > 0}{\arg\min}\ f(x_k + \alpha p_k) \\
&\text{更新：}\quad x_{k+1} = x_k + \alpha_k p_k
\end{aligned}
$$

对二次型（最常见的精确可解情形）

$$f(x) = \tfrac{1}{2} x^\top A x - b^\top x,\quad A = A^\top \succ 0,$$

有 $g_k = Ax_k - b$。

精确线搜索的闭式步长：

$$\alpha_k = \frac{g_k^\top g_k}{g_k^\top A g_k}.$$

性质（精确线搜索）：$g_{k+1}^\top g_k = 0$（相邻梯度正交）。

### 1.3 收敛速率（SPD 二次型）

设 $\kappa(A) = \lambda_{\max}(A)/\lambda_{\min}(A)$ 为条件数，则 SD 的最坏线性收敛率满足：

$$\frac{\|x_{k+1}-x_*\|_A}{\|x_k-x_*\|_A} \le \rho_{\mathrm{SD}} := \left(\frac{\kappa(A)-1}{\kappa(A)+1}\right)^2.$$

条件数越大，收敛越慢。预条件（等价于在加权范数下取“最速”方向）可显著改善表现。

### 1.4 例题：二次规划的最速下降法

考虑

$$f(x) = \tfrac{1}{2} x^\top A x - b^\top x,\quad A = \begin{pmatrix}4 & 1\\ 1 & 3\end{pmatrix} \succ 0,\quad b = \begin{pmatrix}1\\2\end{pmatrix}.$$

最优解满足 $Ax_* = b$，易得 $x_* = (1/11,\,7/11)^\top$。以 $x_0=(0,0)^\top$ 为初值，精确线搜索的 SD：

- $g_0 = Ax_0 - b = (-1,-2)^\top$，$\alpha_0 = \dfrac{g_0^\top g_0}{g_0^\top A g_0} = \dfrac{5}{20} = \tfrac{1}{4}$，$x_1 = x_0 - \alpha_0 g_0 = (\tfrac{1}{4},\tfrac{1}{2})^\top$。
- $g_1 = Ax_1 - b = (0.5,-0.25)^\top$，$\alpha_1 = \dfrac{g_1^\top g_1}{g_1^\top A g_1} = \dfrac{5/16}{15/16} = \tfrac{1}{3}$，$x_2 = x_1 - \alpha_1 g_1 = (\tfrac{1}{12},\tfrac{7}{12})^\top$。
- $g_2 = Ax_2 - b = (-\tfrac{1}{12},-\tfrac{1}{6})^\top$，$\alpha_2 = \tfrac{1}{4}$，$x_3 = (\tfrac{5}{48},\tfrac{5}{8})^\top$。

可见逐步逼近 $x_*=(1/11,7/11)^\top$。该例的 $\kappa(A) = \dfrac{7+\sqrt{5}}{7-\sqrt{5}} \approx 1.939$，$\rho_{\mathrm{SD}} \approx 0.102$，因此收敛较快但非有限步。

---

## 二、共轭梯度法（Conjugate Gradient, CG）

### 2.1 适用场景

用于 SPD 二次型

$$f(x) = \tfrac{1}{2} x^\top A x - b^\top x,\quad A = A^\top \succ 0,$$

或等价的线性方程组 $Ax=b$。CG 在精确算术下至多 n 步达到精确解，并且只需矩阵-向量乘（无需存储或分解矩阵），适合大规模稀疏问题。

### 2.2 迭代公式（线性 CG 标准形式）

记残差 $r_k = b - A x_k$（即负梯度 $-g_k$），方向 $d_k$ 满足 A-共轭：$d_i^\top A d_j = 0$ for $i\neq j$。算法：

$$
\begin{aligned}
&r_0 = b - A x_0,\quad d_0 = r_0.\\
&\alpha_k = \frac{r_k^\top r_k}{d_k^\top A d_k}.\\
&x_{k+1} = x_k + \alpha_k d_k.\\
&r_{k+1} = r_k - \alpha_k A d_k.\\
&\beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}\quad(\text{Fletcher–Reeves}).\\
&d_{k+1} = r_{k+1} + \beta_k d_k.
\end{aligned}
$$

性质：误差在 A-范下最佳逼近于 Krylov 子空间；在精确算术下对 n 维问题至多 n 步收敛；在有限精度下收敛按 $\mathcal{O}\!\left((\tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1})^k\right)$ 估计，$\kappa=\kappa(A)$。预条件 CG（PCG）将 $A$ 替换为 $M^{-1}A$ 以降低条件数。

### 2.3 例题：二次规划的最速下降法

沿用 $A, b$，取 $x_0 = (0,0)^\top$。

1) $r_0 = b - Ax_0 = (1,2)^\top$，$d_0 = r_0$。$A d_0 = (6,7)^\top$，$\alpha_0 = \dfrac{r_0^\top r_0}{d_0^\top A d_0} = \dfrac{5}{20} = \tfrac{1}{4}$。$x_1 = (\tfrac{1}{4},\tfrac{1}{2})^\top$，$r_1 = r_0 - \alpha_0 A d_0 = (-\tfrac{1}{2},\tfrac{1}{4})^\top$。

2) $\beta_0 = \dfrac{r_1^\top r_1}{r_0^\top r_0} = \dfrac{5/16}{5} = \tfrac{1}{16}$，$d_1 = r_1 + \beta_0 d_0 = (-\tfrac{7}{16},\tfrac{3}{8})^\top$。$A d_1 = (-\tfrac{11}{8},\tfrac{11}{16})^\top$。$\alpha_1 = \dfrac{r_1^\top r_1}{d_1^\top A d_1} = \dfrac{5/16}{55/64} = \tfrac{4}{11}$。于是

$$x_2 = x_1 + \alpha_1 d_1 = \begin{pmatrix}1/11\\7/11\end{pmatrix} = x_*,\quad r_2 = 0.$$

二步到达最优解，体现了 CG 在 SPD 二次型上的有限步收敛性。

### 2.4 预条件共轭梯度法 (PCG)

#### 2.4.1 动机与原理

标准 CG 的收敛速度严重依赖于系数矩阵 $A$ 的条件数 $\kappa(A)$。当 $\kappa(A)$ 很大时，CG 收敛缓慢。预条件的核心思想是通过可逆矩阵 $M$ 将原问题

$$Ax = b$$

变换为条件数更小的等价问题。常见的变换策略有：

**左预条件**：$M^{-1}Ax = M^{-1}b$  
**右预条件**：$AM^{-1}y = b$，其中 $x = M^{-1}y$  
**对称预条件**：$M^{-1/2}AM^{-1/2}z = M^{-1/2}b$，其中 $x = M^{-1/2}z$

#### 2.4.2 PCG 算法（对称预条件版本）

设 $M = M^T \succ 0$ 为预条件矩阵，PCG 算法为：

$$
\begin{aligned}
&r_0 = b - A x_0,\quad z_0 = M^{-1}r_0,\quad d_0 = z_0.\\
&\alpha_k = \frac{r_k^T z_k}{d_k^T A d_k}.\\
&x_{k+1} = x_k + \alpha_k d_k.\\
&r_{k+1} = r_k - \alpha_k A d_k.\\
&z_{k+1} = M^{-1}r_{k+1}.\\
&\beta_k = \frac{r_{k+1}^T z_{k+1}}{r_k^T z_k}.\\
&d_{k+1} = z_{k+1} + \beta_k d_k.
\end{aligned}
$$

**关键观察**：
- 当 $M = I$ 时，PCG 退化为标准 CG
- 方向 $d_k$ 关于 $A$ 共轭：$d_i^T A d_j = 0$ for $i \neq j$
- 收敛速度取决于 $\kappa(M^{-1}A)$ 而非 $\kappa(A)$

#### 2.4.3 预条件子的选择

理想的预条件子 $M$ 应满足：
1. **$M^{-1}A$ 的条件数小**：$\kappa(M^{-1}A) \ll \kappa(A)$
2. **求解 $Mz = r$ 廉价**：每步需要计算 $z_k = M^{-1}r_k$
3. **存储需求合理**：$M$ 的表示和存储开销可接受

**常见预条件子**：

| 预条件子类型 | 定义 | 适用场景 | 优缺点 |
|------------|------|---------|--------|
| **Jacobi** | $M = \text{diag}(A)$ | 对角占优矩阵 | 简单但效果有限 |
| **SSOR** | $M = (D+L)D^{-1}(D+U)$ | 中等规模稠密问题 | 平衡效果与开销 |
| **不完全Cholesky** | $M = LL^T$（近似分解） | 稀疏 SPD 系统 | 效果好但构造复杂 |
| **多重网格** | 多层次网格校正 | 偏微分方程离散 | 最优复杂度但实现困难 |
| **代数多重网格** | 基于矩阵图的粗化 | 一般稀疏问题 | 黑盒式，适应性强 |

#### 2.4.4 PCG 例题

考虑带预条件的二次规划：

$$A = \begin{pmatrix} 4 & 1 \\ 1 & 16 \end{pmatrix}, \quad b = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$

**分析**：$\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}} \approx \frac{16.06}{3.94} \approx 4.08$

**选择 Jacobi 预条件子**：$M = \text{diag}(A) = \begin{pmatrix} 4 & 0 \\ 0 & 16 \end{pmatrix}$

则 $M^{-1}A = \begin{pmatrix} 1 & 0.25 \\ 0.0625 & 1 \end{pmatrix}$，$\kappa(M^{-1}A) \approx 1.32 \ll 4.08$

**PCG 迭代**（$x_0 = (0,0)^T$）：

1) $r_0 = (1,2)^T$，$z_0 = M^{-1}r_0 = (0.25, 0.125)^T$，$d_0 = z_0$

2) $\alpha_0 = \frac{r_0^T z_0}{d_0^T A d_0} = \frac{1.25}{1.32} \approx 0.947$

3) $x_1 = (0.237, 0.118)^T$，收敛显著加速

**效果对比**：
- 无预条件 CG：约 4-5 步收敛
- Jacobi PCG：约 2-3 步收敛
- 理想预条件：1 步精确收敛

#### 2.4.5 PCG 的理论性质

**收敛定理**：设 $M = M^T \succ 0$，则 PCG 的收敛速度满足：

$$\frac{\|x_{k+1} - x_*\|_A}{\|x_k - x_*\|_A} \leq 2\left(\frac{\sqrt{\kappa(M^{-1}A)} - 1}{\sqrt{\kappa(M^{-1}A)} + 1}\right)^k$$

**最优预条件**：当 $M = A$ 时，$\kappa(M^{-1}A) = 1$，PCG 一步收敛。但求解 $Az = r$ 等价于原问题，失去意义。

**实际策略**：寻找 $M \approx A$ 但 $M^{-1}$ 易于计算的矩阵，在预条件效果与计算开销间取得平衡。

---

## 三、最速下降法与共轭梯度法对比

| 比较维度 | 最速下降法 (SD) | 共轭梯度法 (CG) |
|---------|----------------|----------------|
| **适用问题** | 任意可微函数 $f(x)$ | SPD 二次型 $\frac{1}{2}x^TAx - b^Tx$ |
| **方向选择** | 负梯度方向 $-\nabla f(x_k)$ | A-共轭方向 $d_k$，满足 $d_i^T A d_j = 0$ |
| **收敛性质** | 线性收敛，速率 $\left(\frac{\kappa-1}{\kappa+1}\right)^2$ | 理论上 $n$ 步精确收敛 |
| **实际收敛** | 受条件数影响显著，可能锯齿形收敛 | 浮点环境下按 $\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$ 衰减 |
| **每步计算** | 1次梯度 + 1次线搜索 | 1次矩阵向量乘 + 若干内积 |
| **存储需求** | $O(n)$：当前点、梯度 | $O(n)$：当前点、残差、方向 |
| **数值稳定性** | 相邻梯度正交，但步长可能过小 | 方向A-共轭，充分利用二次型结构 |
| **预条件化** | 可用预条件矩阵 $M$ | PCG：用 $M^{-1}A$ 降低条件数 |
| **优势** | • 通用性强<br>• 实现简单<br>• 内存友好 | • 有限步收敛<br>• 收敛快速<br>• 适合大规模稀疏问题 |
| **劣势** | • 收敛慢<br>• 易受条件数影响<br>• 锯齿形路径 | • 仅适用SPD系统<br>• 数值误差影响<br>• 重启可能需要 |
| **典型应用** | • 一般非线性优化初始方法<br>• 教学演示 | • 大规模线性系统求解<br>• 二次规划<br>• 有限元分析 |

### 实践建议

**选择原则**：
- **SPD线性系统** → 优先选择 **CG/PCG**
- **一般非线性优化** → SD作为基线，实际推荐 **L-BFGS** 等拟牛顿法
- **大规模稀疏问题** → **预条件共轭梯度法 (PCG)**
- **教学与原理验证** → **最速下降法** 简单易懂

**性能提升策略**：
- SD：使用预条件、自适应步长、重启策略
- CG：选择合适预条件子、监控数值稳定性、必要时重启

---

## 相关例题

如果您想查看本文所有的相关例题，请点击下方链接：

<div style="text-align: center; margin: 30px 0;">
  <a href="/2025/09/12/steepest-descent-conjugate-gradient-examples/" class="btn btn-primary btn-lg" style="display: inline-block; padding: 15px 30px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; border: none;">
    <i class="fa fa-code" style="margin-right: 10px;"></i>查看相关例题
  </a>
</div>


> **相关文章**:
> - [《最优化理论基础》](/2025/09/08/optimization-fundamentals/)
> - [《最优化理论基础例题集》](/2025/09/09/optimization-fundamentals-examples/)



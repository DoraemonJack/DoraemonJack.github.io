---
layout:       post
title:        "最优化理论——拉格朗日乘子法与增广拉格朗日乘子法"
subtitle:     "从经典约束优化到现代算法的完整解析"
date:         2025-09-15 10:00:00
author:       "DoraemonJack"
header-style: text
header-img: "img/post-bg-math.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - Optimization
    - 最优化算法
    - Lagrange Multipliers
    - Augmented Lagrangian
    - Constrained Optimization
    - ADMM
---

本文深入探讨拉格朗日乘子法和增广拉格朗日乘子法的理论基础、算法推导和实际应用。从经典的等式约束优化出发，逐步介绍不等式约束、KKT条件、罚函数法、增广拉格朗日法，最终延伸到现代的ADMM算法。

<div class="alert alert-info" role="alert">
  <strong>前置阅读：</strong>
  <a href="/2025/09/10/kkt-conditions/" target="_blank">KKT条件详解</a>
  <span style="margin-left:8px">（建议先掌握约束优化的基础理论）</span>
</div>

## 一、拉格朗日乘子法的理论基础

### 1.1 约束优化问题的数学表述

考虑一般的约束优化问题：

$$\begin{align}
\min \quad & f(x) \\
\text{s.t.} \quad & h_i(x) = 0, \quad i = 1, 2, \ldots, m \\
& g_j(x) \leq 0, \quad j = 1, 2, \ldots, p
\end{align}$$

其中：
- $$f(x): \mathbb{R}^n \to \mathbb{R}$$ 是目标函数
- $$h_i(x) = 0$$ 是等式约束
- $$g_j(x) \leq 0$$ 是不等式约束
- $$x \in \mathbb{R}^n$$ 是决策变量

### 1.2 等式约束问题的几何直观

对于纯等式约束问题：
$$\begin{align}
\min \quad & f(x) \\
\text{s.t.} \quad & h(x) = 0
\end{align}$$

**几何解释**：
- **约束曲面**：$$h(x) = 0$$ 定义了 $$\mathbb{R}^n$$ 空间中的一个曲面
- **最优解条件**：在最优点 $$x^*$$，目标函数的梯度必须与约束曲面正交
- **数学表述**：$$\nabla f(x^*) = \lambda \nabla h(x^*)$$

这个条件表明：**在最优点，目标函数的梯度必须是约束函数梯度的线性组合**。

### 1.3 拉格朗日函数的构造

#### 1.3.1 拉格朗日函数的定义

对于约束优化问题，定义**拉格朗日函数**：
$$\boxed{L(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{j=1}^p \mu_j g_j(x)}$$

其中：
- $$\lambda = (\lambda_1, \ldots, \lambda_m)^T$$ 是**等式约束的拉格朗日乘子**
- $$\mu = (\mu_1, \ldots, \mu_p)^T$$ 是**不等式约束的拉格朗日乘子**

#### 1.3.2 拉格朗日函数的物理意义

拉格朗日函数可以理解为：
1. **原目标函数** $$f(x)$$
2. **加上约束违反的"惩罚"** $$\lambda_i h_i(x)$$ 和 $$\mu_j g_j(x)$$

**关键洞察**：通过引入拉格朗日乘子，我们将约束优化问题转化为无约束优化问题。

### 1.4 拉格朗日乘子法的数学推导

#### 1.4.1 等式约束的推导

考虑等式约束问题：
$$\min f(x) \quad \text{s.t.} \quad h(x) = 0$$

**步骤1**：构造拉格朗日函数
$$L(x, \lambda) = f(x) + \lambda^T h(x)$$

**步骤2**：必要条件推导

在最优点 $$(x^*, \lambda^*)$$，有：
$$\nabla_x L(x^*, \lambda^*) = \nabla f(x^*) + \lambda^{*T} \nabla h(x^*) = 0$$
$$\nabla_\lambda L(x^*, \lambda^*) = h(x^*) = 0$$

这就是著名的**拉格朗日条件**：
$$\boxed{\begin{cases}
\nabla f(x^*) + \lambda^{*T} \nabla h(x^*) = 0 \\
h(x^*) = 0
\end{cases}}$$

#### 1.4.2 几何解释

拉格朗日条件的几何意义：
1. **梯度正交性**：$$\nabla f(x^*)$$ 与约束曲面的切空间正交
2. **约束满足**：$$h(x^*) = 0$$ 确保解在可行域内
3. **乘子意义**：$$\lambda^*$$ 反映约束对目标函数的"影响程度"

### 1.5 不等式约束与KKT条件

#### 1.5.1 不等式约束的处理

对于不等式约束 $$g(x) \leq 0$$，情况更复杂：
- **非激活约束**：如果 $$g(x^*) < 0$$，约束不起作用，$$\mu^* = 0$$
- **激活约束**：如果 $$g(x^*) = 0$$，约束起作用，$$\mu^* \geq 0$$

#### 1.5.2 KKT条件的完整表述

**Karush-Kuhn-Tucker (KKT) 条件**：

设 $$x^*$$ 是约束优化问题的最优解，且满足约束规范条件，则存在拉格朗日乘子 $$\lambda^*, \mu^*$$ 使得：

1. **梯度条件**：
   $$\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla h_i(x^*) + \sum_{j=1}^p \mu_j^* \nabla g_j(x^*) = 0$$

2. **原始可行性**：
   $$h_i(x^*) = 0, \quad i = 1, \ldots, m$$
   $$g_j(x^*) \leq 0, \quad j = 1, \ldots, p$$

3. **对偶可行性**：
   $$\mu_j^* \geq 0, \quad j = 1, \ldots, p$$

4. **互补松弛条件**：
   $$\mu_j^* g_j(x^*) = 0, \quad j = 1, \ldots, p$$

### 1.6 约束规范条件

KKT条件成立需要满足**约束规范条件**，常见的有：

#### 1.6.1 线性无关约束规范 (LICQ)

在 $$x^*$$ 处，所有激活约束的梯度线性无关：
$$\{\nabla h_i(x^*), \nabla g_j(x^*) : j \in \mathcal{A}(x^*)\}$$ 线性无关

其中 $$\mathcal{A}(x^*) = \{j : g_j(x^*) = 0\}$$ 是激活约束集合。

#### 1.6.2 Mangasarian-Fromovitz约束规范 (MFCQ)

1. $$\{\nabla h_i(x^*)\}_{i=1}^m$$ 线性无关
2. 存在方向 $$d$$ 使得：
   - $$\nabla h_i(x^*)^T d = 0, \quad i = 1, \ldots, m$$
   - $$\nabla g_j(x^*)^T d < 0, \quad j \in \mathcal{A}(x^*)$$

---

## 二、经典拉格朗日乘子法的算法实现

### 2.1 等式约束的Newton-Lagrange方法

对于等式约束问题：
$$\min f(x) \quad \text{s.t.} \quad h(x) = 0$$

#### 2.1.1 算法推导

KKT条件为：
$$F(x, \lambda) = \begin{pmatrix} \nabla f(x) + \nabla h(x)^T \lambda \\ h(x) \end{pmatrix} = 0$$

使用Newton法求解：
$$\begin{pmatrix} x_{k+1} \\ \lambda_{k+1} \end{pmatrix} = \begin{pmatrix} x_k \\ \lambda_k \end{pmatrix} - \begin{pmatrix} \nabla^2_{xx} L_k & \nabla h(x_k)^T \\ \nabla h(x_k) & 0 \end{pmatrix}^{-1} \begin{pmatrix} \nabla f(x_k) + \nabla h(x_k)^T \lambda_k \\ h(x_k) \end{pmatrix}$$

其中 $$\nabla^2_{xx} L_k = \nabla^2 f(x_k) + \sum_{i=1}^m \lambda_{k,i} \nabla^2 h_i(x_k)$$

#### 2.1.2 算法框架

**算法1：Newton-Lagrange方法**

**输入**：初始点 $$(x_0, \lambda_0)$$，容差 $$\epsilon$$
**输出**：最优解 $$(x^*, \lambda^*)$$

1. **初始化**：$$k = 0$$
2. **while** $$\|F(x_k, \lambda_k)\| > \epsilon$$ **do**
3. 　　计算 $$\nabla f(x_k)$$，$$\nabla h(x_k)$$，$$\nabla^2 f(x_k)$$，$$\nabla^2 h_i(x_k)$$
4. 　　构造KKT系统矩阵
5. 　　求解线性系统得到 $$(\Delta x_k, \Delta \lambda_k)$$
6. 　　更新：$$(x_{k+1}, \lambda_{k+1}) = (x_k, \lambda_k) + (\Delta x_k, \Delta \lambda_k)$$
7. 　　$$k = k + 1$$
8. **end while**

### 2.2 不等式约束的序列二次规划 (SQP)

#### 2.2.1 SQP的基本思想

将原问题在当前点附近用**二次规划子问题**近似：
$$\begin{align}
\min_d \quad & \nabla f(x_k)^T d + \frac{1}{2} d^T B_k d \\
\text{s.t.} \quad & \nabla h_i(x_k)^T d + h_i(x_k) = 0, \quad i = 1, \ldots, m \\
& \nabla g_j(x_k)^T d + g_j(x_k) \leq 0, \quad j = 1, \ldots, p
\end{align}$$

其中 $$B_k$$ 是拉格朗日函数Hessian的近似（可用BFGS更新）。

#### 2.2.2 SQP算法框架

**算法2：序列二次规划**

1. 给定初始点 $$x_0$$，初始Hessian近似 $$B_0$$
2. **for** $$k = 0, 1, 2, \ldots$$ **do**
3. 　　求解二次规划子问题得到 $$(d_k, \lambda_{k+1}, \mu_{k+1})$$
4. 　　线搜索确定步长 $$\alpha_k$$
5. 　　更新：$$x_{k+1} = x_k + \alpha_k d_k$$
6. 　　更新Hessian近似：$$B_{k+1}$$ (使用BFGS)
7. **end for**

---

## 三、罚函数法与障碍函数法

### 3.1 罚函数法的基本思想

#### 3.1.1 外罚函数法

**基本思想**：通过添加惩罚项将约束问题转化为无约束问题。

对于约束问题：
$$\min f(x) \quad \text{s.t.} \quad h(x) = 0, \quad g(x) \leq 0$$

构造**外罚函数**：
$$\boxed{P(x, \rho) = f(x) + \rho \left[\sum_{i=1}^m h_i(x)^2 + \sum_{j=1}^p \max(0, g_j(x))^2\right]}$$

其中 $$\rho > 0$$ 是**罚参数**。

#### 3.1.2 外罚函数法的算法

**算法3：外罚函数法**

1. 选择初始点 $$x_0$$，初始罚参数 $$\rho_0 > 0$$，增长因子 $$\beta > 1$$
2. **for** $$k = 0, 1, 2, \ldots$$ **do**
3. 　　求解无约束问题：$$x_{k+1} = \arg\min_x P(x, \rho_k)$$
4. 　　**if** 收敛 **then** 停止
5. 　　更新罚参数：$$\rho_{k+1} = \beta \rho_k$$
6. **end for**

#### 3.1.3 外罚函数法的性质

**定理1**：设 $$\{x_k\}$$ 是外罚函数法生成的序列，$$\rho_k \to \infty$$，则：
1. 任何聚点都是原问题的最优解
2. 如果原问题有唯一最优解 $$x^*$$，则 $$x_k \to x^*$$

**缺点**：
- 需要 $$\rho \to \infty$$ 才能保证精确收敛
- 当 $$\rho$$ 很大时，罚函数变得病态

### 3.2 障碍函数法

#### 3.2.1 内点法的思想

**基本思想**：通过障碍函数防止迭代点离开可行域。

对于不等式约束问题：
$$\min f(x) \quad \text{s.t.} \quad g(x) < 0$$

构造**对数障碍函数**：
$$\boxed{B(x, \mu) = f(x) - \mu \sum_{j=1}^p \ln(-g_j(x))}$$

其中 $$\mu > 0$$ 是**障碍参数**。

#### 3.2.2 障碍函数法的算法

**算法4：障碍函数法**

1. 选择可行初始点 $$x_0$$（满足 $$g(x_0) < 0$$），初始障碍参数 $$\mu_0 > 0$$
2. **for** $$k = 0, 1, 2, \ldots$$ **do**
3. 　　求解：$$x_{k+1} = \arg\min_x B(x, \mu_k)$$
4. 　　**if** 收敛 **then** 停止
5. 　　更新：$$\mu_{k+1} = \sigma \mu_k$$，其中 $$0 < \sigma < 1$$
6. **end for**

#### 3.2.3 中心路径与KKT条件

障碍函数法的最优性条件：
$$\nabla f(x) - \mu \sum_{j=1}^p \frac{1}{-g_j(x)} \nabla g_j(x) = 0$$

定义 $$\mu_j = \frac{\mu}{-g_j(x)}$$，则：
$$\nabla f(x) + \sum_{j=1}^p \mu_j \nabla g_j(x) = 0$$
$$\mu_j g_j(x) = -\mu$$

当 $$\mu \to 0$$ 时，恢复KKT条件！

---

## 四、增广拉格朗日乘子法

### 4.1 增广拉格朗日函数的动机

#### 4.1.1 经典拉格朗日函数的局限性

考虑简单例子：
$$\min \frac{1}{2}x^2 \quad \text{s.t.} \quad x - 1 = 0$$

拉格朗日函数：$$L(x, \lambda) = \frac{1}{2}x^2 + \lambda(x - 1)$$

**问题**：$$\nabla^2_{xx} L = 1 > 0$$，Hessian总是正定，无法区分不同的 $$\lambda$$ 值对应的解。

#### 4.1.2 增广拉格朗日函数的构造

**增广拉格朗日函数**结合了拉格朗日函数和罚函数的优点：

$$\boxed{L_A(x, \lambda, \rho) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \frac{\rho}{2} \sum_{i=1}^m h_i(x)^2}$$

对于不等式约束，使用：
$$L_A(x, \lambda, \mu, \rho) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \frac{\rho}{2} \sum_{i=1}^m h_i(x)^2 + \sum_{j=1}^p \mu_j g_j(x) + \frac{\rho}{2} \sum_{j=1}^p [\max(0, g_j(x))]^2$$

### 4.2 增广拉格朗日函数的理论性质

#### 4.2.1 等价性定理

**定理2**：设 $$(x^*, \lambda^*)$$ 满足KKT条件，则对于任何 $$\rho > 0$$：
$$x^* = \arg\min_x L_A(x, \lambda^*, \rho)$$

**证明思路**：
在 $$x^*$$ 处，$$h(x^*) = 0$$，所以：
$$\nabla_x L_A(x^*, \lambda^*, \rho) = \nabla f(x^*) + \lambda^{*T} \nabla h(x^*) = 0$$

这正是KKT条件！

#### 4.2.2 Hessian矩阵的改善

增广拉格朗日函数的Hessian：
$$\nabla^2_{xx} L_A = \nabla^2 f(x) + \sum_{i=1}^m \lambda_i \nabla^2 h_i(x) + \rho \sum_{i=1}^m \nabla h_i(x) \nabla h_i(x)^T$$

**关键优势**：即使 $$\nabla^2 f(x) + \sum_{i=1}^m \lambda_i \nabla^2 h_i(x)$$ 不正定，增加的项 $$\rho \sum_{i=1}^m \nabla h_i(x) \nabla h_i(x)^T$$ 可以使整个Hessian正定。

### 4.3 增广拉格朗日乘子法算法

#### 4.3.1 基本算法框架

**算法5：增广拉格朗日乘子法 (ALM)**

**输入**：初始点 $$x_0$$，初始乘子 $$\lambda_0$$，罚参数 $$\rho_0$$
**输出**：最优解 $$x^*$$

1. **初始化**：$$k = 0$$
2. **repeat**
3. 　　求解子问题：$$x_{k+1} = \arg\min_x L_A(x, \lambda_k, \rho_k)$$
4. 　　**if** $$\|h(x_{k+1})\| \leq \eta_k$$ **then**
5. 　　　　更新乘子：$$\lambda_{k+1} = \lambda_k + \rho_k h(x_{k+1})$$
6. 　　　　更新容差：$$\eta_{k+1} = \sigma \eta_k$$
7. 　　**else**
8. 　　　　$$\lambda_{k+1} = \lambda_k$$
9. 　　　　增加罚参数：$$\rho_{k+1} = \beta \rho_k$$
10. 　　**end if**
11. 　　$$k = k + 1$$
12. **until** 收敛

#### 4.3.2 乘子更新的理论依据

乘子更新公式 $$\lambda_{k+1} = \lambda_k + \rho_k h(x_{k+1})$$ 来自于：

如果 $$x_{k+1}$$ 是 $$L_A(x, \lambda_k, \rho_k)$$ 的最优解，则：
$$\nabla_x L_A(x_{k+1}, \lambda_k, \rho_k) = 0$$

即：
$$\nabla f(x_{k+1}) + \nabla h(x_{k+1})^T (\lambda_k + \rho_k h(x_{k+1})) = 0$$

这表明 $$\lambda_k + \rho_k h(x_{k+1})$$ 是拉格朗日乘子的更好估计。

### 4.4 收敛性分析

#### 4.4.1 全局收敛性

**定理3**：在适当假设下，ALM算法全局收敛到KKT点。

**主要假设**：
1. 可行域非空且紧
2. 函数连续可微
3. 约束规范条件成立

#### 4.4.2 收敛速率

**定理4**：如果严格二阶充分条件成立，ALM算法局部超线性收敛。

**关键条件**：
- 最优解处的约束规范条件
- 严格互补条件
- 二阶充分条件

---

## 五、不等式约束的增广拉格朗日方法

### 5.1 不等式约束的处理策略

对于不等式约束 $$g_j(x) \leq 0$$，有两种主要处理方式：

#### 5.1.1 直接处理法

使用修正的增广拉格朗日函数：
$$L_A(x, \mu, \rho) = f(x) + \frac{1}{2\rho} \sum_{j=1}^p \left[\max(0, \mu_j + \rho g_j(x))^2 - \mu_j^2\right]$$

#### 5.1.2 松弛变量法

引入松弛变量 $$s_j \geq 0$$，将不等式约束转化为等式约束：
$$g_j(x) + s_j^2 = 0$$

然后应用标准的增广拉格朗日方法。

### 5.2 乘子更新规则

对于不等式约束，乘子更新为：
$$\mu_{j,k+1} = \max(0, \mu_{j,k} + \rho_k g_j(x_{k+1}))$$

这确保了 $$\mu_{j,k+1} \geq 0$$，满足KKT条件的对偶可行性。

### 5.3 算法实现

**算法6：不等式约束的ALM**

1. **初始化**：$$x_0$$，$$\mu_0 \geq 0$$，$$\rho_0 > 0$$
2. **for** $$k = 0, 1, 2, \ldots$$ **do**
3. 　　求解：$$x_{k+1} = \arg\min_x L_A(x, \mu_k, \rho_k)$$
4. 　　更新乘子：$$\mu_{j,k+1} = \max(0, \mu_{j,k} + \rho_k g_j(x_{k+1}))$$
5. 　　**if** 约束违反度足够小 **then**
6. 　　　　保持罚参数：$$\rho_{k+1} = \rho_k$$
7. 　　**else**
8. 　　　　增加罚参数：$$\rho_{k+1} = \beta \rho_k$$
9. 　　**end if**
10. **end for**

---

## 六、交替方向乘子法 (ADMM)

### 6.1 ADMM的问题设定

考虑可分离的约束优化问题：
$$\begin{align}
\min \quad & f(x) + g(z) \\
\text{s.t.} \quad & Ax + Bz = c
\end{align}$$

其中 $$f$$ 和 $$g$$ 是凸函数，$$A$$，$$B$$ 是矩阵。

### 6.2 ADMM算法推导

#### 6.2.1 增广拉格朗日函数

$$L_\rho(x, z, y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2}\|Ax + Bz - c\|^2$$

#### 6.2.2 交替最小化

ADMM通过交替最小化来求解：

**算法7：ADMM算法**

1. **初始化**：$$x^0$$，$$z^0$$，$$y^0$$，$$\rho > 0$$
2. **repeat**
3. 　　$$x^{k+1} = \arg\min_x L_\rho(x, z^k, y^k)$$
4. 　　$$z^{k+1} = \arg\min_z L_\rho(x^{k+1}, z, y^k)$$
5. 　　$$y^{k+1} = y^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$$
6. **until** 收敛

### 6.3 ADMM的收敛性质

#### 6.3.1 收敛性定理

**定理5**：在适当假设下，ADMM算法收敛到最优解。

**主要假设**：
1. $$f$$ 和 $$g$$ 是凸函数
2. 增广拉格朗日函数有鞍点
3. $$\rho > 0$$

#### 6.3.2 收敛速率

- **一般情况**：$$O(1/k)$$ 收敛率
- **强凸情况**：线性收敛率

### 6.4 ADMM的应用

#### 6.4.1 LASSO问题

$$\min \frac{1}{2}\|Ax - b\|^2 + \lambda \|x\|_1$$

可以重写为：
$$\min \frac{1}{2}\|Ax - b\|^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad x - z = 0$$

#### 6.4.2 支持向量机

$$\min \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

通过适当的变量分离，可以用ADMM高效求解。

---

## 七、数值实现与计算技巧

### 7.1 子问题的求解

#### 7.1.1 无约束优化子问题

在ALM中，需要求解：
$$\min_x L_A(x, \lambda_k, \rho_k)$$

**求解方法**：
1. **Newton法**：适用于小规模问题
2. **拟Newton法**：L-BFGS等，适用于大规模问题
3. **共轭梯度法**：适用于二次子问题

#### 7.1.2 近似求解策略

实际中不需要精确求解子问题，只需：
$$\|\nabla_x L_A(x_{k+1}, \lambda_k, \rho_k)\| \leq \eta_k$$

其中 $$\eta_k$$ 是递减的容差序列。

### 7.2 罚参数的选择与更新

#### 7.2.1 初始罚参数

**经验法则**：
$$\rho_0 = \frac{\|\nabla f(x_0)\|}{\|h(x_0)\|}$$

这确保目标函数项和约束项的量级相当。

#### 7.2.2 自适应更新策略

```python
def update_penalty_parameter(rho, h_norm, h_norm_prev, beta=10):
    """自适应更新罚参数"""
    if h_norm > 0.9 * h_norm_prev:
        # 约束违反度下降不够，增加罚参数
        return beta * rho
    else:
        # 约束违反度下降良好，保持罚参数
        return rho
```

### 7.3 数值稳定性考虑

#### 7.3.1 病态问题的处理

当 $$\rho$$ 很大时，增广拉格朗日函数可能变得病态：

**解决方案**：
1. **预条件技术**：使用适当的预条件子
2. **尺度化**：对变量和约束进行适当的尺度化
3. **正则化**：添加小的正则化项

#### 7.3.2 收敛判据

使用多重收敛判据：
1. **KKT条件残差**：$$\|\nabla L(x, \lambda)\| \leq \epsilon_1$$
2. **约束违反度**：$$\|h(x)\| \leq \epsilon_2$$
3. **变量变化**：$$\|x_{k+1} - x_k\| \leq \epsilon_3$$

---

## 八、详细算例分析

### 8.1 等式约束的二次规划

**例题1**：求解
$$\begin{align}
\min \quad & \frac{1}{2}(x_1^2 + x_2^2) \\
\text{s.t.} \quad & x_1 + x_2 - 2 = 0
\end{align}$$

**问题分析**：这是一个简单的二次规划问题，目标函数是凸的二次型，约束是线性等式约束。几何上，我们要在直线 $$x_1 + x_2 = 2$$ 上找到距离原点最近的点。

#### 8.1.1 拉格朗日乘子法求解

**步骤1**：构造拉格朗日函数

根据拉格朗日乘子法，对于等式约束问题，我们构造：
$$L(x_1, x_2, \lambda) = f(x_1, x_2) + \lambda \cdot h(x_1, x_2)$$

其中 $$f(x_1, x_2) = \frac{1}{2}(x_1^2 + x_2^2)$$ 是目标函数，$$h(x_1, x_2) = x_1 + x_2 - 2$$ 是约束函数。

因此拉格朗日函数为：
$$L(x_1, x_2, \lambda) = \frac{1}{2}(x_1^2 + x_2^2) + \lambda(x_1 + x_2 - 2)$$

**步骤2**：建立KKT条件

最优解必须满足KKT条件，即拉格朗日函数对所有变量的偏导数都为零：

对 $$x_1$$ 求偏导：
$$\frac{\partial L}{\partial x_1} = \frac{\partial}{\partial x_1}\left[\frac{1}{2}(x_1^2 + x_2^2) + \lambda(x_1 + x_2 - 2)\right] = x_1 + \lambda = 0$$

对 $$x_2$$ 求偏导：
$$\frac{\partial L}{\partial x_2} = \frac{\partial}{\partial x_2}\left[\frac{1}{2}(x_1^2 + x_2^2) + \lambda(x_1 + x_2 - 2)\right] = x_2 + \lambda = 0$$

对 $$\lambda$$ 求偏导（确保约束满足）：
$$\frac{\partial L}{\partial \lambda} = x_1 + x_2 - 2 = 0$$

得到KKT方程组：
$$\begin{cases}
x_1 + \lambda = 0 \quad \text{（目标函数梯度与约束梯度平衡）} \\
x_2 + \lambda = 0 \quad \text{（目标函数梯度与约束梯度平衡）} \\
x_1 + x_2 - 2 = 0 \quad \text{（约束条件必须满足）}
\end{cases}$$

**步骤3**：求解方程组

从前两个方程可以看出：
$$x_1 = -\lambda \quad \text{和} \quad x_2 = -\lambda$$

这说明 $$x_1 = x_2$$，即最优解具有对称性。

将 $$x_1 = x_2 = -\lambda$$ 代入第三个方程：
$$(-\lambda) + (-\lambda) - 2 = 0$$
$$-2\lambda - 2 = 0$$
$$\lambda = -1$$

因此：$$x_1^* = x_2^* = -(-1) = 1$$

**解的验证**：
- **约束满足**：$$x_1^* + x_2^* = 1 + 1 = 2$$ ✓
- **拉格朗日乘子意义**：$$\lambda^* = -1 < 0$$ 表示约束是"紧的"，如果约束右端增加1个单位，目标函数值会减少1个单位

**最优解**：$$x_1^* = x_2^* = 1$$，$$\lambda^* = -1$$，最优目标函数值为 $$f^* = \frac{1}{2}(1^2 + 1^2) = 1$$

#### 8.1.2 增广拉格朗日法求解

**步骤1**：构造增广拉格朗日函数

增广拉格朗日函数在经典拉格朗日函数基础上添加了二次罚项：
$$L_A(x_1, x_2, \lambda, \rho) = \underbrace{\frac{1}{2}(x_1^2 + x_2^2)}_{\text{原目标函数}} + \underbrace{\lambda(x_1 + x_2 - 2)}_{\text{拉格朗日项}} + \underbrace{\frac{\rho}{2}(x_1 + x_2 - 2)^2}_{\text{二次罚项}}$$

其中：
- $$\rho > 0$$ 是罚参数，控制约束违反的惩罚程度
- 二次罚项 $$\frac{\rho}{2}(x_1 + x_2 - 2)^2$$ 确保即使 $$\lambda$$ 估计不准确，约束违反也会受到惩罚

**步骤2**：ALM子问题求解

在每次迭代中，我们需要求解无约束子问题：
$$\min_{x_1, x_2} L_A(x_1, x_2, \lambda_k, \rho_k)$$

对增广拉格朗日函数求梯度：
$$\frac{\partial L_A}{\partial x_1} = x_1 + \lambda_k + \rho_k(x_1 + x_2 - 2) = 0$$
$$\frac{\partial L_A}{\partial x_2} = x_2 + \lambda_k + \rho_k(x_1 + x_2 - 2) = 0$$

整理得到：
$$x_1 + \lambda_k + \rho_k(x_1 + x_2 - 2) = 0 \quad \Rightarrow \quad x_1(1 + \rho_k) + \rho_k x_2 = 2\rho_k - \lambda_k$$
$$x_2 + \lambda_k + \rho_k(x_1 + x_2 - 2) = 0 \quad \Rightarrow \quad \rho_k x_1 + x_2(1 + \rho_k) = 2\rho_k - \lambda_k$$

由于系统的对称性，显然 $$x_1 = x_2$$。设 $$x_1 = x_2 = x$$，则：
$$x(1 + \rho_k) + \rho_k x = 2\rho_k - \lambda_k$$
$$x(1 + 2\rho_k) = 2\rho_k - \lambda_k$$
$$x = \frac{2\rho_k - \lambda_k}{1 + 2\rho_k}$$

**步骤3**：乘子更新

ALM的乘子更新公式为：
$$\lambda_{k+1} = \lambda_k + \rho_k \cdot h(x_{k+1})$$

其中 $$h(x_{k+1}) = x_1^{(k+1)} + x_2^{(k+1)} - 2$$ 是约束违反度。

<div style="font-size: 0.85em;">

#### ALM迭代过程详细表

| k | $$\lambda_k$$ | $$\rho_k$$ | 子问题解 $$x_k = \frac{2\rho_k - \lambda_k}{1 + 2\rho_k}$$ | $$h(x_k) = 2x_k - 2$$ | $$\lambda_{k+1} = \lambda_k + \rho_k h(x_k)$$ |
|---|---------------|------------|----------------------------------------------------------|----------------------|------------------------------------------------|
| 0 | 0 | 1 | $$x = \frac{2 \cdot 1 - 0}{1 + 2 \cdot 1} = \frac{2}{3} \approx 0.67$$ | $$2 \cdot \frac{2}{3} - 2 = -\frac{2}{3}$$ | $$0 + 1 \cdot (-\frac{2}{3}) = -\frac{2}{3}$$ |
| 1 | $$-\frac{2}{3}$$ | 1 | $$x = \frac{2 - (-\frac{2}{3})}{3} = \frac{8/3}{3} = \frac{8}{9} \approx 0.89$$ | $$2 \cdot \frac{8}{9} - 2 = -\frac{2}{9}$$ | $$-\frac{2}{3} + 1 \cdot (-\frac{2}{9}) = -\frac{8}{9}$$ |
| 2 | $$-\frac{8}{9}$$ | 1 | $$x = \frac{2 + \frac{8}{9}}{3} = \frac{26/9}{3} = \frac{26}{27} \approx 0.96$$ | $$2 \cdot \frac{26}{27} - 2 = -\frac{2}{27}$$ | $$-\frac{8}{9} + 1 \cdot (-\frac{2}{27}) = -\frac{26}{27}$$ |

</div>

**收敛分析**：
- **约束违反度**快速减少：$$-\frac{2}{3} \to -\frac{2}{9} \to -\frac{2}{27} \to \cdots$$
- **乘子收敛**：$$\lambda_k \to -1$$（真实的拉格朗日乘子）
- **解收敛**：$$x_k \to 1$$（真实的最优解）

ALM的优势在于即使初始乘子估计 $$\lambda_0 = 0$$ 不准确，算法仍能快速收敛到正确解。

### 8.2 不等式约束问题

**例题2**：求解
$$\begin{align}
\min \quad & x_1^2 + x_2^2 \\
\text{s.t.} \quad & x_1 + x_2 - 1 \geq 0 \\
& x_1, x_2 \geq 0
\end{align}$$

#### 8.2.1 KKT条件分析

标准形式：
$$\begin{align}
\min \quad & x_1^2 + x_2^2 \\
\text{s.t.} \quad & -x_1 - x_2 + 1 \leq 0 \\
& -x_1 \leq 0, \quad -x_2 \leq 0
\end{align}$$

**KKT条件**：
$$\begin{cases}
2x_1 - \mu_1 - \mu_2 = 0 \\
2x_2 - \mu_1 - \mu_3 = 0 \\
-x_1 - x_2 + 1 \leq 0, \quad -x_1 \leq 0, \quad -x_2 \leq 0 \\
\mu_1, \mu_2, \mu_3 \geq 0 \\
\mu_1(-x_1 - x_2 + 1) = 0, \quad \mu_2(-x_1) = 0, \quad \mu_3(-x_2) = 0
\end{cases}$$

#### 8.2.2 情况分析

**情况1**：内点解（所有不等式严格满足）
如果 $$x_1, x_2 > 0$$ 且 $$x_1 + x_2 > 1$$，则 $$\mu_1 = \mu_2 = \mu_3 = 0$$
从KKT条件：$$2x_1 = 0$$，$$2x_2 = 0$$，得 $$x_1 = x_2 = 0$$
但这与 $$x_1 + x_2 > 1$$ 矛盾。

**情况2**：边界解
约束 $$x_1 + x_2 = 1$$ 激活，$$x_1, x_2 > 0$$
则 $$\mu_2 = \mu_3 = 0$$，$$\mu_1 > 0$$
KKT条件给出：$$2x_1 = \mu_1$$，$$2x_2 = \mu_1$$
所以 $$x_1 = x_2$$，结合 $$x_1 + x_2 = 1$$，得 $$x_1 = x_2 = 0.5$$

**验证**：$$\mu_1 = 2 \times 0.5 = 1 > 0$$ ?

**最优解**：$$x^* = (0.5, 0.5)^T$$，目标函数值 $$f^* = 0.5$$

### 8.3 ADMM算法实例

**例题3**：LASSO问题
$$\min \frac{1}{2}\|Ax - b\|^2 + \lambda \|x\|_1$$

**问题分析**：LASSO问题结合了最小二乘项（保证数据拟合）和L1正则项（促进稀疏性）。直接求解困难，因为L1范数不可微。ADMM通过变量分离巧妙地处理了这个问题。

#### 8.3.1 ADMM分解

**步骤1**：变量分离

引入辅助变量 $$z$$，将原问题重写为：
$$\min \frac{1}{2}\|Ax - b\|^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad x - z = 0$$

这样我们将问题分解为：
- $$f(x) = \frac{1}{2}\|Ax - b\|^2$$（可微的二次项）
- $$g(z) = \lambda \|z\|_1$$（不可微但有简单的近似算子）
- 约束：$$x - z = 0$$

**步骤2**：构造增广拉格朗日函数

$$L_\rho(x, z, y) = \underbrace{\frac{1}{2}\|Ax - b\|^2}_{\text{数据拟合项}} + \underbrace{\lambda \|z\|_1}_{\text{稀疏正则项}} + \underbrace{y^T(x - z)}_{\text{拉格朗日项}} + \underbrace{\frac{\rho}{2}\|x - z\|^2}_{\text{二次罚项}}$$

其中：
- $$y$$ 是拉格朗日乘子向量
- $$\rho > 0$$ 是罚参数

#### 8.3.2 ADMM更新推导

ADMM通过交替最小化来求解：

**x-更新**（解析解）：

求解子问题：
$$x^{k+1} = \arg\min_x \left[\frac{1}{2}\|Ax - b\|^2 + y^{kT}x + \frac{\rho}{2}\|x - z^k\|^2\right]$$

展开目标函数：
$$\frac{1}{2}\|Ax - b\|^2 + y^{kT}x + \frac{\rho}{2}\|x - z^k\|^2$$
$$= \frac{1}{2}(Ax - b)^T(Ax - b) + y^{kT}x + \frac{\rho}{2}(x - z^k)^T(x - z^k)$$
$$= \frac{1}{2}x^TA^TAx - b^TAx + \frac{1}{2}b^Tb + y^{kT}x + \frac{\rho}{2}x^Tx - \rho x^Tz^k + \frac{\rho}{2}(z^k)^Tz^k$$

对 $$x$$ 求导并令其为零：
$$A^TAx - A^Tb + y^k + \rho x - \rho z^k = 0$$
$$(A^TA + \rho I)x = A^Tb + \rho z^k - y^k$$

因此：
$$\boxed{x^{k+1} = (A^TA + \rho I)^{-1}(A^Tb + \rho z^k - y^k)}$$

**z-更新**（软阈值算子）：

求解子问题：
$$z^{k+1} = \arg\min_z \left[\lambda \|z\|_1 + \frac{\rho}{2}\|x^{k+1} - z\|^2\right]$$

这等价于：
$$z^{k+1} = \arg\min_z \left[\lambda \|z\|_1 + \frac{\rho}{2}\left\|z - \left(x^{k+1} + \frac{y^k}{\rho}\right)\right\|^2\right]$$

这是标准的**软阈值问题**，解为：
$$\boxed{z^{k+1} = S_{\lambda/\rho}\left(x^{k+1} + \frac{y^k}{\rho}\right)}$$

其中软阈值算子定义为：
$$S_\kappa(a) = \begin{cases}
a - \kappa, & \text{if } a > \kappa \\
0, & \text{if } |a| \leq \kappa \\
a + \kappa, & \text{if } a < -\kappa
\end{cases} = \text{sign}(a) \max(|a| - \kappa, 0)$$

**y-更新**（对偶变量更新）：

$$\boxed{y^{k+1} = y^k + \rho(x^{k+1} - z^{k+1})}$$

这个更新确保了约束 $$x - z = 0$$ 逐步得到满足。

#### 8.3.3 算法收敛分析

**收敛条件**：
1. **原始残差**：$$r^k = x^k - z^k \to 0$$
2. **对偶残差**：$$s^k = \rho(z^k - z^{k-1}) \to 0$$

**停止准则**：
$$\|r^k\| \leq \epsilon_{\text{pri}} \quad \text{and} \quad \|s^k\| \leq \epsilon_{\text{dual}}$$

其中 $$\epsilon_{\text{pri}}, \epsilon_{\text{dual}}$$ 是预设的容差。

---

## 九、现代发展与应用

### 9.1 分布式优化中的ADMM

#### 9.1.1 共识问题

考虑分布式优化：
$$\min \sum_{i=1}^N f_i(x_i) \quad \text{s.t.} \quad x_i = z, \quad i = 1, \ldots, N$$

ADMM更新：
- **局部更新**：$$x_i^{k+1} = \arg\min_{x_i} \left[f_i(x_i) + \frac{\rho}{2}\|x_i - z^k + u_i^k\|^2\right]$$
- **全局更新**：$$z^{k+1} = \frac{1}{N}\sum_{i=1}^N (x_i^{k+1} + u_i^k)$$
- **对偶更新**：$$u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}$$

#### 9.1.2 通信效率

ADMM在分布式优化中的优势：
1. **局部计算**：每个节点只需求解自己的子问题
2. **简单通信**：只需传输平均值信息
3. **异步实现**：可以容忍节点间的异步性

### 9.2 机器学习中的应用

#### 9.2.1 支持向量机

对偶SVM问题：
$$\max \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

使用ALM求解可以处理大规模数据集。

#### 9.2.2 深度学习中的约束

在神经网络训练中，经常需要处理各种约束：
- **权重约束**：$$\|W\|_F \leq C$$
- **稀疏约束**：$$\|W\|_1 \leq S$$
- **正交约束**：$$W^TW = I$$

ALM和ADMM提供了统一的处理框架。

### 9.3 大规模优化的挑战

#### 9.3.1 内存效率

对于超大规模问题：
- **随机ALM**：使用随机梯度估计
- **分块ADMM**：将变量分块处理
- **低秩近似**：利用问题的低秩结构

#### 9.3.2 并行计算

现代ALM/ADMM实现：
- **GPU加速**：利用GPU的并行计算能力
- **分布式计算**：多机器协同求解
- **异步算法**：减少同步开销

---

## 十、算法比较与选择指南

### 10.1 方法比较

<div style="font-size: 0.85em;">

#### 约束优化算法性能对比

| 方法 | 收敛速度 | 内存需求 | 实现复杂度 | 适用问题 | 全局收敛性 |
|------|----------|----------|------------|----------|------------|
| Newton-Lagrange | 二次收敛 | $$O(n^2)$$ | 高 | 小规模等式约束 | 局部 |
| SQP | 超线性收敛 | $$O(n^2)$$ | 高 | 一般约束问题 | 局部 |
| 外罚函数法 | 线性收敛 | $$O(n)$$ | 低 | 一般约束问题 | 全局 |
| 障碍函数法 | 线性收敛 | $$O(n)$$ | 中等 | 不等式约束 | 局部 |
| ALM | 超线性收敛 | $$O(n)$$ | 中等 | 一般约束问题 | 全局 |
| ADMM | 线性收敛 | $$O(n)$$ | 低 | 可分离问题 | 全局 |

</div>

### 10.2 算法选择指南

#### 10.2.1 根据问题特征选择

1. **小规模问题** ($$n < 100$$)
   - 等式约束：Newton-Lagrange法
   - 一般约束：SQP法

2. **中等规模问题** ($$100 \leq n \leq 10^4$$)
   - 光滑约束：增广拉格朗日法
   - 非光滑约束：ADMM

3. **大规模问题** ($$n > 10^4$$)
   - 可分离结构：ADMM
   - 一般结构：随机ALM

#### 10.2.2 根据约束类型选择

1. **等式约束为主**：ALM或Newton-Lagrange
2. **不等式约束为主**：障碍函数法或ALM
3. **混合约束**：ALM或SQP
4. **可分离约束**：ADMM

### 10.3 实现建议

#### 10.3.1 参数调优

1. **罚参数初值**：$$\rho_0 = 1 \sim 10$$
2. **增长因子**：$$\beta = 2 \sim 10$$
3. **容差序列**：$$\eta_k = 0.1/k$$
4. **收敛容差**：$$\epsilon = 10^{-6} \sim 10^{-8}$$

#### 10.3.2 数值稳定性

1. **尺度化**：确保变量和约束的量级相当
2. **预条件**：使用适当的预条件子
3. **正则化**：添加小的正则化项防止奇异性

---

## 总结

本文系统地介绍了拉格朗日乘子法和增广拉格朗日乘子法的理论基础与实际应用：

### 主要内容回顾

1. **经典拉格朗日乘子法**：
   - 从几何直观出发，推导了KKT条件
   - 介绍了Newton-Lagrange法和SQP算法
   - 分析了约束规范条件的重要性

2. **罚函数方法**：
   - 外罚函数法：简单但可能病态
   - 障碍函数法：保持可行性但需要可行初始点

3. **增广拉格朗日方法**：
   - 结合了拉格朗日函数和罚函数的优点
   - 具有良好的收敛性质和数值稳定性
   - 是现代约束优化的主流方法

4. **ADMM算法**：
   - 适用于可分离的约束优化问题
   - 在机器学习和分布式优化中有广泛应用
   - 具有良好的并行性和收敛性

### 理论贡献

1. **统一框架**：将各种方法置于统一的理论框架下
2. **收敛分析**：详细分析了各算法的收敛性质
3. **实用指导**：提供了算法选择和参数调优的指南

### 实践价值

增广拉格朗日方法在现代优化中仍然是核心方法，特别是在：
- **工程优化**：结构优化、控制系统设计
- **机器学习**：支持向量机、深度学习约束
- **分布式计算**：大规模优化问题的分解求解

<div style="text-align: center; margin: 30px 0;">
  <a href="/2025/09/16/lagrange-multipliers-examples/" class="btn btn-primary btn-lg" style="display: inline-block; padding: 15px 30px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; border: none;">
    <i class="fa fa-calculator" style="margin-right: 10px;"></i>查看详细例题
  </a>
</div>

> **相关文章**:
> - [《KKT条件详解》](/2025/09/10/kkt-conditions/)
> - [《牛顿法与拟牛顿法详解》](/2025/09/13/newton-quasi-newton-methods/)
> - [《最优化理论基础》](/2025/09/08/optimization-fundamentals/)
> - [《最速下降法与共轭梯度法》](/2025/09/11/steepest-descent-and-conjugate-gradient/)

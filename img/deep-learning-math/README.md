# 深度学习数学基础 - 可视化图表说明

本文件夹包含了《深度学习的数学基础：反向传播、微积分与优化理论完全指南》中使用的所有可视化图表。

## 图表清单

### 1. **01_perceptron_vs_xor.png**
**位置**：基础 1 - 感知机与多层感知机
**说明**：
- 左图：展示线性可分问题，感知机可以用一条直线完美分离两类数据
- 右图：经典的 XOR 问题，展示为什么单层感知机无法解决，激发了深度学习的需求

**学习要点**：理解感知机的局限性，认识到多层网络的必要性

---

### 2. **02_activation_functions.png**
**位置**：基础 1 - 激活函数的非线性本质
**说明**：
- 第一行：Sigmoid、Tanh、ReLU 函数及其导数对比
- 第二行 (1-2)：Leaky ReLU 和 ELU 函数
- 第二行 (3)：所有激活函数的导数对比，直观显示梯度特性

**学习要点**：
- Sigmoid 和 Tanh 的导数最大值分别为 0.25 和 1.0
- ReLU 导数为 0 或 1（不衰减）
- 理解为什么 ReLU 比 Sigmoid 更适合深度网络

---

### 3. **03_gradient_vanishing.png**
**位置**：第二部分 - 梯度消失问题
**说明**：
- 左图：对数坐标显示梯度随网络深度的衰减，Sigmoid 呈指数衰减，ReLU 保持不变
- 右图：具体数值对比，展示 50 层网络中 Sigmoid 的梯度几近消失

**学习要点**：
- 梯度消失问题的数学本质：连乘效应
- 为什么 ReLU 解决了深度网络中的梯度消失问题
- 0.25^50 ≈ 10^-30 的实际意义

---

### 4. **04_loss_functions.png**
**位置**：基础 1 - 损失函数的选择与设计
**说明**：
- 第一行左：MSE 和 CrossEntropy 损失函数对比
- 第一行右：两种损失函数的梯度对比
- 第二行左：两个不同场景下的损失值对比（接近正确 vs 完全错误）
- 第二行右：训练过程中的收敛速度对比

**学习要点**：
- CrossEntropy 的梯度更清晰且稳定
- MSE 在预测完全错误时梯度可能太小（被 Sigmoid 压制）
- CrossEntropy 是分类任务的最佳选择

---

### 5. **05_learning_rate_effect.png**
**位置**：基础 1 - 优化算法基础
**说明**：
- 左上：学习率太小（0.001），收敛极慢
- 右上：学习率太大（0.1），振荡甚至发散
- 左下：最优学习率（0.01），平稳快速收敛
- 右下：三种学习率的直接对比

**学习要点**：
- 学习率的选择对训练至关重要
- 太小：浪费时间；太大：错过最小值
- 实践中通常需要通过验证集来调整学习率

---

### 6. **06_sgd_convergence.png**
**位置**：基础 1 - 反向传播算法详解（作为 SGD vs Full Batch 的对比）
**说明**：
- 左上：全批梯度下降的轨迹，平稳但需要所有数据
- 右上：SGD 的轨迹，快速但有噪声
- 左下：损失曲线对比
- 右下：详细的方法对比表

**学习要点**：
- Full Batch GD：收敛平稳但慢
- SGD：快速但嘈杂
- Mini-Batch SGD：结合两者优点，是现代深度学习的标准

---

### 7. **07_mlp_architecture.png**
**位置**：基础 1 - 多层感知机（MLP）
**说明**：
- 可视化完整的 MLP 网络结构
- 显示输入层、两个隐藏层和输出层的连接
- 标注所有前向传播公式
- 统计参数数量（总计 64 个参数）

**学习要点**：
- MLP 的通用架构
- 参数数量的计算方法
- 理解前向传播的完整流程

---

## 使用建议

### 教学用途
1. **从上到下的学习路径**：
   - 先看 01_perceptron_vs_xor.png 理解为什么需要深度学习
   - 再看 02_activation_functions.png 理解激活函数的作用
   - 然后看 03_gradient_vanishing.png 理解深度学习的困难
   - 看 04_loss_functions.png 选择合适的损失函数
   - 看 05_learning_rate_effect.png 理解参数调优
   - 看 06_sgd_convergence.png 理解优化算法
   - 最后看 07_mlp_architecture.png 理解完整的网络

2. **分主题学习**：
   - 激活函数：02, 03
   - 损失函数：04
   - 优化算法：05, 06
   - 网络架构：07, 01

### 演讲/报告使用
所有图表高分辨率 (300 DPI)，适合在 PowerPoint、Keynote 等演讲软件中直接使用。

### 论文/文档
可直接引用这些图表，建议配合文章中的详细数学推导使用。

---

## 图表生成方法

所有图表都通过 Python 脚本生成：
- `generate_visualizations_en.py`：主要生成脚本
- 依赖库：numpy, matplotlib, scipy

### 重新生成图表
```bash
cd img/deep-learning-math/
MPLBACKEND=Agg python3 generate_visualizations_en.py
```

---

## 相关论文/参考

- Goodfellow et al., "Deep Learning" (2016)
- LeCun et al., "Gradient-based learning applied to document recognition" (1998) - 反向传播的经典论文
- Rumelhart et al., "Learning representations by back-propagating errors" (1986) - 反向传播的开创性论文
- He et al., "Delving Deep into Rectifiers" (2015) - ReLU 和权重初始化
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) - Adam 优化器

---

## 许可证

这些图表是为教育和学习目的创建的。可自由使用和修改。

---

**最后更新**：2026-01-24

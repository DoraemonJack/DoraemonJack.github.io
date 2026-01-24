#!/usr/bin/env python3
"""
生成深度学习数学基础的可视化图表 (中文版本)
"""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# 确保保存路径
import os
save_dir = os.path.dirname(os.path.abspath(__file__))

def create_perceptron_vs_xor():
    """
    1. 感知机与XOR问题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：线性可分
    np.random.seed(42)
    X_class0 = np.random.randn(50, 2) + np.array([-1, -1])
    X_class1 = np.random.randn(50, 2) + np.array([1, 1])
    
    ax1.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='类别0', s=50, alpha=0.7)
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='类别1', s=50, alpha=0.7)
    
    # 决策边界
    x_range = np.linspace(-4, 4, 100)
    y_boundary = x_range
    ax1.plot(x_range, y_boundary, 'k--', linewidth=2, label='决策边界')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('特征1', fontsize=11)
    ax1.set_ylabel('特征2', fontsize=11)
    ax1.set_title('线性可分（感知机有效）', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：XOR问题
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    colors = ['blue' if y == 0 else 'red' for y in y_xor]
    ax2.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, (x, y) in enumerate(X_xor):
        ax2.text(x+0.05, y+0.05, f'({x},{y})\n→{y_xor[i]}', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlabel('特征1', fontsize=11)
    ax2.set_ylabel('特征2', fontsize=11)
    ax2.set_title('XOR问题（感知机失效）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/01_perceptron_vs_xor.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 01_perceptron_vs_xor.png")
    plt.close()


def create_activation_functions():
    """
    2. 激活函数对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    x = np.linspace(-5, 5, 1000)
    
    # 1. Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2, label='sigmoid(x)')
    axes[0, 0].plot(x, sigmoid_deriv, 'r--', linewidth=2, label="sigmoid'(x)")
    axes[0, 0].axhline(y=0.25, color='r', linestyle=':', alpha=0.5, label='最大导数 = 0.25')
    axes[0, 0].set_title('Sigmoid函数', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('输出', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-0.2, 1.2])
    
    # 2. Tanh
    tanh = np.tanh(x)
    tanh_deriv = 1 - tanh**2
    
    axes[0, 1].plot(x, tanh, 'g-', linewidth=2, label='tanh(x)')
    axes[0, 1].plot(x, tanh_deriv, 'r--', linewidth=2, label="tanh'(x)")
    axes[0, 1].axhline(y=1, color='r', linestyle=':', alpha=0.5, label='最大导数 = 1.0')
    axes[0, 1].set_title('Tanh函数', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ReLU
    relu = np.maximum(0, x)
    relu_deriv = (x > 0).astype(float)
    
    axes[0, 2].plot(x, relu, 'purple', linewidth=2, label='ReLU(x)')
    axes[0, 2].plot(x, relu_deriv, 'r--', linewidth=2, label="ReLU'(x)")
    axes[0, 2].set_title('ReLU函数', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylim([-0.2, 5.2])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Leaky ReLU
    alpha = 0.1
    leaky_relu = np.where(x > 0, x, alpha * x)
    leaky_relu_deriv = np.where(x > 0, 1, alpha)
    
    axes[1, 0].plot(x, leaky_relu, 'orange', linewidth=2, label='Leaky ReLU(x)')
    axes[1, 0].plot(x, leaky_relu_deriv, 'r--', linewidth=2, label="Leaky ReLU'(x)")
    axes[1, 0].set_title('Leaky ReLU (α=0.1)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('输入 (x)', fontsize=11)
    axes[1, 0].set_ylabel('输出', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. ELU
    alpha = 1.0
    elu = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elu_deriv = np.where(x > 0, 1, alpha * np.exp(x))
    
    axes[1, 1].plot(x, elu, 'brown', linewidth=2, label='ELU(x)')
    axes[1, 1].plot(x, elu_deriv, 'r--', linewidth=2, label="ELU'(x)")
    axes[1, 1].set_title('ELU函数', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('输入 (x)', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 导数对比
    axes[1, 2].plot(x, sigmoid_deriv, label="Sigmoid'", linewidth=2)
    axes[1, 2].plot(x, tanh_deriv, label="Tanh'", linewidth=2)
    axes[1, 2].plot(x, relu_deriv, label="ReLU'", linewidth=2)
    axes[1, 2].plot(x, leaky_relu_deriv, label="Leaky ReLU'", linewidth=2)
    axes[1, 2].set_title('激活函数导数对比', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('输入 (x)', fontsize=11)
    axes[1, 2].set_ylabel('导数', fontsize=11)
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1.2])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/02_activation_functions.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 02_activation_functions.png")
    plt.close()


def create_gradient_vanishing():
    """
    3. 梯度消失问题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = np.arange(1, 101)
    gradient_sigmoid = 0.25 ** layers
    gradient_relu = np.ones_like(layers)
    
    ax1.semilogy(layers, gradient_sigmoid, 'r-', linewidth=2.5, label="Sigmoid (最大导数=0.25)")
    ax1.semilogy(layers, gradient_relu, 'g-', linewidth=2.5, label="ReLU (导数=1)")
    ax1.axhline(y=1e-10, color='gray', linestyle='--', alpha=0.5, label='检测限')
    ax1.fill_between(layers, gradient_sigmoid, 1e-30, alpha=0.2, color='red')
    ax1.set_xlabel('网络深度', fontsize=11)
    ax1.set_ylabel('梯度大小', fontsize=11)
    ax1.set_title('梯度消失：随深度衰减', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim([1e-30, 1e1])
    
    depth_points = [10, 20, 30, 40, 50]
    grad_values_sigmoid = [0.25**d for d in depth_points]
    grad_values_relu = [1.0] * len(depth_points)
    
    x_pos = np.arange(len(depth_points))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, np.log10(grad_values_sigmoid), width, 
                     label='Sigmoid', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, np.log10(grad_values_relu), width, 
                     label='ReLU', color='green', alpha=0.7)
    
    ax2.set_xlabel('网络深度', fontsize=11)
    ax2.set_ylabel('Log10(梯度)', fontsize=11)
    ax2.set_title('不同深度下的梯度对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{d}层' for d in depth_points])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (v1, v2) in enumerate(zip(grad_values_sigmoid, grad_values_relu)):
        ax2.text(i - width/2, np.log10(v1) - 1, f'{v1:.0e}', 
                ha='center', va='top', fontsize=8, fontweight='bold')
        ax2.text(i + width/2, np.log10(v2) + 0.3, '1.0', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/03_gradient_vanishing.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 03_gradient_vanishing.png")
    plt.close()


def create_loss_functions():
    """
    4. 损失函数对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    y_true = 1.0
    y_pred = np.linspace(0.001, 0.999, 1000)
    
    mse_loss = (y_true - y_pred) ** 2
    bce_loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    mse_grad = -2 * (y_true - y_pred)
    bce_grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    # 损失函数
    axes[0, 0].plot(y_pred, mse_loss, 'b-', linewidth=2.5, label='MSE损失')
    axes[0, 0].plot(y_pred, bce_loss, 'r-', linewidth=2.5, label='交叉熵损失')
    axes[0, 0].axvline(x=y_true, color='gray', linestyle='--', alpha=0.5, label='真实标签')
    axes[0, 0].set_xlabel('预测概率', fontsize=11)
    axes[0, 0].set_ylabel('损失值', fontsize=11)
    axes[0, 0].set_title('分类任务中的损失函数', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 10])
    
    # 梯度
    axes[0, 1].plot(y_pred, mse_grad, 'b-', linewidth=2.5, label='MSE梯度')
    axes[0, 1].plot(y_pred, bce_grad, 'r-', linewidth=2.5, label='交叉熵梯度')
    axes[0, 1].axvline(x=y_true, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('预测概率', fontsize=11)
    axes[0, 1].set_ylabel('梯度大小', fontsize=11)
    axes[0, 1].set_title('损失函数梯度对比', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-5, 15])
    
    # 场景对比
    scenarios = ['接近正确\n(y=0.8)', '完全错误\n(y=0.1)']
    y_preds_eval = [0.8, 0.1]
    
    mse_vals = [(y_true - yp)**2 for yp in y_preds_eval]
    bce_vals = [-y_true * np.log(yp) - (1-y_true) * np.log(1-yp) for yp in y_preds_eval]
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x_pos - width/2, mse_vals, width, label='MSE', color='blue', alpha=0.7)
    bars2 = axes[1, 0].bar(x_pos + width/2, bce_vals, width, label='交叉熵', color='red', alpha=0.7)
    
    axes[1, 0].set_ylabel('损失值', fontsize=11)
    axes[1, 0].set_title('不同场景下的损失', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(scenarios, fontsize=10)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, (v1, v2) in enumerate(zip(mse_vals, bce_vals)):
        axes[1, 0].text(i - width/2, v1 + 0.05, f'{v1:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1, 0].text(i + width/2, v2 + 0.05, f'{v2:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 收敛速度
    iterations = np.arange(0, 100, 1)
    mse_learning = 0.5 * np.exp(-0.03 * iterations) + 0.05
    bce_learning = 0.5 * np.exp(-0.05 * iterations) + 0.05
    
    axes[1, 1].plot(iterations, mse_learning, 'b-', linewidth=2.5, label='MSE', marker='o', markersize=3, markevery=10)
    axes[1, 1].plot(iterations, bce_learning, 'r-', linewidth=2.5, label='交叉熵', marker='s', markersize=3, markevery=10)
    axes[1, 1].set_xlabel('训练迭代次数', fontsize=11)
    axes[1, 1].set_ylabel('损失值', fontsize=11)
    axes[1, 1].set_title('训练收敛速度', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/04_loss_functions.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 04_loss_functions.png")
    plt.close()


def create_learning_rate_effect():
    """
    5. 学习率的影响
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = np.arange(0, 200, 1)
    
    # 学习率很小
    loss_small_lr = 1.0 - 0.003 * iterations
    loss_small_lr = np.maximum(loss_small_lr, 0.01)
    
    # 学习率很大
    loss_large_lr = 0.5 * np.sin(0.05 * iterations) * np.exp(-0.001 * iterations) + 0.3
    loss_large_lr = np.abs(loss_large_lr)
    
    # 最优学习率
    loss_optimal = 0.8 * np.exp(-0.02 * iterations) + 0.05
    
    # 学习率很小
    axes[0, 0].plot(iterations, loss_small_lr, 'orange', linewidth=2.5)
    axes[0, 0].fill_between(iterations, loss_small_lr, alpha=0.2, color='orange')
    axes[0, 0].set_xlabel('训练迭代次数', fontsize=11)
    axes[0, 0].set_ylabel('损失值', fontsize=11)
    axes[0, 0].set_title('学习率过小（α=0.001）', fontsize=12, fontweight='bold', color='orange')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(100, 0.5, '收敛太慢！', fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 学习率很大
    axes[0, 1].plot(iterations, loss_large_lr, 'r', linewidth=2.5)
    axes[0, 1].fill_between(iterations, loss_large_lr, alpha=0.2, color='red')
    axes[0, 1].set_xlabel('训练迭代次数', fontsize=11)
    axes[0, 1].set_ylabel('损失值', fontsize=11)
    axes[0, 1].set_title('学习率过大（α=0.1）', fontsize=12, fontweight='bold', color='red')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(100, 0.5, '振荡/发散！', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 最优学习率
    axes[1, 0].plot(iterations, loss_optimal, 'g', linewidth=2.5)
    axes[1, 0].fill_between(iterations, loss_optimal, alpha=0.2, color='green')
    axes[1, 0].set_xlabel('训练迭代次数', fontsize=11)
    axes[1, 0].set_ylabel('损失值', fontsize=11)
    axes[1, 0].set_title('最优学习率（α=0.01）', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(100, 0.4, '平稳快速收敛！', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 对比
    axes[1, 1].plot(iterations, loss_small_lr, 'orange', linewidth=2.5, label='α=0.001 (过小)')
    axes[1, 1].plot(iterations, loss_large_lr, 'r', linewidth=2.5, label='α=0.1 (过大)')
    axes[1, 1].plot(iterations, loss_optimal, 'g', linewidth=2.5, label='α=0.01 (最优)')
    axes[1, 1].set_xlabel('训练迭代次数', fontsize=11)
    axes[1, 1].set_ylabel('损失值', fontsize=11)
    axes[1, 1].set_title('学习率对比', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/05_learning_rate_effect.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 05_learning_rate_effect.png")
    plt.close()


def create_sgd_convergence():
    """
    6. SGD与全批梯度下降对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.linspace(-5, 5, 100)
    y = x**2 + 2
    
    # ========== 全批梯度下降 ==========
    ax = axes[0, 0]
    X, Y = np.meshgrid(x, x)
    Z = Y**2 + X**2
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    
    theta = 4.0
    trajectory_full = [theta]
    for _ in range(30):
        grad = 4 * theta
        theta -= 0.1 * grad
        trajectory_full.append(theta)
    
    ax.plot(trajectory_full, [t**2 + 2 for t in trajectory_full], 'r-o', linewidth=2, markersize=6, label='全批GD')
    ax.set_xlabel('参数', fontsize=11)
    ax.set_ylabel('损失', fontsize=11)
    ax.set_title('全批梯度下降\n平稳但需要全部数据', fontsize=12, fontweight='bold')
    ax.legend()
    
    # ========== SGD ==========
    ax = axes[0, 1]
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    
    np.random.seed(42)
    theta = 4.0
    trajectory_sgd = [theta]
    for _ in range(30):
        grad = 4 * theta + np.random.randn() * 0.5
        theta -= 0.1 * grad
        trajectory_sgd.append(theta)
    
    ax.plot(trajectory_sgd, [t**2 + 2 for t in trajectory_sgd], 'b-o', linewidth=2, markersize=6, label='SGD')
    ax.set_xlabel('参数', fontsize=11)
    ax.set_ylabel('损失', fontsize=11)
    ax.set_title('随机梯度下降\n快速但含噪声', fontsize=12, fontweight='bold')
    ax.legend()
    
    # ========== 损失曲线 ==========
    ax = axes[1, 0]
    iterations = np.arange(len(trajectory_full))
    loss_full = [t**2 + 2 for t in trajectory_full]
    loss_sgd = [t**2 + 2 + np.random.randn() * 0.1 for t in trajectory_sgd]
    
    ax.plot(iterations, loss_full, 'r-o', linewidth=2, label='全批', markersize=4, markevery=3)
    ax.plot(iterations, loss_sgd, 'b-o', linewidth=2, label='SGD', markersize=4, markevery=3)
    ax.fill_between(iterations, loss_full, alpha=0.2, color='red')
    ax.fill_between(iterations, loss_sgd, alpha=0.2, color='blue')
    ax.set_xlabel('迭代次数', fontsize=11)
    ax.set_ylabel('损失', fontsize=11)
    ax.set_title('收敛曲线对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ========== 对比表 ==========
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = (
        '全批GD vs SGD\n\n'
        '─────────────────────────\n'
        '全批梯度下降\n'
        '─────────────────────────\n'
        '优点：收敛平稳\n'
        '      方向准确\n'
        '缺点：速度慢（需全数据）\n'
        '      内存占用大\n'
        '      易陷入局部最小值\n\n'
        '─────────────────────────\n'
        '随机梯度下降\n'
        '─────────────────────────\n'
        '优点：速度快（小批量）\n'
        '      内存节省\n'
        '      噪声帮助逃离局部最小值\n'
        '缺点：收敛含噪声\n'
        '      方向不稳定\n\n'
        '─────────────────────────\n'
        '推荐：小批量SGD\n'
        '（批量大小：32-256）'
    )
    
    ax.text(0.1, 0.95, comparison_text, fontsize=10, family='monospace',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/06_sgd_convergence.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 06_sgd_convergence.png")
    plt.close()


def create_mlp_architecture():
    """
    7. MLP架构
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, '多层感知机(MLP)架构详解', 
           fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    layers_info = [
        {'name': '输入层', 'x': 1, 'nodes': 4, 'color': 'lightblue'},
        {'name': '隐藏层1', 'x': 3, 'nodes': 5, 'color': 'lightgreen'},
        {'name': '隐藏层2', 'x': 5, 'nodes': 4, 'color': 'lightyellow'},
        {'name': '输出层', 'x': 7, 'nodes': 3, 'color': 'lightcoral'},
    ]
    
    # 绘制节点
    all_nodes = []
    for layer_idx, layer_info in enumerate(layers_info):
        x = layer_info['x']
        n_nodes = layer_info['nodes']
        nodes = []
        
        y_start = 4.5 - (n_nodes - 1) * 0.4
        
        for i in range(n_nodes):
            y = y_start + i * 0.8
            circle = patches.Circle((x, y), 0.2, facecolor=layer_info['color'], 
                                   edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            
            if layer_idx == 0:
                ax.text(x, y - 0.35, f'x{i+1}', fontsize=8, ha='center')
            elif layer_idx == len(layers_info) - 1:
                ax.text(x, y - 0.35, f'ŷ{i+1}', fontsize=8, ha='center')
            else:
                ax.text(x, y - 0.35, f'h{i+1}', fontsize=8, ha='center')
            
            nodes.append((x, y))
        
        all_nodes.append(nodes)
        
        ax.text(x, 8.5, layer_info['name'], fontsize=10, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=layer_info['color'], alpha=0.8))
    
    # 绘制连接
    for layer_idx in range(len(all_nodes) - 1):
        from_nodes = all_nodes[layer_idx]
        to_nodes = all_nodes[layer_idx + 1]
        
        for j, to_node in enumerate(to_nodes[:3]):
            n_connections = min(2, len(from_nodes))
            for i in range(n_connections):
                from_node = from_nodes[i]
                ax.plot([from_node[0] + 0.2, to_node[0] - 0.2], 
                       [from_node[1], to_node[1]], 
                       'gray', alpha=0.3, linewidth=1)
    
    # 公式
    formula_y = 2.0
    ax.text(1, formula_y, '前向传播：', fontsize=10, fontweight='bold')
    formulas = [
        r'$z^{(1)} = W^{(1)} \cdot x + b^{(1)}$',
        r'$h^{(1)} = \sigma(z^{(1)})$',
        r'$z^{(2)} = W^{(2)} \cdot h^{(1)} + b^{(2)}$',
        r'$h^{(2)} = \sigma(z^{(2)})$',
        r'$z^{(3)} = W^{(3)} \cdot h^{(2)} + b^{(3)}$',
        r'$\hat{y} = \sigma_{out}(z^{(3)})$',
    ]
    
    for i, formula in enumerate(formulas):
        ax.text(1, formula_y - 0.35 - i*0.3, formula, fontsize=9, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 参数数量
    params_text = (
        '参数数量：\n'
        'W(1): 4×5 = 20\n'
        'b(1): 5\n'
        'W(2): 5×4 = 20\n'
        'b(2): 4\n'
        'W(3): 4×3 = 12\n'
        'b(3): 3\n'
        '总计：64个参数'
    )
    ax.text(8.5, 4, params_text, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/07_mlp_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成: 07_mlp_architecture.png")
    plt.close()


if __name__ == '__main__':
    print("正在生成深度学习数学基础的可视化图表（中文版）...\n")
    
    create_perceptron_vs_xor()
    create_activation_functions()
    create_gradient_vanishing()
    create_loss_functions()
    create_learning_rate_effect()
    create_sgd_convergence()
    create_mlp_architecture()
    
    print("\n✅ 所有可视化图表生成完成！")
    print(f"📁 保存路径: {save_dir}/")

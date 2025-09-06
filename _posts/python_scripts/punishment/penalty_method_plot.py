#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
惩罚函数法可视化脚本
用于生成优化理论中惩罚函数法的图形演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def penalty_function_plot():
    """绘制惩罚函数法的基本概念图"""
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：目标函数和约束
    x = np.linspace(-2, 3, 1000)
    
    # 目标函数 f(x) = x^2
    f_x = x**2
    
    # 约束 g(x) = x - 1 >= 0
    g_x = x - 1
    
    # 惩罚函数 P(x) = max(0, -g(x))^2
    penalty = np.maximum(0, -g_x)**2
    
    # 不同惩罚参数下的惩罚函数
    r_values = [0.1, 0.5, 1.0, 2.0]
    colors = ['red', 'orange', 'green', 'blue']
    
    ax1.plot(x, f_x, 'k-', linewidth=2, label='f(x) = x²')
    ax1.plot(x, g_x, 'r--', linewidth=2, label='g(x) = x - 1')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.fill_between(x, 0, penalty, alpha=0.2, color='red', label='惩罚区域')
    
    for i, r in enumerate(r_values):
        phi_x = f_x + r * penalty
        ax1.plot(x, phi_x, color=colors[i], linewidth=1.5, 
                label=f'φ(x,r={r}) = f(x) + r·P(x)')
    
    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-1, 8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('函数值')
    ax1.set_title('惩罚函数法基本概念')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：可行域和最优解
    x1 = np.linspace(-1, 3, 100)
    x2 = np.linspace(-1, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 约束条件：x1 + x2 >= 1
    constraint = X1 + X2 - 1
    
    # 目标函数：f(x1,x2) = x1² + x2²
    objective = X1**2 + X2**2
    
    # 绘制等高线
    contour = ax2.contour(X1, X2, objective, levels=20, colors='blue', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # 绘制约束边界
    ax2.contour(X1, X2, constraint, levels=[0], colors='red', linewidths=2, label='g(x) = x₁ + x₂ - 1 = 0')
    
    # 填充可行域
    feasible_mask = constraint >= 0
    ax2.contourf(X1, X2, feasible_mask, levels=[0, 1], colors=['lightcoral'], alpha=0.3)
    
    # 标记最优解
    ax2.plot(0.5, 0.5, 'ro', markersize=8, label='最优解 (0.5, 0.5)')
    
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 3)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('约束优化问题')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def convergence_plot():
    """绘制惩罚函数法的收敛过程"""
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 模拟迭代过程
    iterations = np.arange(0, 20)
    
    # 惩罚参数序列
    r_k = 0.1 * (1.5 ** iterations)
    
    # 模拟解的变化（趋向于最优解）
    x_optimal = 0.5
    x_k = x_optimal + 0.5 * np.exp(-iterations * 0.3) * np.random.normal(0, 0.1, len(iterations))
    
    # 绘制收敛过程
    ax.plot(iterations, x_k, 'bo-', linewidth=2, markersize=6, label='迭代解 x^(k)')
    ax.axhline(y=x_optimal, color='red', linestyle='--', linewidth=2, label='最优解 x*')
    
    # 绘制惩罚参数变化
    ax2 = ax.twinx()
    ax2.semilogy(iterations, r_k, 'g^-', linewidth=2, markersize=6, label='惩罚参数 r^(k)')
    
    ax.set_xlabel('迭代次数 k')
    ax.set_ylabel('解 x^(k)', color='blue')
    ax2.set_ylabel('惩罚参数 r^(k)', color='green')
    ax.set_title('惩罚函数法收敛过程')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("生成惩罚函数法可视化图形...")
    
    # 生成基本概念图
    fig1 = penalty_function_plot()
    fig1.savefig('penalty_function_concept.png', dpi=300, bbox_inches='tight')
    print("已保存: penalty_function_concept.png")
    
    # 生成收敛过程图
    fig2 = convergence_plot()
    fig2.savefig('penalty_method_convergence.png', dpi=300, bbox_inches='tight')
    print("已保存: penalty_method_convergence.png")
    
    print("可视化完成！")

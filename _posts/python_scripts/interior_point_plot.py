#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内点法可视化脚本
用于生成优化理论中内点法的图形演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def interior_point_concept_plot():
    """绘制内点法的基本概念图"""
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：目标函数和约束
    x = np.linspace(-1, 2.5, 1000)
    
    # 目标函数 f(x) = x^2
    f_x = x**2
    
    # 约束 g(x) = x - 1 <= 0
    g_x = x - 1
    
    # 障碍函数 B(x) = -log(1-x) for x < 1
    x_feasible = x[x < 1]
    barrier = -np.log(1 - x_feasible)
    
    # 不同障碍参数下的障碍函数
    r_values = [0.5, 0.2, 0.1, 0.05]
    colors = ['red', 'orange', 'green', 'blue']
    
    ax1.plot(x, f_x, 'k-', linewidth=2, label='f(x) = x²')
    ax1.plot(x, g_x, 'r--', linewidth=2, label='g(x) = x - 1')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=1, color='red', linestyle=':', linewidth=2, label='约束边界 x=1')
    
    # 填充可行域
    feasible_x = x[x <= 1]
    feasible_f = f_x[x <= 1]
    ax1.fill_between(feasible_x, 0, feasible_f, alpha=0.2, color='green', label='可行域')
    
    for i, r in enumerate(r_values):
        phi_x = f_x[x < 1] + r * barrier
        ax1.plot(x_feasible, phi_x, color=colors[i], linewidth=1.5, 
                label=f'φ(x,r={r}) = f(x) + r·B(x)')
    
    ax1.set_xlim(-1, 2.5)
    ax1.set_ylim(-1, 8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('函数值')
    ax1.set_title('内点法基本概念')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：可行域和最优解
    x1 = np.linspace(-0.5, 2.5, 100)
    x2 = np.linspace(-0.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # 约束条件：x1 + x2 <= 2, x1 >= 0, x2 >= 0
    constraint1 = X1 + X2 - 2
    constraint2 = -X1
    constraint3 = -X2
    
    # 目标函数：f(x1,x2) = x1² + x2²
    objective = X1**2 + X2**2
    
    # 绘制等高线
    contour = ax2.contour(X1, X2, objective, levels=20, colors='blue', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # 绘制约束边界
    ax2.contour(X1, X2, constraint1, levels=[0], colors='red', linewidths=2, label='x₁ + x₂ = 2')
    ax2.contour(X1, X2, constraint2, levels=[0], colors='green', linewidths=2, label='x₁ = 0')
    ax2.contour(X1, X2, constraint3, levels=[0], colors='purple', linewidths=2, label='x₂ = 0')
    
    # 填充可行域
    feasible_mask = (constraint1 <= 0) & (constraint2 <= 0) & (constraint3 <= 0)
    ax2.contourf(X1, X2, feasible_mask, levels=[0, 1], colors=['lightgreen'], alpha=0.3)
    
    # 标记最优解
    ax2.plot(1, 1, 'ro', markersize=8, label='最优解 (1, 1)')
    
    # 绘制内点法搜索路径
    search_path_x = np.linspace(0.1, 0.9, 10)
    search_path_y = 2 - search_path_x
    ax2.plot(search_path_x, search_path_y, 'go-', markersize=4, linewidth=2, 
             label='内点法搜索路径', alpha=0.7)
    
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('约束优化问题（内点法）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def interior_point_convergence_plot():
    """绘制内点法的收敛过程"""
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 模拟迭代过程
    iterations = np.arange(0, 20)
    
    # 障碍参数序列（递减）
    r_k = 0.5 * (0.8 ** iterations)
    
    # 模拟解的变化（从可行域内部趋向边界）
    x_optimal = 1.0
    x_k = x_optimal - 0.8 * np.exp(-iterations * 0.2) * np.random.normal(0, 0.05, len(iterations))
    x_k = np.maximum(x_k, 0.01)  # 确保在可行域内
    
    # 绘制收敛过程
    ax.plot(iterations, x_k, 'bo-', linewidth=2, markersize=6, label='迭代解 x^(k)')
    ax.axhline(y=x_optimal, color='red', linestyle='--', linewidth=2, label='最优解 x*')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='约束边界')
    
    # 绘制障碍参数变化
    ax2 = ax.twinx()
    ax2.semilogy(iterations, r_k, 'g^-', linewidth=2, markersize=6, label='障碍参数 r^(k)')
    
    ax.set_xlabel('迭代次数 k')
    ax.set_ylabel('解 x^(k)', color='blue')
    ax2.set_ylabel('障碍参数 r^(k)', color='green')
    ax.set_title('内点法收敛过程')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def simple_interior_point_test():
    """绘制简单内点法示例"""
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    x = np.linspace(0.01, 0.99, 1000)
    
    # 目标函数
    f_x = x**2
    
    # 障碍函数
    barrier = -np.log(1 - x)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制函数
    plt.plot(x, f_x, 'b-', linewidth=2, label='f(x) = x²')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='约束边界 x=1')
    
    # 不同障碍参数
    r_values = [0.5, 0.2, 0.1, 0.05]
    colors = ['orange', 'green', 'purple', 'brown']
    
    for i, r in enumerate(r_values):
        phi_x = f_x + r * barrier
        plt.plot(x, phi_x, color=colors[i], linewidth=1.5, 
                label=f'φ(x,r={r}) = f(x) + r·B(x)')
    
    # 标记最优解
    plt.plot(1, 1, 'ro', markersize=8, label='最优解 x*=1')
    
    plt.xlim(0, 1.2)
    plt.ylim(0, 2)
    plt.xlabel('x')
    plt.ylabel('函数值')
    plt.title('内点法示例：min x², s.t. x ≤ 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 填充可行域
    feasible_x = x[x <= 1]
    feasible_f = f_x[x <= 1]
    plt.fill_between(feasible_x, 0, feasible_f, alpha=0.2, color='green', label='可行域')
    
    plt.tight_layout()
    plt.savefig('simple_interior_point_test.png', dpi=300, bbox_inches='tight')
    print("已保存: simple_interior_point_test.png")

if __name__ == "__main__":
    print("生成内点法可视化图形...")
    
    # 生成基本概念图
    fig1 = interior_point_concept_plot()
    fig1.savefig('interior_point_concept.png', dpi=300, bbox_inches='tight')
    print("已保存: interior_point_concept.png")
    
    # 生成收敛过程图
    fig2 = interior_point_convergence_plot()
    fig2.savefig('interior_point_convergence.png', dpi=300, bbox_inches='tight')
    print("已保存: interior_point_convergence.png")
    
    # 生成简单示例图
    simple_interior_point_test()
    
    print("可视化完成！")

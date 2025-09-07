#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优化理论基础可视化脚本
用于生成凸分析、对偶理论等概念的图形演示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def plot_convex_sets():
    """绘制凸集和非凸集的对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：凸集示例
    # 圆形（凸集）
    circle = patches.Circle((0, 0), 1, alpha=0.3, color='blue', label='圆形（凸集）')
    ax1.add_patch(circle)
    
    # 矩形（凸集）
    rect = patches.Rectangle((-0.5, -0.5), 1, 1, alpha=0.3, color='green', label='矩形（凸集）')
    ax1.add_patch(rect)
    
    # 凸多边形
    triangle = patches.Polygon([(0, 0), (1, 0), (0.5, 1)], alpha=0.3, color='red', label='三角形（凸集）')
    ax1.add_patch(triangle)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title('凸集示例')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：非凸集示例
    # 环形（非凸集）
    outer_circle = patches.Circle((0, 0), 1, alpha=0.3, color='blue', label='环形（非凸集）')
    inner_circle = patches.Circle((0, 0), 0.5, alpha=1, color='white')
    ax2.add_patch(outer_circle)
    ax2.add_patch(inner_circle)
    
    # 月牙形（非凸集）
    moon = patches.Wedge((0, 0), 1, 0, 180, alpha=0.3, color='green', label='月牙形（非凸集）')
    ax2.add_patch(moon)
    
    # 星形（非凸集）
    star_points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5
        outer_r = 1
        inner_r = 0.4
        star_points.append((outer_r * np.cos(angle), outer_r * np.sin(angle)))
        star_points.append((inner_r * np.cos(angle + np.pi/5), inner_r * np.sin(angle + np.pi/5)))
    star = patches.Polygon(star_points, alpha=0.3, color='red', label='星形（非凸集）')
    ax2.add_patch(star)
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title('非凸集示例')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_convex_functions():
    """绘制凸函数和非凸函数"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.linspace(-2, 2, 1000)
    
    # 左图：凸函数
    f1 = x**2
    f2 = np.exp(x)
    f3 = -np.log(x[x > 0])
    x3 = x[x > 0]
    
    ax1.plot(x, f1, 'b-', linewidth=2, label='$f(x) = x^2$')
    ax1.plot(x, f2, 'r-', linewidth=2, label='$f(x) = e^x$')
    ax1.plot(x3, f3, 'g-', linewidth=2, label='$f(x) = -\\log(x)$')
    
    # 绘制弦
    x1, x2 = -1, 1
    y1, y2 = x1**2, x2**2
    ax1.plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.7, label='弦')
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 8)
    ax1.set_title('凸函数示例')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：非凸函数
    f4 = x**3
    f5 = np.sin(x)
    f6 = x**4 - 2*x**2
    
    ax2.plot(x, f4, 'b-', linewidth=2, label='$f(x) = x^3$')
    ax2.plot(x, f5, 'r-', linewidth=2, label='$f(x) = \\sin(x)$')
    ax2.plot(x, f6, 'g-', linewidth=2, label='$f(x) = x^4 - 2x^2$')
    
    # 绘制弦
    x1, x2 = -1, 1
    y1, y2 = x1**3, x2**3
    ax2.plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.7, label='弦')
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-3, 3)
    ax2.set_title('非凸函数示例')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_conjugate_functions():
    """绘制共轭函数"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：二次函数的共轭
    x = np.linspace(-3, 3, 1000)
    f = 0.5 * x**2
    f_conj = 0.5 * x**2  # 二次函数的共轭还是二次函数
    
    ax1.plot(x, f, 'b-', linewidth=2, label='$f(x) = \\frac{1}{2}x^2$')
    ax1.plot(x, f_conj, 'r--', linewidth=2, label='$f^*(y) = \\frac{1}{2}y^2$')
    
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 5)
    ax1.set_title('二次函数及其共轭')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：负对数函数的共轭
    x_pos = np.linspace(0.1, 3, 1000)
    f_log = -np.log(x_pos)
    y_neg = np.linspace(-3, -0.1, 1000)
    f_log_conj = -1 - np.log(-y_neg)
    
    ax2.plot(x_pos, f_log, 'b-', linewidth=2, label='$f(x) = -\\log(x)$')
    ax2.plot(y_neg, f_log_conj, 'r--', linewidth=2, label='$f^*(y) = -1 - \\log(-y)$')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-5, 5)
    ax2.set_title('负对数函数及其共轭')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_lagrange_duality():
    """绘制拉格朗日对偶的几何解释"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制可行域
    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-1, 3, 1000)
    X, Y = np.meshgrid(x, y)
    
    # 约束：x + y <= 2, x >= 0, y >= 0
    feasible = (X + Y <= 2) & (X >= 0) & (Y >= 0)
    
    # 目标函数：f(x,y) = x^2 + y^2
    Z = X**2 + Y**2
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=20, colors='blue', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制约束边界
    ax.plot(x, 2-x, 'r-', linewidth=2, label='$x + y = 2$')
    ax.axvline(x=0, color='green', linewidth=2, label='$x = 0$')
    ax.axhline(y=0, color='purple', linewidth=2, label='$y = 0$')
    
    # 填充可行域
    ax.contourf(X, Y, feasible, levels=[0, 1], colors=['lightgreen'], alpha=0.3)
    
    # 标记最优解
    ax.plot(1, 1, 'ro', markersize=8, label='最优解 $(1, 1)$')
    
    # 绘制拉格朗日函数的等高线（固定拉格朗日乘子）
    lambda_val = 1
    Z_lag = X**2 + Y**2 + lambda_val * np.maximum(0, X + Y - 2)
    contour_lag = ax.contour(X, Y, Z_lag, levels=10, colors='orange', alpha=0.8, linestyles='--')
    ax.clabel(contour_lag, inline=True, fontsize=8)
    
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('拉格朗日对偶几何解释')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_kkt_conditions():
    """绘制KKT条件的几何解释"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制可行域
    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-1, 3, 1000)
    X, Y = np.meshgrid(x, y)
    
    # 约束：x^2 + y^2 <= 1, x >= 0
    feasible = (X**2 + Y**2 <= 1) & (X >= 0)
    
    # 目标函数：f(x,y) = x + y
    Z = X + Y
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=20, colors='blue', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制约束边界
    theta = np.linspace(0, 2*np.pi, 1000)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='$x^2 + y^2 = 1$')
    ax.axvline(x=0, color='green', linewidth=2, label='$x = 0$')
    
    # 填充可行域
    ax.contourf(X, Y, feasible, levels=[0, 1], colors=['lightgreen'], alpha=0.3)
    
    # 标记最优解
    ax.plot(1/np.sqrt(2), 1/np.sqrt(2), 'ro', markersize=8, label='最优解')
    
    # 绘制梯度向量
    x_opt = 1/np.sqrt(2)
    y_opt = 1/np.sqrt(2)
    
    # 目标函数梯度
    grad_f = np.array([1, 1])
    ax.arrow(x_opt, y_opt, 0.3*grad_f[0], 0.3*grad_f[1], head_width=0.05, head_length=0.05, fc='blue', ec='blue', label='$\\nabla f$')
    
    # 约束梯度
    grad_g1 = np.array([2*x_opt, 2*y_opt])
    ax.arrow(x_opt, y_opt, 0.2*grad_g1[0], 0.2*grad_g1[1], head_width=0.05, head_length=0.05, fc='red', ec='red', label='$\\nabla g_1$')
    
    grad_g2 = np.array([-1, 0])
    ax.arrow(x_opt, y_opt, 0.2*grad_g2[0], 0.2*grad_g2[1], head_width=0.05, head_length=0.05, fc='green', ec='green', label='$\\nabla g_2$')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('KKT条件几何解释')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_3d_convex_function():
    """绘制3D凸函数"""
    fig = plt.figure(figsize=(12, 5))
    
    # 左图：凸函数
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('凸函数：$f(x,y) = x^2 + y^2$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$f(x,y)$')
    
    # 右图：非凸函数
    ax2 = fig.add_subplot(122, projection='3d')
    Z2 = X**4 + Y**4 - 2*X**2 - 2*Y**2
    
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
    ax2.set_title('非凸函数：$f(x,y) = x^4 + y^4 - 2x^2 - 2y^2$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$f(x,y)$')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("生成最优化理论基础可视化图形...")
    
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 生成各种图形
    fig1 = plot_convex_sets()
    fig1.savefig('convex_sets.png', dpi=300, bbox_inches='tight')
    print("已保存: convex_sets.png")
    
    fig2 = plot_convex_functions()
    fig2.savefig('convex_functions.png', dpi=300, bbox_inches='tight')
    print("已保存: convex_functions.png")
    
    fig3 = plot_conjugate_functions()
    fig3.savefig('conjugate_functions.png', dpi=300, bbox_inches='tight')
    print("已保存: conjugate_functions.png")
    
    fig4 = plot_lagrange_duality()
    fig4.savefig('lagrange_duality.png', dpi=300, bbox_inches='tight')
    print("已保存: lagrange_duality.png")
    
    fig5 = plot_kkt_conditions()
    fig5.savefig('kkt_conditions.png', dpi=300, bbox_inches='tight')
    print("已保存: kkt_conditions.png")
    
    fig6 = plot_3d_convex_function()
    fig6.savefig('3d_convex_function.png', dpi=300, bbox_inches='tight')
    print("已保存: 3d_convex_function.png")
    
    print("可视化完成！")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优性条件可视化脚本
用于生成无约束和约束优化问题最优性条件的图形演示
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

def plot_unconstrained_optimality():
    """绘制无约束优化问题的最优性条件"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.linspace(-3, 3, 1000)
    
    # 左图：一阶条件 - 梯度为零
    f1 = (x - 1)**2 + 1
    ax1.plot(x, f1, 'b-', linewidth=2, label='$f(x) = (x-1)^2 + 1$')
    ax1.plot(1, 1, 'ro', markersize=8, label='最优解 $x^* = 1$')
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='$\\nabla f(x^*) = 0$')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 8)
    ax1.set_title('一阶条件：$\\nabla f(x^*) = 0$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中图：二阶条件 - Hessian半正定
    f2 = x**4 - 4*x**2 + 4
    ax2.plot(x, f2, 'b-', linewidth=2, label='$f(x) = x^4 - 4x^2 + 4$')
    ax2.plot(0, 4, 'ro', markersize=8, label='局部最优解 $x^* = 0$')
    ax2.plot(np.sqrt(2), 0, 'go', markersize=8, label='全局最优解 $x^* = \\sqrt{2}$')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(x=np.sqrt(2), color='green', linestyle='--', alpha=0.7)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-1, 8)
    ax2.set_title('二阶条件：$\\nabla^2 f(x^*) \\succeq 0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 右图：鞍点 - 梯度为零但非最优
    f3 = x**3 - 3*x
    ax3.plot(x, f3, 'b-', linewidth=2, label='$f(x) = x^3 - 3x$')
    ax3.plot(0, 0, 'ro', markersize=8, label='鞍点 $x^* = 0$')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='$\\nabla f(0) = 0$')
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-10, 10)
    ax3.set_title('鞍点：梯度为零但非最优')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_constrained_optimality():
    """绘制约束优化问题的最优性条件"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制可行域
    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-1, 3, 1000)
    X, Y = np.meshgrid(x, y)
    
    # 约束：x^2 + y^2 <= 1, x >= 0, y >= 0
    feasible = (X**2 + Y**2 <= 1) & (X >= 0) & (Y >= 0)
    
    # 目标函数：f(x,y) = x + y
    Z = X + Y
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=20, colors='blue', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制约束边界
    theta = np.linspace(0, np.pi/2, 1000)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='$x^2 + y^2 = 1$')
    ax.axvline(x=0, color='green', linewidth=2, label='$x = 0$')
    ax.axhline(y=0, color='purple', linewidth=2, label='$y = 0$')
    
    # 填充可行域
    ax.contourf(X, Y, feasible, levels=[0, 1], colors=['lightgreen'], alpha=0.3)
    
    # 标记最优解
    x_opt = 1/np.sqrt(2)
    y_opt = 1/np.sqrt(2)
    ax.plot(x_opt, y_opt, 'ro', markersize=8, label='最优解')
    
    # 绘制梯度向量
    # 目标函数梯度
    grad_f = np.array([1, 1])
    ax.arrow(x_opt, y_opt, 0.3*grad_f[0], 0.3*grad_f[1], head_width=0.05, head_length=0.05, 
             fc='blue', ec='blue', label='$\\nabla f$')
    
    # 约束梯度
    grad_g1 = np.array([2*x_opt, 2*y_opt])
    ax.arrow(x_opt, y_opt, 0.2*grad_g1[0], 0.2*grad_g1[1], head_width=0.05, head_length=0.05, 
             fc='red', ec='red', label='$\\nabla g_1$')
    
    grad_g2 = np.array([-1, 0])
    ax.arrow(x_opt, y_opt, 0.2*grad_g2[0], 0.2*grad_g2[1], head_width=0.05, head_length=0.05, 
             fc='green', ec='green', label='$\\nabla g_2$')
    
    grad_g3 = np.array([0, -1])
    ax.arrow(x_opt, y_opt, 0.2*grad_g3[0], 0.2*grad_g3[1], head_width=0.05, head_length=0.05, 
             fc='purple', ec='purple', label='$\\nabla g_3$')
    
    # 绘制法锥
    lambda1 = 1/np.sqrt(2)
    lambda2 = 0
    lambda3 = 0
    normal_cone = lambda1 * grad_g1 + lambda2 * grad_g2 + lambda3 * grad_g3
    ax.arrow(x_opt, y_opt, 0.3*normal_cone[0], 0.3*normal_cone[1], head_width=0.05, head_length=0.05, 
             fc='orange', ec='orange', linewidth=3, label='法锥')
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('约束优化问题的最优性条件')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_kkt_conditions_detailed():
    """绘制KKT条件的详细解释"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 左上：平稳性条件
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)
    
    # 目标函数：f(x,y) = x^2 + y^2
    Z = X**2 + Y**2
    
    contour1 = ax1.contour(X, Y, Z, levels=20, colors='blue', alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=8)
    
    # 约束：x + y <= 1
    constraint_line = 1 - X
    ax1.contour(X, Y, constraint_line, levels=[0], colors='red', linewidths=2, label='$x + y = 1$')
    ax1.fill_between(x, 1-x, 2, alpha=0.3, color='lightgreen', label='可行域')
    
    # 最优解
    ax1.plot(0.5, 0.5, 'ro', markersize=8, label='最优解')
    
    # 梯度
    ax1.arrow(0.5, 0.5, 0.3, 0.3, head_width=0.05, head_length=0.05, fc='blue', ec='blue', label='$\\nabla f$')
    ax1.arrow(0.5, 0.5, 0.2, 0.2, head_width=0.05, head_length=0.05, fc='red', ec='red', label='$\\lambda \\nabla g$')
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_title('平稳性条件：$\\nabla f + \\lambda \\nabla g = 0$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右上：原始可行性
    x2 = np.linspace(-1, 2, 1000)
    y2 = np.linspace(-1, 2, 1000)
    X2, Y2 = np.meshgrid(x2, y2)
    
    feasible = (X2 + Y2 <= 1) & (X2 >= 0) & (Y2 >= 0)
    ax2.contourf(X2, Y2, feasible, levels=[0, 1], colors=['lightgreen'], alpha=0.3, label='可行域')
    
    # 约束边界
    ax2.plot(x2, 1-x2, 'r-', linewidth=2, label='$x + y = 1$')
    ax2.axvline(x=0, color='green', linewidth=2, label='$x = 0$')
    ax2.axhline(y=0, color='purple', linewidth=2, label='$y = 0$')
    
    # 最优解
    ax2.plot(0.5, 0.5, 'ro', markersize=8, label='最优解')
    
    ax2.set_xlim(-1, 2)
    ax2.set_ylim(-1, 2)
    ax2.set_title('原始可行性：$g(x^*) \\leq 0$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 左下：对偶可行性
    lambda_vals = np.linspace(-1, 2, 1000)
    ax3.plot(lambda_vals, np.maximum(0, lambda_vals), 'b-', linewidth=2, label='$\\lambda \\geq 0$')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='$\\lambda = 0$')
    ax3.fill_between(lambda_vals, 0, np.maximum(0, lambda_vals), alpha=0.3, color='lightblue', label='可行区域')
    
    ax3.set_xlim(-1, 2)
    ax3.set_ylim(-0.5, 2)
    ax3.set_xlabel('$\\lambda$')
    ax3.set_ylabel('$\\lambda$')
    ax3.set_title('对偶可行性：$\\lambda \\geq 0$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 右下：互补松弛性
    x4 = np.linspace(-1, 2, 1000)
    g_vals = 1 - x4  # g(x) = 1 - x
    lambda_vals = np.maximum(0, 1 - x4)  # λ = max(0, 1-x)
    product = g_vals * lambda_vals
    
    ax4.plot(x4, g_vals, 'b-', linewidth=2, label='$g(x) = 1 - x$')
    ax4.plot(x4, lambda_vals, 'r-', linewidth=2, label='$\\lambda = \\max(0, 1-x)$')
    ax4.plot(x4, product, 'g-', linewidth=2, label='$\\lambda g(x) = 0$')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax4.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    
    ax4.set_xlim(-1, 2)
    ax4.set_ylim(-1, 2)
    ax4.set_xlabel('$x$')
    ax4.set_ylabel('值')
    ax4.set_title('互补松弛性：$\\lambda g(x) = 0$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_3d_optimality():
    """绘制3D最优性条件"""
    fig = plt.figure(figsize=(15, 5))
    
    # 左图：无约束优化
    ax1 = fig.add_subplot(131, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.plot([0], [0], [0], 'ro', markersize=10, label='最优解')
    ax1.set_title('无约束优化：$f(x,y) = x^2 + y^2$')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$f(x,y)$')
    
    # 中图：约束优化
    ax2 = fig.add_subplot(132, projection='3d')
    Z2 = X + Y
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
    
    # 约束边界
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_z = circle_x + circle_y
    ax2.plot(circle_x, circle_y, circle_z, 'r-', linewidth=3, label='约束边界')
    
    ax2.plot([1/np.sqrt(2)], [1/np.sqrt(2)], [np.sqrt(2)], 'ro', markersize=10, label='最优解')
    ax2.set_title('约束优化：$f(x,y) = x + y$, s.t. $x^2 + y^2 = 1$')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$f(x,y)$')
    
    # 右图：KKT条件几何解释
    ax3 = fig.add_subplot(133, projection='3d')
    Z3 = X**2 + Y**2
    surf3 = ax3.plot_surface(X, Y, Z3, cmap='coolwarm', alpha=0.6)
    
    # 约束
    ax3.plot(circle_x, circle_y, circle_z, 'r-', linewidth=3, label='约束边界')
    
    # 最优解
    x_opt = 1/np.sqrt(2)
    y_opt = 1/np.sqrt(2)
    z_opt = x_opt**2 + y_opt**2
    ax3.plot([x_opt], [y_opt], [z_opt], 'ro', markersize=10, label='最优解')
    
    # 梯度向量
    ax3.quiver(x_opt, y_opt, z_opt, 0.5, 0.5, 0, color='blue', label='$\\nabla f$')
    ax3.quiver(x_opt, y_opt, z_opt, 0.3, 0.3, 0, color='red', label='$\\lambda \\nabla g$')
    
    ax3.set_title('KKT条件几何解释')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_zlabel('$f(x,y)$')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("生成最优性条件可视化图形...")
    
    # 忽略字体警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 生成各种图形
    fig1 = plot_unconstrained_optimality()
    fig1.savefig('unconstrained_optimality.png', dpi=300, bbox_inches='tight')
    print("已保存: unconstrained_optimality.png")
    
    fig2 = plot_constrained_optimality()
    fig2.savefig('constrained_optimality.png', dpi=300, bbox_inches='tight')
    print("已保存: constrained_optimality.png")
    
    fig3 = plot_kkt_conditions_detailed()
    fig3.savefig('kkt_conditions_detailed.png', dpi=300, bbox_inches='tight')
    print("已保存: kkt_conditions_detailed.png")
    
    fig4 = plot_3d_optimality()
    fig4.savefig('3d_optimality.png', dpi=300, bbox_inches='tight')
    print("已保存: 3d_optimality.png")
    
    print("可视化完成！")

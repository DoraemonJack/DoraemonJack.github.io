#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学公式测试脚本
用于测试各种数学计算和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def test_tree_properties():
    """测试树结构相关数学公式"""
    print("=== 树结构数学公式测试 ===")
    
    # 测试 m叉树第i层最多节点数
    m = 3  # 3叉树
    for i in range(5):
        max_nodes = m ** i
        print(f"3叉树第{i}层最多有 {max_nodes} 个节点")
    
    print()
    
    # 测试高度为h的m叉树最多节点数
    h = 4
    total_max = (m ** (h + 1) - 1) / (m - 1)
    print(f"高度为{h}的{m}叉树最多有 {total_max:.0f} 个节点")
    
    # 测试最小高度公式
    n = 20  # 节点数
    min_height = np.ceil(np.log(m * (n - 1) + 1) / np.log(m)) - 1
    print(f"具有{n}个节点的{m}叉树的最小高度为 {min_height:.0f}")

def test_penalty_function():
    """测试惩罚函数法相关计算"""
    print("\n=== 惩罚函数法测试 ===")
    
    # 简单例子：min x², s.t. x ≥ 1
    x_values = np.linspace(0, 3, 100)
    
    # 目标函数
    f_x = x_values ** 2
    
    # 约束函数
    g_x = x_values - 1
    
    # 惩罚函数
    penalty = np.maximum(0, -g_x) ** 2
    
    # 不同惩罚参数
    r_values = [0.1, 0.5, 1.0, 2.0]
    
    print("惩罚参数\t最优解\t\t函数值")
    print("-" * 40)
    
    for r in r_values:
        # 惩罚函数
        phi_x = f_x + r * penalty
        
        # 找到最小值
        min_idx = np.argmin(phi_x)
        x_opt = x_values[min_idx]
        f_opt = f_x[min_idx]
        
        print(f"r = {r}\t\tx* = {x_opt:.3f}\tf(x*) = {f_opt:.3f}")

def plot_simple_function():
    """绘制简单函数图"""
    x = np.linspace(-2, 3, 1000)
    
    # 目标函数
    f_x = x ** 2
    
    # 约束
    g_x = x - 1
    
    # 惩罚函数
    penalty = np.maximum(0, -g_x) ** 2
    
    plt.figure(figsize=(10, 6))
    
    # 绘制函数
    plt.plot(x, f_x, 'b-', linewidth=2, label='f(x) = x²')
    plt.plot(x, g_x, 'r--', linewidth=2, label='g(x) = x - 1')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 填充惩罚区域
    plt.fill_between(x, 0, penalty, alpha=0.2, color='red', label='惩罚区域')
    
    # 不同惩罚参数
    r_values = [0.1, 0.5, 1.0, 2.0]
    colors = ['orange', 'green', 'purple', 'brown']
    
    for i, r in enumerate(r_values):
        phi_x = f_x + r * penalty
        plt.plot(x, phi_x, color=colors[i], linewidth=1.5, 
                label=f'φ(x,r={r}) = f(x) + r·P(x)')
    
    plt.xlim(-2, 3)
    plt.ylim(-1, 8)
    plt.xlabel('x')
    plt.ylabel('函数值')
    plt.title('惩罚函数法示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_penalty_test.png', dpi=300, bbox_inches='tight')
    print("已保存: simple_penalty_test.png")

if __name__ == "__main__":
    print("开始数学公式测试...")
    
    # 测试树结构公式
    test_tree_properties()
    
    # 测试惩罚函数法
    test_penalty_function()
    
    # 绘制简单图形
    plot_simple_function()
    
    print("\n测试完成！")

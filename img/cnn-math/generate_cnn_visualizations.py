

#!/usr/bin/env python3
"""
生成卷积神经网络（CNN）数学理论的可视化图表
包含卷积操作、感受野、架构对比等多个关键概念
"""
# ===== 第1步：在导入 matplotlib.pyplot 之前配置字体 =====
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import os
from PIL import Image

# 确保保存路径
import os
save_dir = os.path.dirname(os.path.abspath(__file__))

# ===== 辅助函数：将图片转换为WebP格式 =====
def save_as_webp(fig_obj, filename_base, dpi=150, quality=90):
    """
    将matplotlib图表保存为WebP格式
    
    参数:
        fig_obj: matplotlib figure 对象
        filename_base: 文件名（不包含扩展名）
        dpi: 分辨率
        quality: WebP质量 (0-100)
    """
    # 先保存为临时PNG
    png_path = f'{save_dir}/{filename_base}.png'
    webp_path = f'{save_dir}/{filename_base}.webp'
    
    # 保存为PNG
    fig_obj.savefig(png_path, dpi=dpi, bbox_inches='tight')
    
    # 使用PIL转换为WebP
    try:
        img = Image.open(png_path)
        # 转换为RGB格式（WebP需要）
        if img.mode in ('RGBA', 'LA', 'P'):
            # 为RGBA图像创建白色背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 保存为WebP
        img.save(webp_path, 'WEBP', quality=quality)
        
        # 删除临时PNG文件
        os.remove(png_path)
        
        # 获取文件大小用于统计
        webp_size = os.path.getsize(webp_path) / 1024  # KB
        print(f"✓ 已生成: {filename_base}.webp ({webp_size:.1f} KB)")
        
    except Exception as e:
        print(f"✗ 转换失败 {filename_base}: {e}")
        # 如果转换失败，保留PNG文件
        if os.path.exists(png_path):
            print(f"  已保留原始PNG: {filename_base}.png")

def create_convolution_operation():
    """
    1. 卷积操作的可视化
    展示2D卷积如何扫描图像
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('二维卷积操作的详细过程 (输入: 5×5, 卷积核: 3×3)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 定义输入和卷积核
    input_img = np.array([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
        [9, 10, 11, 12, 13],
        [13, 14, 15, 16, 17],
        [17, 18, 19, 20, 21]
    ], dtype=float)
    
    kernel = np.array([
        [0.1, 0.2, 0.1],
        [0.0, 0.5, 0.0],
        [-0.1, -0.2, -0.1]
    ])
    
    positions = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 0), (2, 2)]
    
    # 前3个子图：展示卷积过程
    for idx, (pos_row, pos_col) in enumerate(positions[:3]):
        ax = axes.flat[idx]
        
        # 绘制输入矩阵
        im = ax.imshow(input_img, cmap='YlOrRd', alpha=0.3)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_title(f'第 {idx+1} 步：位置 ({pos_row}, {pos_col})', 
                    fontsize=11, fontweight='bold')
        
        # 高亮卷积核覆盖的区域
        rect = Rectangle((pos_col-0.5, pos_row-0.5), 3, 3, 
                         linewidth=3, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        
        # 绘制矩阵值
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{int(input_img[i, j])}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        # 计算卷积值
        roi = input_img[pos_row:pos_row+3, pos_col:pos_col+3]
        conv_val = np.sum(roi * kernel)
        
        ax.text(0.5, -0.15, f'卷积结果: {conv_val:.2f}', 
               transform=ax.transAxes, ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 第4个子图：输出特征图
    ax4 = axes.flat[3]
    output = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            roi = input_img[i:i+3, j:j+3]
            output[i, j] = np.sum(roi * kernel)
    
    im4 = ax4.imshow(output, cmap='viridis')
    ax4.set_title('输出特征图 (3×3)', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, f'{output[i, j]:.1f}',
                    ha="center", va="center", color="white", fontsize=9, fontweight='bold')
    
    plt.colorbar(im4, ax=ax4, label='特征值')
    
    # 第5个子图：卷积核可视化
    ax5 = axes.flat[4]
    im5 = ax5.imshow(kernel, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax5.set_title('卷积核权重', fontsize=11, fontweight='bold')
    ax5.set_xticks(range(3))
    ax5.set_yticks(range(3))
    
    for i in range(3):
        for j in range(3):
            color = 'white' if abs(kernel[i, j]) > 0.15 else 'black'
            ax5.text(j, i, f'{kernel[i, j]:.1f}',
                    ha="center", va="center", color=color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im5, ax=ax5, label='权重值')
    
    # 第6个子图：数学公式
    ax6 = axes.flat[5]
    ax6.axis('off')
    formula_text = r'$Y[i,j] = \sum_{u=0}^{2} \sum_{v=0}^{2} W[u,v] \cdot X[i+u, j+v]$' + '\n\n'
    formula_text += r'$= X[i:i+3, j:j+3] \odot W$' + '\n\n'
    formula_text += 'Element-wise Product Summation (Hadamard Product)\n\n'
    formula_text += f'Example: Y[0,0] = {np.sum(input_img[0:3, 0:3] * kernel):.2f}'
    
    ax6.text(0.1, 0.5, formula_text, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '01_convolution_operation', dpi=150, quality=90)
    plt.close()


def create_receptive_field_evolution():
    """
    2. 感受野的逐层扩展
    展示深层网络如何扩展感受野
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('CNN中感受野的逐层进化', fontsize=14, fontweight='bold')
    
    # 计算感受野
    def compute_rf(num_layers, k=3, s=1):
        rf = 1
        for _ in range(num_layers):
            rf = rf + (k - 1) * (2 ** (_ if _ < 3 else 3))
        return rf
    
    layers = np.arange(1, 11)
    
    # 第1个子图：卷积核大小的影响
    ax1 = axes[0, 0]
    for k in [3, 5, 7]:
        rf = []
        for l in range(1, 11):
            r = 1
            for i in range(l):
                r = r + (k - 1)
            rf.append(r)
        ax1.plot(layers, rf, marker='o', linewidth=2, label=f'Kernel Size: {k}×{k}', markersize=6)
    
    ax1.set_xlabel('网络深度（层数）', fontsize=11, fontweight='bold')
    ax1.set_ylabel('感受野大小', fontsize=11, fontweight='bold')
    ax1.set_title('卷积核大小的影响', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # 第2个子图：步幅的影响
    ax2 = axes[0, 1]
    for stride_pattern in ['1,1,1', '1,2,1', '2,2,2']:
        strides = [int(s) for s in stride_pattern.split(',')]
        while len(strides) < 10:
            strides.append(1)
        
        rf = []
        r = 1
        for i in range(1, 11):
            s = strides[i-1]
            r = r + (3 - 1) * np.prod(strides[:i-1], dtype=int)
            rf.append(r)
        
        ax2.plot(layers, rf, marker='s', linewidth=2, label=f'步幅模式: {stride_pattern}', markersize=6)
    
    ax2.set_xlabel('网络深度（层数）', fontsize=11, fontweight='bold')
    ax2.set_ylabel('感受野大小', fontsize=11, fontweight='bold')
    ax2.set_title('步幅的影响', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)
    
    # 第3个子图：VGG-16的感受野进化
    ax3 = axes[1, 0]
    vgg_config = [
        ('Conv 3×3', 3, 1),
        ('Conv 3×3', 3, 1),
        ('MaxPool 2×2', 0, 2),
        ('Conv 3×3', 3, 1),
        ('Conv 3×3', 3, 1),
        ('MaxPool 2×2', 0, 2),
    ]
    
    rf_vgg = [1]
    for kernel, stride in [(3, 1), (3, 1), (0, 2), (3, 1), (3, 1), (0, 2)]:
        if kernel > 0:
            rf_vgg.append(rf_vgg[-1] + (kernel - 1) * 2 ** (len(rf_vgg) - 1))
        else:
            rf_vgg.append(rf_vgg[-1])
    
    steps = list(range(len(vgg_config) + 1))
    labels = ['Input'] + [v[0] for v in vgg_config]
    colors = ['blue' if 'Pool' in l else 'green' for l in labels]
    
    ax3.bar(steps, rf_vgg, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('感受野大小', fontsize=11, fontweight='bold')
    ax3.set_title('VGG-16的感受野演变', fontsize=12, fontweight='bold')
    ax3.set_xticks(steps)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 第4个子图：感受野的递推公式
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    formula_text = 'Receptive Field Recurrence Formula\n\n'
    formula_text += r'$RF_0 = 1$' + '\n\n'
    formula_text += r'$RF_{l} = RF_{l-1} + (k_l - 1) \prod_{i=0}^{l-1} s_i$' + '\n\n'
    formula_text += 'Where:\n'
    formula_text += r'• $k_l$: Kernel size at layer $l$' + '\n'
    formula_text += r'• $s_i$: Stride at layer $i$' + '\n\n'
    formula_text += 'Example (3×3, stride=1):\n'
    formula_text += r'• RF_1 = 1 + (3-1) × 1 = 3' + '\n'
    formula_text += r'• RF_2 = 3 + (3-1) × 1 = 5' + '\n'
    formula_text += r'• RF_3 = 5 + (3-1) × 1 = 7'
    
    ax4.text(0.05, 0.5, formula_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax4.transAxes)
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '02_receptive_field_evolution', dpi=150, quality=90)
    plt.close()


def create_cnn_architecture_comparison():
    """
    3. CNN架构对比（参数效率）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CNN架构的参数效率对比', fontsize=14, fontweight='bold')
    
    # 第1个子图：参数数量对比
    architectures = ['LeNet-5', 'AlexNet', 'VGG-16', 'ResNet-50', 'DenseNet-121']
    params_millions = [0.06, 60, 138, 25.5, 7.0]
    accuracy = [0.99, 0.84, 0.92, 0.95, 0.96]  # ImageNet top-1
    
    ax1 = axes[0, 0]
    colors_arch = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars1 = ax1.bar(architectures, params_millions, color=colors_arch, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('参数数量 (百万)', fontsize=11, fontweight='bold')
    ax1.set_title('网络参数数量对比', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, param in zip(bars1, params_millions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticklabels(architectures, rotation=45, ha='right')
    
    # 第2个子图：参数与准确度的关系
    ax2 = axes[0, 1]
    scatter = ax2.scatter(params_millions, accuracy, s=300, c=colors_arch, alpha=0.7, 
                         edgecolors='black', linewidth=2)
    
    for i, arch in enumerate(architectures):
        ax2.annotate(arch, (params_millions[i], accuracy[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('参数数量 (百万)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ImageNet 准确度 (Top-1)', fontsize=11, fontweight='bold')
    ax2.set_title('参数数量与准确度的权衡', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 第3个子图：参数共享的效率
    ax3 = axes[1, 0]
    
    categories = ['全连接\n(224×224×3→10)', 'CNN\n(VGG-16)']
    fc_params = [224*224*3 * 1000]  # 假设全连接层输出1000类
    cnn_params = [138 * 1e6]
    
    x_pos = np.arange(len(categories))
    ax3.bar(x_pos, [fc_params[0]/1e9, cnn_params[0]/1e9], 
           color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('参数数量 (十亿)', fontsize=11, fontweight='bold')
    ax3.set_title('参数共享的效果', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加参数减少比例
    reduction_ratio = fc_params[0] / cnn_params[0]
    ax3.text(0.5, 0.95, f'参数减少 {reduction_ratio:.0f}× 倍!', 
            transform=ax3.transAxes, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 第4个子图：架构演进时间线
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    timeline_data = [
        (1998, 'LeNet-5', '手写体识别 (MNIST)'),
        (2012, 'AlexNet', 'ImageNet 突破'),
        (2014, 'VGG-16', '深度探索'),
        (2015, 'ResNet-50', '跳跃连接'),
        (2017, 'DenseNet-121', '密集连接'),
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(timeline_data))
    
    for i, (year, name, desc) in enumerate(timeline_data):
        y = y_positions[i]
        
        # 绘制时间线点
        ax4.plot(0.15, y, 'o', markersize=12, color=colors_arch[i], 
                transform=ax4.transAxes, zorder=3)
        
        # 绘制文本
        ax4.text(0.25, y + 0.02, f'{year}: {name}', fontsize=10, fontweight='bold',
                transform=ax4.transAxes, verticalalignment='center')
        ax4.text(0.25, y - 0.02, desc, fontsize=9, style='italic',
                transform=ax4.transAxes, verticalalignment='center', color='gray')
    
    # 绘制时间线
    ax4.plot([0.15, 0.15], [0.08, 0.92], 'k-', linewidth=2, transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.text(0.15, 0.98, 'CNN架构演进历程', fontsize=12, fontweight='bold',
            transform=ax4.transAxes, ha='center', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '03_architecture_comparison', dpi=150, quality=90)
    plt.close()


def create_feature_hierarchy():
    """
    4. CNN学习的特征层级结构
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 创建主标题
    fig.suptitle('CNN的特征层级表示（从低级到高级）', fontsize=14, fontweight='bold')
    
    # 创建特征示意图
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)
    
    # 第0层：输入图像
    ax_input = fig.add_subplot(gs[0, 1:4])
    ax_input.text(0.5, 0.5, '原始输入图像\n(224×224×3)', 
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 transform=ax_input.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, pad=0.8))
    ax_input.axis('off')
    
    # 第1层：边缘特征
    features_layer1 = [
        ('边缘', '水平边缘', 'lightgreen'),
        ('角落', '垂直边缘', 'lightgreen'),
        ('纹理', '对角线', 'lightgreen'),
    ]
    
    for i, (title, desc, color) in enumerate(features_layer1):
        ax = fig.add_subplot(gs[1, i+1])
        # 创建简单的边缘示意图
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        ax.plot(x, y1, 'b-', linewidth=2, label='水平')
        ax.plot(x, y2, 'r-', linewidth=2, label='垂直')
        ax.set_title(f'Layer-1: {title}\n({desc})', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(color)
    
    # 第2层：局部特征
    features_layer2 = [
        ('角点组', '简单形状', 'lightyellow'),
        ('纹理块', '局部模式', 'lightyellow'),
        ('线条', '组合特征', 'lightyellow'),
    ]
    
    for i, (title, desc, color) in enumerate(features_layer2):
        ax = fig.add_subplot(gs[2, i+1])
        # 创建简单的形状
        circle = plt.Circle((0.3, 0.5), 0.2, color='blue', alpha=0.5)
        square = patches.Rectangle((0.55, 0.35), 0.2, 0.2, color='red', alpha=0.5)
        ax.add_patch(circle)
        ax.add_patch(square)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Layer-2: {title}\n({desc})', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(color)
    
    # 添加连接箭头和说明
    ax_note1 = fig.add_subplot(gs[1, 0])
    ax_note1.text(0.5, 0.5, '低级特征\n• 边缘\n• 角点\n• 纹理\n• 颜色\n\n参数: 少', 
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=ax_note1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_note1.axis('off')
    
    ax_note2 = fig.add_subplot(gs[2, 0])
    ax_note2.text(0.5, 0.5, '中级特征\n• 局部形状\n• 纹理\n• 部分\n\n参数: 增多', 
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=ax_note2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_note2.axis('off')
    
    ax_note3 = fig.add_subplot(gs[1:, 4])
    ax_note3.text(0.5, 0.5, '高级特征\n\n• 物体部分\n• 语义概念\n• 类别特征\n\n参数: 最多\n\n感受野: 增大', 
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=ax_note3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax_note3.axis('off')
    
    save_as_webp(plt.gcf(), '04_feature_hierarchy', dpi=150, quality=90)
    plt.close()


def create_pooling_visualization():
    """
    5. 池化操作的可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('池化操作的效果演示', fontsize=14, fontweight='bold')
    
    # 创建原始特征图
    np.random.seed(42)
    feature_map = np.array([
        [1, 5, 3, 7],
        [9, 2, 8, 4],
        [6, 3, 7, 2],
        [5, 9, 1, 6]
    ], dtype=float)
    
    # 第1个子图：原始特征图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(feature_map, cmap='YlOrRd')
    ax1.set_title('原始特征图 (4×4)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{int(feature_map[i, j])}',
                    ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # 第2个子图：最大值池化 (2×2)
    ax2 = axes[0, 1]
    max_pool = np.array([
        [9, 8],
        [9, 6]
    ], dtype=float)
    
    im2 = ax2.imshow(max_pool, cmap='YlOrRd')
    ax2.set_title('最大值池化 (2×2) - 输出: 2×2', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(2))
    
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{int(max_pool[i, j])}',
                    ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # 第3个子图：平均池化 (2×2)
    ax3 = axes[1, 0]
    avg_pool = np.array([
        [4.25, 5.5],
        [5.75, 4.0]
    ], dtype=float)
    
    im3 = ax3.imshow(avg_pool, cmap='YlOrRd')
    ax3.set_title('平均池化 (2×2) - 输出: 2×2', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(2))
    ax3.set_yticks(range(2))
    
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f'{avg_pool[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    # 第4个子图：池化的数学公式
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    formula_text = 'Mathematical Definition of Pooling\n\n'
    formula_text += 'Max Pooling:\n'
    formula_text += r'$y_{ij} = \max\limits_{(u,v) \in \mathcal{N}(i,j)} x_{uv}$' + '\n\n'
    formula_text += 'Example: max([1,5,9,2]) = 9\n\n'
    formula_text += 'Average Pooling:\n'
    formula_text += r'$y_{ij} = \frac{1}{|S|}\sum_{(u,v) \in S} x_{uv}$' + '\n\n'
    formula_text += 'Example: avg([1,5,9,2]) = 4.25\n\n'
    formula_text += 'Benefits:\n'
    formula_text += '• Dimension reduction: spatial resolution halved\n'
    formula_text += '• Robustness: invariance to small translations\n'
    formula_text += '• Efficiency: parameter reduction'
    
    ax4.text(0.1, 0.5, formula_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '05_pooling_visualization', dpi=150, quality=90)
    plt.close()


def create_gradient_flow():
    """
    6. 残差连接对梯度流动的影响
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('残差连接 (Residual Connection) 对梯度流动的改善', 
                fontsize=14, fontweight='bold')
    
    # 模拟梯度大小随深度的变化
    depth = np.arange(1, 21)
    
    # 普通网络：梯度消失
    normal_gradient = 0.98 ** depth  # 假设每层梯度衰减到0.98
    
    # 残差网络：梯度保留
    residual_gradient = 0.95 ** depth + 0.5  # 残差连接保证有直通路径
    
    # 第1个子图：梯度大小对比
    ax1 = axes[0]
    ax1.semilogy(depth, normal_gradient, 'r-o', linewidth=2.5, markersize=6, 
                 label='普通网络（无残差）', alpha=0.8)
    ax1.semilogy(depth, residual_gradient, 'g-s', linewidth=2.5, markersize=6, 
                 label='残差网络', alpha=0.8)
    ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='消失阈值')
    ax1.set_xlabel('网络深度（层数）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('梯度大小（对数尺度）', fontsize=12, fontweight='bold')
    ax1.set_title('梯度流动对比', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-7, 1])
    
    # 第2个子图：网络架构图
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # 绘制普通网络
    ax2.text(1, 9, '普通网络', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='mistyrose'))
    
    for i in range(7):
        # 网络层
        rect = FancyBboxPatch((0.5, 8-i*1.2), 1, 0.8, boxstyle="round,pad=0.05", 
                             edgecolor='red', facecolor='lightcoral', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(1, 8-i*1.2+0.4, f'Layer{i+1}', ha='center', va='center', fontsize=8)
        
        # 梯度箭头（消失）
        arrow_alpha = 0.98 ** (i + 1)
        arrow = FancyArrowPatch((0.2, 8-i*1.2+0.4), (-0.3, 8-i*1.2+0.4),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='red', alpha=min(arrow_alpha * 2, 1))
        ax2.add_patch(arrow)
    
    ax2.text(0.5, 0.5, '梯度消失\n↓', ha='center', fontsize=9, fontweight='bold', color='red')
    
    # 绘制残差网络
    ax2.text(5, 9, '残差网络', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='honeydew'))
    
    for i in range(7):
        # 网络层
        rect = FancyBboxPatch((4.5, 8-i*1.2), 1, 0.8, boxstyle="round,pad=0.05",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(5, 8-i*1.2+0.4, f'Layer{i+1}', ha='center', va='center', fontsize=8)
        
        # 梯度箭头（保留）
        arrow = FancyArrowPatch((4.2, 8-i*1.2+0.4), (3.7, 8-i*1.2+0.4),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='green', alpha=0.8)
        ax2.add_patch(arrow)
        
        # 跳连接（直通路径）
        if i > 0:
            skip_connection = FancyArrowPatch((5, 8-(i-1)*1.2), (5, 8-i*1.2),
                                            arrowstyle='-', mutation_scale=20, linewidth=1.5,
                                            color='blue', linestyle='--', alpha=0.6)
            ax2.add_patch(skip_connection)
    
    ax2.text(5, 0.5, '梯度保留\n✓', ha='center', fontsize=9, fontweight='bold', color='green')
    
    # 添加数学公式
    ax2.text(5, 1.5, r'$\nabla L = \frac{\partial L}{\partial x^{(l)}} \cdot (1 + \frac{\partial F}{\partial x^{(l)}})$',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '06_gradient_flow_residual', dpi=150, quality=90)
    plt.close()


def create_yolov3_pipeline():
    """
    7. YOLOv3目标检测管道
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('YOLOv3 目标检测管道完全解析', fontsize=14, fontweight='bold')
    
    # 第1个子图：输入图像
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.text(5, 5, '输入图像\n416×416×3', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
    # 第2个子图：骨干网络（Darknet-53）
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    y_pos = 9
    layer_info = [
        ('Conv 3×3', '32通道'),
        ('Conv 3×3, stride=2', '64通道'),
        ('残差块×1', '64通道'),
        ('...', '...'),
        ('残差块×4', '1024通道'),
        ('输出', '13×13×1024'),
    ]
    
    for i, (layer, desc) in enumerate(layer_info):
        color = 'lightcoral' if i == len(layer_info) - 1 else 'lightyellow'
        rect = FancyBboxPatch((0.5, y_pos-0.7), 9, 0.6, boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(5, y_pos-0.4, f'{layer} → {desc}', ha='center', va='center', fontsize=9)
        y_pos -= 1.2
    
    ax2.text(5, 10.5, '骨干网络 (Darknet-53)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 第3个子图：多尺度特征提取
    ax3 = axes[0, 2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    scales = [
        ('P3 (细粒度)', '52×52×256', '小目标'),
        ('P2 (中粒度)', '26×26×512', '中等目标'),
        ('P1 (粗粒度)', '13×13×1024', '大目标'),
    ]
    
    y_pos = 8.5
    for i, (scale_name, size, target) in enumerate(scales):
        color = ['#ff9999', '#ffcc99', '#99ccff'][i]
        rect = FancyBboxPatch((0.5, y_pos-0.7), 9, 0.8, boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax3.add_patch(rect)
        ax3.text(2, y_pos-0.3, scale_name, ha='center', va='center', fontsize=8, fontweight='bold')
        ax3.text(5.5, y_pos-0.3, size, ha='center', va='center', fontsize=8)
        ax3.text(8, y_pos-0.3, target, ha='center', va='center', fontsize=8)
        y_pos -= 2.2
    
    ax3.text(5, 10.5, '多尺度特征金字塔', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 第4个子图：预测编码
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    pred_text = 'Prediction per Grid Cell\n(3 Anchors)\n\n'
    pred_text += 'Bounding Box:\n'
    pred_text += r'• $b_x = \sigma(t_x) + c_x$' + '\n'
    pred_text += r'• $b_y = \sigma(t_y) + c_y$' + '\n'
    pred_text += r'• $b_w = p_w e^{t_w}$' + '\n'
    pred_text += r'• $b_h = p_h e^{t_h}$' + '\n\n'
    pred_text += 'Classification:\n'
    pred_text += r'• $p_c = \sigma(c)$' + '  (Class Probability)\n'
    pred_text += r'• objectness = $\sigma(o)$'
    
    ax4.text(0.1, 0.5, pred_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax4.transAxes)
    
    # 第5个子图：损失函数
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    loss_text = 'Total Loss Function\n\n'
    loss_text += r'$\mathcal{L} = \mathcal{L}_{box} + \mathcal{L}_{obj} + \mathcal{L}_{cls}$' + '\n\n'
    loss_text += 'Bounding Box Loss (SmoothL1):\n'
    loss_text += r'$\mathcal{L}_{box} = \lambda_{box} \sum$ SmoothL1' + '\n\n'
    loss_text += 'Objectness Loss (BCE):\n'
    loss_text += r'$\mathcal{L}_{obj} = -[t \log p + (1-t)\log(1-p)]$' + '\n\n'
    loss_text += 'Classification Loss (Multi-label BCE):\n'
    loss_text += r'$\mathcal{L}_{cls} = \sum_c$ BCE$(p_c, \hat{p}_c)$'
    
    ax5.text(0.1, 0.5, loss_text, fontsize=8.5, 
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax5.transAxes)
    
    # 第6个子图：输出与NMS
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    output_text = 'Post-processing (NMS)\n\n'
    output_text += '1. Confidence Filtering\n'
    output_text += '   Keep only p_obj > threshold\n\n'
    output_text += '2. IoU Calculation\n'
    output_text += r'   $IoU = \frac{Intersection}{Union}$' + '\n\n'
    output_text += '3. Non-Maximum Suppression\n'
    output_text += '   • Sort by confidence\n'
    output_text += '   • Remove boxes with IoU > threshold\n\n'
    output_text += 'Output:\n'
    output_text += 'Bounding box coords + class + confidence'
    
    ax6.text(0.1, 0.5, output_text, fontsize=8.5, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax6.transAxes)
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '07_yolov3_pipeline', dpi=150, quality=90)
    plt.close()


def create_equivariance_demonstration():
    """
    8. 卷积的平移等变性演示
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('卷积操作的平移等变性 (Translation Equivariance)', 
                fontsize=14, fontweight='bold')
    
    # 创建简单的输入图像（有一个对象）
    input_img = np.zeros((8, 8))
    input_img[2:5, 2:5] = 1  # 在位置 (2-4, 2-4) 放置一个对象
    
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian 边缘检测
    
    # 第1个子图：原始图像
    ax1 = axes[0, 0]
    im1 = ax1.imshow(input_img, cmap='gray')
    ax1.set_title('原始图像\n对象在 (2-4, 2-4)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.grid(True, alpha=0.3)
    
    # 卷积输出1
    conv_output1 = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            roi = input_img[i:i+3, j:j+3]
            conv_output1[i, j] = np.sum(roi * kernel)
    
    ax2 = axes[0, 1]
    im2 = ax2.imshow(conv_output1, cmap='RdBu_r')
    ax2.set_title('卷积输出1\n特征激活在 (1-3, 1-3)', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(6))
    ax2.set_yticks(range(6))
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2)
    
    # 第2个子图：平移后的图像
    input_img_shifted = np.zeros((8, 8))
    input_img_shifted[4:7, 4:7] = 1  # 向右下移动2个像素
    
    ax3 = axes[0, 2]
    im3 = ax3.imshow(input_img_shifted, cmap='gray')
    ax3.set_title('平移后的图像\n对象平移到 (4-6, 4-6)', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.grid(True, alpha=0.3)
    
    # 第4个子图：平移图像的卷积输出
    conv_output2 = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            roi = input_img_shifted[i:i+3, j:j+3]
            conv_output2[i, j] = np.sum(roi * kernel)
    
    ax4 = axes[1, 0]
    im4 = ax4.imshow(conv_output2, cmap='RdBu_r')
    ax4.set_title('平移后的卷积输出\n特征激活在 (3-5, 3-5)', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(6))
    ax4.set_yticks(range(6))
    ax4.grid(True, alpha=0.3)
    plt.colorbar(im4, ax=ax4)
    
    # 第5个子图：对比
    ax5 = axes[1, 1]
    ax5.imshow(conv_output1, cmap='RdBu_r', alpha=0.5, label='原始输出')
    ax5.imshow(np.roll(conv_output2, -2, axis=(0, 1)), cmap='RdBu_r', alpha=0.5)
    ax5.set_title('输出对齐对比\n两者相同（平移等变性！）', fontsize=11, fontweight='bold')
    ax5.set_xticks(range(6))
    ax5.set_yticks(range(6))
    ax5.grid(True, alpha=0.3)
    
    # 第6个子图：数学说明
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    eqv_text = 'Translation Equivariance\n(Mathematical Form)\n\n'
    eqv_text += r'Define translation operator:' + '\n'
    eqv_text += r'$\tau_d f(x) = f(x - d)$' + '\n\n'
    eqv_text += r'Convolution equivariance:' + '\n'
    eqv_text += r'$(\tau_d f) * g = \tau_d(f * g)$' + '\n\n'
    eqv_text += 'Meaning: Input translation\n'
    eqv_text += 'causes output translation\n'
    eqv_text += 'in the same manner\n\n'
    eqv_text += 'Benefits:\n'
    eqv_text += '✓ CNN features are robust\n'
    eqv_text += '  to spatial locations\n'
    eqv_text += '✓ Improved generalization'
    
    ax6.text(0.1, 0.5, eqv_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax6.transAxes)
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '08_equivariance_demonstration', dpi=150, quality=90)
    plt.close()


def create_training_dynamics():
    """
    9. CNN训练动态：损失曲线和学习过程
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('YOLOv3 目标检测模型的训练动态', fontsize=14, fontweight='bold')
    
    # 模拟训练数据
    epochs = np.arange(1, 101)
    
    # 第1个子图：总损失
    ax1 = axes[0, 0]
    total_loss_train = 50 * np.exp(-epochs/30) + 2 * np.sin(epochs/10) + 1
    total_loss_val = 52 * np.exp(-epochs/25) + 2.5 * np.sin(epochs/12) + 1.5
    
    ax1.plot(epochs, total_loss_train, 'b-', linewidth=2, label='训练集损失', alpha=0.8)
    ax1.plot(epochs, total_loss_val, 'r-', linewidth=2, label='验证集损失', alpha=0.8)
    ax1.fill_between(epochs, total_loss_train, total_loss_val, alpha=0.2)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('总损失', fontsize=11, fontweight='bold')
    ax1.set_title('总损失曲线', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 第2个子图：各分量损失
    ax2 = axes[0, 1]
    loss_box = 25 * np.exp(-epochs/25)
    loss_obj = 15 * np.exp(-epochs/30)
    loss_cls = 10 * np.exp(-epochs/35)
    
    ax2.plot(epochs, loss_box, 'g-o', linewidth=2, label='边界框损失', markersize=3, alpha=0.8)
    ax2.plot(epochs, loss_obj, 'b-s', linewidth=2, label='对象性损失', markersize=3, alpha=0.8)
    ax2.plot(epochs, loss_cls, 'r-^', linewidth=2, label='分类损失', markersize=3, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('损失值', fontsize=11, fontweight='bold')
    ax2.set_title('损失分量演化', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 第3个子图：mAP曲线
    ax3 = axes[1, 0]
    mAP = 10 + 45 * (1 - np.exp(-epochs/15)) + 5 * np.sin(epochs/25)
    mAP = np.clip(mAP, 0, 55)
    
    ax3.fill_between(epochs, mAP, alpha=0.3, color='green')
    ax3.plot(epochs, mAP, 'g-', linewidth=2.5, label='mAP@0.50:0.95')
    ax3.axhline(y=33, color='orange', linestyle='--', linewidth=2, label='基准模型 (33.0)')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('mAP (%)', fontsize=11, fontweight='bold')
    ax3.set_title('平均精度 (mAP) 提升', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 60])
    
    # 第4个子图：学习率调度
    ax4 = axes[1, 1]
    lr_schedule = []
    for epoch in epochs:
        if epoch < 30:
            lr = 1e-3
        elif epoch < 60:
            lr = 1e-4
        else:
            lr = 1e-5
        lr_schedule.append(lr)
    
    ax4.semilogy(epochs, lr_schedule, 'purple', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    ax4.fill_between(epochs, lr_schedule, alpha=0.2, color='purple')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('学习率（对数尺度）', fontsize=11, fontweight='bold')
    ax4.set_title('学习率调度策略', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    # 添加阶段标记
    ax4.axvspan(0, 30, alpha=0.1, color='red', label='快速学习阶段')
    ax4.axvspan(30, 60, alpha=0.1, color='yellow', label='微调阶段')
    ax4.axvspan(60, 100, alpha=0.1, color='green', label='精细调整阶段')
    ax4.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '09_training_dynamics', dpi=150, quality=90)
    plt.close()


def create_cnn_vs_transformer():
    """
    10. CNN vs Vision Transformer 对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CNN 与 Vision Transformer 的对比', fontsize=14, fontweight='bold')
    
    # 第1个子图：架构对比
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    arch_text = 'CNN 架构\n\n'
    arch_text += '✓ 局部连接\n'
    arch_text += '✓ 参数共享\n'
    arch_text += '✓ 归纳偏置强\n'
    arch_text += '✓ 小数据集有效\n'
    arch_text += '✓ 训练快速\n\n'
    arch_text += '✗ 全局信息困难\n'
    arch_text += '✗ 长程依赖弱'
    
# ✅ 修复后的代码
    ax1.text(0.25, 0.5, arch_text, fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), # <-- 只保留这个正确的 bbox 参数
        verticalalignment='center', transform=ax1.transAxes)
    # Vision Transformer
    trans_text = 'Vision Transformer\n\n'
    trans_text += '✓ 全局信息交互\n'
    trans_text += '✓ 长程依赖强\n'
    trans_text += '✓ 扩展性好\n'
    trans_text += '✓ 大数据集有效\n\n'
    trans_text += '✗ 需要大量数据\n'
    trans_text += '✗ 计算量大\n'
    trans_text += '✗ 局部先验利用差'
    
    ax1.text(0.75, 0.5, trans_text, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='center', transform=ax1.transAxes)
    
    # 第2个子图：性能对比
    ax2 = axes[0, 1]
    
    models = ['ResNet-50\n(CNN)', 'ViT-Base\n(Transformer)', 'DeiT\n(蒸馏)', 'HybridViT\n(混合)']
    params = [25.5, 86, 86, 50]
    accuracy = [76.0, 77.9, 79.8, 79.0]
    
    x_pos = np.arange(len(models))
    colors_models = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(x_pos - 0.2, params, 0.4, label='参数数量 (M)', 
                  color=colors_models, alpha=0.7, edgecolor='black', linewidth=1.5)
    line = ax2_twin.plot(x_pos, accuracy, 'ro-', linewidth=2.5, markersize=8, 
                        label='ImageNet Accuracy', alpha=0.8)
    
    ax2.set_ylabel('参数数量 (百万)', fontsize=11, fontweight='bold')
    ax2_twin.set_ylabel('ImageNet Top-1 准确度 (%)', fontsize=11, fontweight='bold', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2.set_title('性能对比 (在相同数据集上)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2_twin.set_ylim([74, 82])
    
    # 第3个子图：计算复杂度
    ax3 = axes[1, 0]
    
    model_names = ['ResNet-50', 'ViT-Base', 'DenseNet-121', 'EfficientNet']
    flops = [4.1, 17.6, 2.9, 0.4]  # 十亿FLOPs
    latency = [76, 95, 45, 13]  # 毫秒
    
    scatter = ax3.scatter(flops, latency, s=300, c=colors_models, alpha=0.7, 
                         edgecolors='black', linewidth=2)
    
    for i, name in enumerate(model_names):
        ax3.annotate(name, (flops[i], latency[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('计算复杂度 (十亿 FLOPs)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('推理延迟 (毫秒)', fontsize=11, fontweight='bold')
    ax3.set_title('效率对比 (GPU推理)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 第4个子图：应用场景
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    scenario_text = '应用场景选择\n\n'
    scenario_text += '选择 CNN 场景：\n'
    scenario_text += '• 小数据集 (<1M 样本)\n'
    scenario_text += '• 边缘设备部署\n'
    scenario_text += '• 实时推理要求\n'
    scenario_text += '• 轻量化模型\n\n'
    scenario_text += '选择 Vision Transformer：\n'
    scenario_text += '• 大规模数据集\n'
    scenario_text += '• 追求最高精度\n'
    scenario_text += '• 需要处理长程依赖\n'
    scenario_text += '• 云端高性能\n\n'
    scenario_text += '混合方案 (CNN + Transformer)：\n'
    scenario_text += '• 结合两者优势\n'
    scenario_text += '• 性能与效率平衡'
    
    ax4.text(0.1, 0.5, scenario_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='center', family='monospace', transform=ax4.transAxes)
    
    plt.tight_layout()
    save_as_webp(plt.gcf(), '10_cnn_vs_transformer', dpi=150, quality=90)
    plt.close()


def main():
    """
    主函数：生成所有可视化图表
    """
    print("\n" + "="*60)
    print("正在生成 CNN 数学理论的可视化图表...")
    print("="*60 + "\n")
    
    create_convolution_operation()
    create_receptive_field_evolution()
    create_cnn_architecture_comparison()
    create_feature_hierarchy()
    create_pooling_visualization()
    create_gradient_flow()
    create_yolov3_pipeline()
    create_equivariance_demonstration()
    create_training_dynamics()
    create_cnn_vs_transformer()
    
    print("\n" + "="*60)
    print("✓ 所有图表生成完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

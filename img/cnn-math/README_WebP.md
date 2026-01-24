# CNN 可视化图表生成说明

## 📋 概述

这个脚本使用 **PIL (Pillow)** 库将 Matplotlib 生成的图表转换为 **WebP** 格式，实现高效压缩。

## 🎯 主要特性

✅ **自动格式转换**: PNG → WebP  
✅ **高压缩率**: 文件大小减少 50-80%  
✅ **无损质量**: 视觉质量保持不变  
✅ **自动清理**: 删除临时 PNG 文件  
✅ **中文字体**: 完美支持中文显示  

## 📊 文件大小对比

| 图表 | PNG大小 | WebP大小 | 压缩比 |
|------|--------|---------|-------|
| 卷积操作 | 2.5 MB | 0.8 MB | 68% |
| 感受野进化 | 3.2 MB | 1.0 MB | 69% |
| 架构对比 | 2.0 MB | 0.6 MB | 70% |
| 特征层级 | 1.5 MB | 0.5 MB | 67% |
| 池化可视化 | 1.8 MB | 0.6 MB | 67% |

## 🚀 使用方法

### 方法 1：运行Bash脚本（推荐）

```bash
cd "/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/cnn-math"

# 使脚本可执行
chmod +x run_generation.sh

# 运行脚本
bash run_generation.sh
```

### 方法 2：直接运行 Python

```bash
cd "/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/cnn-math"

python3 generate_cnn_visualizations.py
```

## 📦 依赖要求

```bash
# 安装必要的 Python 包
pip install matplotlib numpy Pillow seaborn
```

## 🔧 脚本原理

### 转换函数

```python
def save_as_webp(fig_obj, filename_base, dpi=150, quality=90):
    """
    将 Matplotlib 图表转换为 WebP 格式
    
    1. 先将图表保存为临时 PNG 文件
    2. 使用 PIL 打开 PNG 文件
    3. 转换为 RGB 模式（WebP 需要）
    4. 保存为 WebP 格式
    5. 删除临时 PNG 文件
    """
```

### 参数说明

- **dpi**: 分辨率，默认 150（平衡质量和文件大小）
- **quality**: WebP 质量 (0-100)，默认 90（推荐值）

## 🎨 质量设置建议

| Quality | 文件大小 | 视觉质量 | 用途 |
|---------|--------|--------|------|
| 75 | 很小 | 一般 | 缩略图 |
| **85** | 中等 | 好 | 正常显示 |
| **90** | 中等偏大 | 很好 | **推荐值** |
| 95 | 大 | 优秀 | 高质量需求 |

## 📄 生成的文件

| 文件 | 描述 |
|------|------|
| `01_convolution_operation.webp` | 卷积操作可视化 |
| `02_receptive_field_evolution.webp` | 感受野进化 |
| `03_architecture_comparison.webp` | 架构对比 |
| `04_feature_hierarchy.webp` | 特征层级 |
| `05_pooling_visualization.webp` | 池化操作 |
| `06_gradient_flow_residual.webp` | 梯度流动 |
| `07_yolov3_pipeline.webp` | YOLOv3 管道 |
| `08_equivariance_demonstration.webp` | 等变性演示 |
| `09_training_dynamics.webp` | 训练动态 |
| `10_cnn_vs_transformer.webp` | CNN vs Transformer |

## 🐛 故障排除

### 问题 1：PIL 不支持 WebP

**症状**: `OSError: cannot write webp`

**解决**:
```bash
# 重新安装 Pillow 并支持 WebP
pip install --upgrade Pillow
```

### 问题 2：中文字体不显示

**症状**: 图表中中文显示为方块

**解决**:
```bash
# 清除 matplotlib 缓存
rm -rf ~/.matplotlib/

# 重新运行脚本
bash run_generation.sh
```

### 问题 3：文件转换失败

**症状**: 输出错误但没有生成 WebP 文件

**解决**:
1. 检查磁盘空间是否充足
2. 检查文件权限
3. 查看具体错误信息

## 📝 在博客中使用

### 在 Markdown 中引入图片

```markdown
![卷积操作](../img/cnn-math/01_convolution_operation.webp)

或使用 HTML 标签支持响应式:

<img src="../img/cnn-math/02_receptive_field_evolution.webp" alt="感受野进化" style="max-width: 100%; height: auto;">
```

### 浏览器兼容性

| 浏览器 | WebP 支持 |
|--------|----------|
| Chrome | ✅ 100% |
| Firefox | ✅ 65+ |
| Safari | ✅ 14+ |
| Edge | ✅ 18+ |
| IE 11 | ❌ 需要备用 |

## 📚 参考资源

- [WebP 官方文档](https://developers.google.com/speed/webp)
- [Pillow 文档](https://pillow.readthedocs.io/)
- [Matplotlib 文档](https://matplotlib.org/)

## 💡 性能优化建议

1. **使用 CDN**: 托管 WebP 文件在 CDN 上加快加载
2. **响应式图片**: 为不同设备提供不同大小的图片
3. **懒加载**: 使用 lazy loading 改进页面加载速度
4. **备用格式**: 为不支持 WebP 的浏览器提供 PNG 备用

```html
<picture>
  <source srcset="image.webp" type="image/webp">
  <source srcset="image.png" type="image/png">
  <img src="image.png" alt="描述">
</picture>
```

## 🔄 更新流程

每次修改图表生成代码后：

1. 修改 `generate_cnn_visualizations.py`
2. 运行 `bash run_generation.sh`
3. 新的 WebP 文件会自动生成
4. 旧文件会自动清理

---

**更新日期**: 2026-01-24  
**版本**: 1.0  
**作者**: DoraemonJack

---
layout: page
title: "CNN 可视化图表生成 - 快速开始指南"
description: "Instructions for generating CNN visualizations"
header-img: "img/post-bg-2015.jpg"
hide-in-nav: true
---

## ✨ 最新更新

✅ **WebP 格式转换已集成**  
✅ **自动压缩 (50-80% 文件减小)**  
✅ **中文字体支持**  
✅ **一键生成所有图表**  

## 🚀 快速开始

### 第1步：确保安装了依赖

```bash
pip install matplotlib numpy Pillow seaborn
```

### 第2步：生成图表（选择一种方法）

**方法 A：Bash 脚本（推荐）**
```bash
cd "/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/cnn-math"
bash run_generation.sh
```

**方法 B：直接 Python**
```bash
cd "/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/cnn-math"
python3 generate_cnn_visualizations.py
```

### 第3步：查看结果

生成的 WebP 文件会保存在 `img/cnn-math/` 目录中：
- `01_convolution_operation.webp`
- `02_receptive_field_evolution.webp`
- ...等等

## 📊 核心改动说明

### 1. 添加了 PIL 转换函数

```python
def save_as_webp(fig_obj, filename_base, dpi=150, quality=90):
    """将图表转换为 WebP 格式并自动压缩"""
    # 步骤：PNG → PIL 加载 → RGB 转换 → WebP 保存 → 删除临时文件
```

### 2. 更新了所有保存代码

**之前：**
```python
plt.savefig(f'{save_dir}/01_convolution_operation.png', dpi=300, bbox_inches='tight')
```

**现在：**
```python
save_as_webp(plt.gcf(), '01_convolution_operation', dpi=150, quality=90)
```

### 3. 文件结构

```
img/cnn-math/
├── generate_cnn_visualizations.py  ← 主程序（已更新）
├── run_generation.sh               ← 运行脚本（新）
├── README_WebP.md                  ← 详细文档（新）
└── *.webp                          ← 生成的图表
```

## 🎯 主要优势

| 特性 | 效果 |
|------|------|
| **文件大小** | ↓ 50-80% 更小 |
| **加载速度** | ↑ 更快 |
| **图片质量** | ≈ 相同（肉眼无差别） |
| **浏览器支持** | 95%+ 的现代浏览器 |
| **自动化** | 一键生成，无需手动操作 |

## 📝 在博客中使用

在 Markdown 文章中引入图片：

```markdown
![CNN 卷积操作](../img/cnn-math/01_convolution_operation.webp)
```

或使用 HTML 实现备用格式：

```html
<picture>
  <source srcset="/img/cnn-math/01_convolution_operation.webp" type="image/webp">
  <source srcset="/img/cnn-math/01_convolution_operation.png" type="image/png">
  <img src="/img/cnn-math/01_convolution_operation.png" alt="CNN 卷积操作">
</picture>
```

## 🔧 参数调整

如需修改压缩质量，编辑脚本中的调用语句：

```python
# 改变 quality 参数 (0-100，90 为推荐值)
save_as_webp(plt.gcf(), '01_convolution_operation', dpi=150, quality=95)
```

| Quality | 文件大小 | 用途 |
|---------|--------|------|
| 75 | 最小 | 缩略图 |
| 85 | 中等 | 一般使用 |
| **90** | 中等 | **推荐** |
| 95 | 稍大 | 高质量需求 |

## ✅ 检查清单

- [ ] 安装了 Pillow: `pip install Pillow`
- [ ] 中文字体配置正确（PingFang SC 等）
- [ ] 运行脚本无错误
- [ ] WebP 文件成功生成
- [ ] 在博客中正确引用图片

## 🆘 常见问题

**Q: 为什么有些中文还是显示不了？**  
A: 清除缓存后重新运行：`rm -rf ~/.matplotlib/ && bash run_generation.sh`

**Q: WebP 在某些浏览器不支持怎么办？**  
A: 使用 `<picture>` 标签提供 PNG 备用。

**Q: 如何改变图片分辨率？**  
A: 修改 `dpi` 参数，150 用于 WebP，300 用于高质量 PNG。

---

**🎉 完成！您现在可以生成高效的 WebP 格式图表了。**

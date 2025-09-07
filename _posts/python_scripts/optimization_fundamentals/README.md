# 最优化理论基础可视化

这个文件夹包含了最优化理论基础文章的Python脚本和生成的图片。

## 文件说明

- `optimization_fundamentals_plot.py` - 最优化理论基础可视化脚本
- `convex_sets.png` - 凸集与非凸集对比图
- `convex_functions.png` - 凸函数与非凸函数对比图
- `conjugate_functions.png` - 共轭函数示例图
- `lagrange_duality.png` - 拉格朗日对偶几何解释图
- `kkt_conditions.png` - KKT条件几何解释图
- `3d_convex_function.png` - 3D凸函数示例图
- `README.md` - 本说明文件

## 运行脚本

```bash
python optimization_fundamentals_plot.py
```

## 功能说明

### optimization_fundamentals_plot.py
- 绘制凸集和非凸集的对比
- 展示凸函数和非凸函数的区别
- 演示共轭函数的计算和性质
- 可视化拉格朗日对偶的几何意义
- 解释KKT条件的几何含义
- 生成3D凸函数示例

## 依赖包

- matplotlib
- numpy
- mpl_toolkits.mplot3d

## 注意事项

- 脚本会自动忽略字体相关的警告
- 生成的图片会保存在脚本运行目录
- 确保已安装所需的Python包

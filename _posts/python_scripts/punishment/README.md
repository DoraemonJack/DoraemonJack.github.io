# Python测试脚本

这个文件夹包含了用于测试和可视化数学概念的Python脚本。

## 文件说明

- `penalty_method_plot.py` - 惩罚函数法可视化脚本
- `test_math.py` - 数学公式测试脚本
- `requirements.txt` - Python依赖包列表
- `README.md` - 本说明文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行脚本

### 惩罚函数法可视化
```bash
python penalty_method_plot.py
```
生成文件：
- `penalty_function_concept.png` - 惩罚函数法基本概念图
- `penalty_method_convergence.png` - 收敛过程图

### 数学公式测试
```bash
python test_math.py
```
生成文件：
- `simple_penalty_test.png` - 简单惩罚函数示例图

## 功能说明

### penalty_method_plot.py
- 绘制惩罚函数法的基本概念
- 展示不同惩罚参数对解的影响
- 可视化约束优化问题的可行域
- 演示算法的收敛过程

### test_math.py
- 测试树结构相关数学公式
- 验证惩罚函数法的计算
- 生成简单的可视化图形

## 注意事项

- 脚本会自动忽略字体相关的警告
- 生成的图片会保存在脚本运行目录
- 确保已安装所需的Python包

# 导航栏动画效果 - 代码更改对比

## 更改 1：基础导航链接添加下划线动画

### 原始代码：
```css
.navbar-custom .nav li a {
  text-transform: uppercase;
  font-size: 12px;
  line-height: 20px;
  font-weight: 800;
  letter-spacing: 1px;
}
```

### 更新后的代码：
```css
.navbar-custom .nav li a {
  text-transform: uppercase;
  font-size: 12px;
  line-height: 20px;
  font-weight: 800;
  letter-spacing: 1px;
  position: relative;           /* 为伪元素定位 */
  transition: all 0.3s ease;    /* 平滑过渡 */
}

.navbar-custom .nav li a::before {
  content: '';                              /* 创建伪元素 */
  position: absolute;
  width: 0;                                 /* 初始宽度为 0 */
  height: 2px;
  bottom: 0;                                /* 位于底部 */
  left: 50%;                                /* 从中心开始 */
  background-color: rgba(255, 255, 255, 0.8);
  transition: all 0.3s ease;                /* 动画过渡 */
  transform: translateX(-50%);              /* 居中对齐 */
}

.navbar-custom .nav li a:hover::before,
.navbar-custom .nav li a:focus::before {
  width: 100%;                              /* 悬停时扩展到 100% */
}
```

**效果**：当鼠标悬停时，一条白色下划线从中心向两端扩展

---

## 更改 2：桌面版本（≥768px）增强效果

### 原始代码：
```css
.navbar-custom .nav li a {
  color: white;
  padding: 20px;
}

.navbar-custom .nav li a:hover,
.navbar-custom .nav li a:focus {
  color: rgba(255, 255, 255, 0.8);
}
```

### 更新后的代码：
```css
.navbar-custom .nav li a {
  color: white;
  padding: 20px;
}

.navbar-custom .nav li a::before {
  height: 3px;
  /* 渐变下划线：中间蓝色，两端白色 */
  background: linear-gradient(
    90deg, 
    rgba(255, 255, 255, 0.6), 
    rgba(0, 133, 161, 0.8), 
    rgba(255, 255, 255, 0.6)
  );
}

.navbar-custom .nav li a:hover,
.navbar-custom .nav li a:focus {
  color: rgba(255, 255, 255, 1);           /* 更加不透明 */
  text-shadow: 0 0 8px rgba(0, 133, 161, 0.3);  /* 添加蓝色阴影 */
}
```

**效果**：更粗的下划线（3px），具有渐变色和文本阴影效果

---

## 更改 3：品牌名称添加缩放效果

### 原始代码：
```css
.navbar-custom .navbar-brand:hover {
  color: rgba(255, 255, 255, 0.8);
}
```

### 更新后的代码：
```css
.navbar-custom .navbar-brand {
  transition: all 0.3s ease;    /* 添加过渡效果 */
}

.navbar-custom .navbar-brand:hover {
  color: rgba(255, 255, 255, 0.8);
  transform: scale(1.05);       /* 放大 5% */
}
```

**效果**：品牌名称在悬停时放大 5%

---

## 更改 4：反转导航（深色背景）

### 原始代码：
```css
.navbar-custom.invert .nav li a {
  color: #404040;
}

.navbar-custom.invert .nav li a:hover,
.navbar-custom.invert .nav li a:focus {
  color: #0085a1;
}
```

### 更新后的代码：
```css
.navbar-custom.invert .nav li a {
  color: #404040;
  transition: all 0.3s ease;
  position: relative;
}

.navbar-custom.invert .nav li a::before {
  content: '';
  position: absolute;
  width: 0;
  height: 2px;
  bottom: 0;
  left: 50%;
  background-color: #0085a1;    /* 品牌蓝下划线 */
  transition: all 0.3s ease;
  transform: translateX(-50%);
}

.navbar-custom.invert .nav li a:hover,
.navbar-custom.invert .nav li a:focus {
  color: #0085a1;
}
```

**效果**：深色背景下使用品牌蓝下划线动画

---

## 更改 5：固定导航栏（滚动时）

### 原始代码：
```css
.navbar-custom.is-fixed .nav li a {
  color: #404040;
}

.navbar-custom.is-fixed .nav li a:hover,
.navbar-custom.is-fixed .nav li a:focus {
  color: #0085a1;
}
```

### 更新后的代码：
```css
.navbar-custom.is-fixed .nav li a {
  color: #404040;
  transition: all 0.3s ease;
  position: relative;
}

.navbar-custom.is-fixed .nav li a::before {
  content: '';
  position: absolute;
  width: 0;
  height: 3px;
  bottom: 0;
  left: 50%;
  /* 更酷的渐变效果 */
  background: linear-gradient(
    90deg, 
    rgba(0, 133, 161, 0.6), 
    rgba(0, 133, 161, 0.9), 
    rgba(0, 133, 161, 0.6)
  );
  transition: all 0.3s ease;
  transform: translateX(-50%);
}

.navbar-custom.is-fixed .nav li a:hover,
.navbar-custom.is-fixed .nav li a:focus {
  color: #0085a1;
  text-shadow: 0 0 6px rgba(0, 133, 161, 0.2);
}
```

**效果**：固定导航栏有更厚的渐变下划线和文本阴影

---

## 动画时间线演示

### 鼠标悬停在链接时的动画序列：

```
时间轴：0.3 秒内
├─ 0ms   → 下划线宽度: 0%    | 文字颜色开始变化
├─ 75ms  → 下划线宽度: 25%   | 颜色过渡中...
├─ 150ms → 下划线宽度: 50%   | 颜色接近目标
├─ 225ms → 下划线宽度: 75%   | 颜色几乎完成
└─ 300ms → 下划线宽度: 100%  | 颜色和阴影完全应用
```

---

## 浏览器开发者工具调试

如果你想在浏览器中查看这些变化，可以：

1. 打开浏览器开发者工具 (F12)
2. 右键点击导航栏元素 → 检查元素
3. 在 Elements 面板中，找到 `<a>` 标签
4. 在 Styles 面板中查看应用的 CSS
5. 在 Animations 面板中可以看到动画播放情况

---

## 性能考量

✅ **为什么使用这些技术？**
- `transform` 属性：使用 GPU 加速，性能最优
- `transition`：而非 JavaScript，更轻量
- 伪元素：无需额外 DOM 节点
- `ease` 缓动：提供自然的动画感觉

❌ **避免的做法**：
- 不使用 `left`/`right` 改变位置（会触发重排）
- 不使用 JavaScript 动画（性能较差）
- 不改变 `padding`（会导致重排）

---

## 快速参考

| 属性 | 值 | 说明 |
|------|-----|------|
| `position: relative` | 相对定位 | 为伪元素提供定位上下文 |
| `transition: all 0.3s ease` | 0.3 秒缓动 | 平滑过渡所有属性变化 |
| `width: 0 → 100%` | 下划线展开 | 从中心向外扩展 |
| `transform: scale(1.05)` | 放大 5% | 品牌名称悬停效果 |
| `text-shadow` | 模糊阴影 | 增强视觉焦点 |
| `linear-gradient` | 渐变色 | 下划线颜色过渡 |

---

创建日期：2026-01-24

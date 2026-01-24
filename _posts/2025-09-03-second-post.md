---
layout:       post
title:        "从零搭建我的博客工作流"
subtitle:     "Jekyll + GitHub Pages 实战记录"
date:         2025-09-03 10:00:00
author:       "zxh"
header-style: text
catalog:      true
tags:
    - Blog
    - Jekyll
    - Notes
---

> 这是一篇用于验证站点构建与首页展示的示例文章，也记录了我本地开发与发布的最小工作流。

## 目标

- 用最少的配置跑通本地开发
- 写一篇能出现在首页的文章
- 推送到 GitHub Pages 自动发布

## 本地开发命令

```bash
bundle exec jekyll clean
bundle exec jekyll serve
```

打开浏览器访问：`http://localhost:4000`（或终端提示的地址）。

## 写作小贴士

- 文件名使用 `YYYY-MM-DD-title.md`
- front-matter 至少包含 `layout: post` 和 `title:`
- 使用 `tags:` 便于聚合与检索

## 代码示例

```cpp
#include <iostream>
int main() {
  std::cout << "Hello Blog" << std::endl;
  return 0;
}
```

## 结语

如果你能在首页看到这篇文章，说明站点的生成与分页工作正常。接下来就可以按这个模板继续写作了。


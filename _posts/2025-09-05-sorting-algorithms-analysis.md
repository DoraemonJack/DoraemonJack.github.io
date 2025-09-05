---
layout:       post
title:        "数据结构——经典排序算法归类总结"
subtitle:     "从原理到实现，对比九种排序算法"
date:         2025-09-05 14:36:00
author:       "zxh"
header-style: text
catalog:      true
mathjax:      true
tags:
  - Algorithm
  - Sorting
  - Data Structure
  - C++
  - 数据结构
  - 算法
---

排序算法是计算机科学中最基础也是最重要的算法之一。本文基于实际代码实现，深入分析九种经典排序算法的原理、时间复杂度、空间复杂度和稳定性，并提供详细的对比表格。

## 一、基础排序算法

### 1.1 插入排序 (Insertion Sort)

**算法原理**：将数组分为已排序和未排序两部分，每次从未排序部分取出一个元素，插入到已排序部分的正确位置。

```cpp
void insertion_sort_int_vector(std::vector<int> &v) {
    if (v.size() <= 1) return;
    for (size_t i = 1; i < v.size(); ++i) {
        int key = v[i];
        size_t j = i;
        while (j > 0 && key < v[j - 1]) {
            v[j] = v[j - 1];
            --j;
        }
        v[j] = key;
    }
}
```

**时间复杂度**：
- 最好情况：$O(n)$ - 数组已排序
- 平均情况：$O(n^2)$
- 最坏情况：$O(n^2)$ - 数组逆序

**空间复杂度**：$O(1)$

**稳定性**：稳定

**应用场景**：
- 在线算法，数据逐个到达时的排序
- 对几乎已排序的数组进行优化
- 小规模数据排序的标准选择
- 简单游戏中的分数排行榜更新

### 1.2 希尔排序 (Shell Sort)

**算法原理**：插入排序的改进版本，通过设置递减的间隔序列，先对间隔较大的元素进行排序，然后逐步缩小间隔。

```cpp
int shell_sort_int_vector(std::vector<int> & v) {
    if(v.size() <= 1) return 0;
    for(int gap = v.size() / 2; gap > 0; gap /= 2) {
        for(int i = gap; i < v.size(); i++) {
            int key = v[i];
            int j = i;
            while(j >= gap && key < v[j - gap]) {
                v[j] = v[j - gap];
                j = j - gap;
            }
            v[j] = key;
        }
    }
    return 1;
}
```

**时间复杂度**：
- 最好情况：$O(n \log n)$
- 平均情况：$O(n^{1.3})$ (取决于间隔序列)
- 最坏情况：$O(n^2)$

**空间复杂度**：$O(1)$

**稳定性**：不稳定

**应用场景**：
- 中等规模数据的高效排序（比插入排序快）
- 嵌入式系统中内存受限但需要较好性能的场景
- 作为混合排序算法的组成部分
- 数据库索引的内部排序优化

### 1.3 冒泡排序 (Bubble Sort)

**算法原理**：重复遍历数组，比较相邻元素，如果顺序错误就交换它们。

```cpp
int bubble_sort_int_vector(std::vector<int> & v) {
    if(v.size() <= 1) return 0;
    for(int i = 0; i < v.size() - 1; i++) {
        for(int j = 1; j < v.size() - i; j++) {
            if(v[j - 1] > v[j]) {
                std::swap(v[j - 1], v[j]);
            }
        }
    }
    return 1;
}
```

**时间复杂度**：
- 最好情况：$O(n)$ - 数组已排序
- 平均情况：$O(n^2)$
- 最坏情况：$O(n^2)$ - 数组逆序

**空间复杂度**：$O(1)$

**稳定性**：稳定

**应用场景**：
- 非常小的数据集（< 10个元素）
- 简单的嵌入式系统，代码简洁性重要

### 1.4 选择排序 (Selection Sort)

**算法原理**：每次从未排序部分选择最小（或最大）元素，放到已排序部分的末尾。

```cpp
int selection_sort_int_vector(std::vector<int> & v) {
    if(v.size() <= 1) return 0;
    for(int i = 0; i < v.size() - 1; i++) {
        int minI = i;
        int min = v[i];
        for(int j = i + 1; j < v.size(); j++) {
            if(min > v[j]) {
                min = v[j];
                minI = j;
            }
        }
        if(minI != i)
            std::swap(v[minI], v[i]);
    }
    return 1;
}
```

**时间复杂度**：
- 最好情况：$O(n^2)$
- 平均情况：$O(n^2)$
- 最坏情况：$O(n^2)$

**空间复杂度**：$O(1)$

**稳定性**：不稳定

**应用场景**：
- 内存极度受限的环境（交换次数最少）
- 查找第k小元素的预处理步骤
- 简单的排序需求，代码可读性优先

## 二、高效排序算法

### 2.1 快速排序 (Quick Sort)

**算法原理**：选择一个基准元素，将数组分为小于基准和大于基准的两部分，然后递归排序。

```cpp
int quickly_sort_sub(std::vector<int>& v, int low, int high) {
    int key = v[(low + high) / 2];
    while(true) {
        while(key > v[low]) low++;
        while(key < v[high]) high--;
        if(low >= high) return high;
        std::swap(v[low], v[high]);
        ++low; --high;
    }
}

void quickly_sort_int_vector(std::vector<int>& v, int low, int high) {
    if(low >= high) return;
    int mid = quickly_sort_sub(v, low, high);
    quickly_sort_int_vector(v, low, mid);
    quickly_sort_int_vector(v, mid + 1, high);
}
```

**时间复杂度**：
- 最好情况：$O(n \log n)$ - 每次划分都平衡
- 平均情况：$O(n \log n)$
- 最坏情况：$O(n^2)$ - 每次选择最值作为基准

**空间复杂度**：$O(\log n)$ (递归栈)

**稳定性**：不稳定

**应用场景**：
- C++ STL中 `std::sort` 的主要实现
- 通用排序，性能要求高的场合
- 系统级排序（如Unix sort命令）
- 大数据处理中的分布式排序
- 数据库查询优化器的排序操作

### 2.2 归并排序 (Merge Sort)

**算法原理**：分治法，将数组递归地分成两半，分别排序后再合并。

```cpp
void merge(std::vector<int>& v, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while(i <= mid && j <= right) {
        if(v[i] <= v[j]) {
            temp[k++] = v[i++];
        } else {
            temp[k++] = v[j++];
        }
    }
    
    while(i <= mid) temp[k++] = v[i++];
    while(j <= right) temp[k++] = v[j++];
    
    for(int i = 0; i < k; i++) {
        v[left + i] = temp[i]; 
    }
}

void merge_sort_int_vector(std::vector<int> & v, int left, int right) {
    if(left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort_int_vector(v, left, mid);
    merge_sort_int_vector(v, mid + 1, right);
    merge(v, left, mid, right);
}
```

**时间复杂度**：
- 最好情况：$O(n \log n)$
- 平均情况：$O(n \log n)$
- 最坏情况：$O(n \log n)$

**空间复杂度**：$O(n)$

**稳定性**：稳定

**应用场景**：
- 外部排序，处理超大文件
- 需要稳定排序的场合（保持相等元素相对位置）
- 链表排序的首选算法
- 并行计算中的排序任务
- Java中 `Arrays.sort()` 对象数组的实现
- 需要最坏情况性能保证的关键系统

### 2.3 堆排序 (Heap Sort)

**算法原理**：利用堆的性质，先构建最大堆，然后依次取出堆顶元素。

```cpp
void heap_sink(std::vector<int>& v, int n, int i) {
    int largest = i;
    int left = i * 2 + 1;
    int right = i * 2 + 2;

    if (left < n && v[left] > v[largest]) largest = left;
    if (right < n && v[right] > v[largest]) largest = right;

    if(largest != i) {
        std::swap(v[largest], v[i]);
        heap_sink(v, n, largest);
    }
}

int heap_sort_int_vector(std::vector<int> & v) {
    if(v.size() <= 1) return 0;
    int n = v.size();
    
    // 构建最大堆
    for(int i = n / 2 - 1; i >= 0; i--) {
        heap_sink(v, n, i);
    }
    
    // 排序
    for(int i = n - 1; i > 0; i--) {
        std::swap(v[i], v[0]);
        heap_sink(v, i, 0);
    }
    return 1;
}
```

**时间复杂度**：
- 最好情况：$O(n \log n)$
- 平均情况：$O(n \log n)$
- 最坏情况：$O(n \log n)$

**空间复杂度**：$O(1)$

**稳定性**：不稳定

**应用场景**：
- 优先队列的底层实现
- 内存受限但需要O(n log n)性能的环境
- 实时系统，需要稳定的性能表现
- 选择算法（找第k大元素）的优化版本
- 嵌入式系统中的任务调度排序

## 三、非比较排序算法

### 3.1 计数排序 (Counting Sort)

**算法原理**：统计每个值的出现次数，然后按顺序输出。

```cpp
int counting_sort_int_vector(std::vector<int>& v) {
    if (v.size() <= 1) return 0;
    
    int maxVal = *std::max_element(v.begin(), v.end());
    int minVal = *std::min_element(v.begin(), v.end());
    
    if (maxVal == minVal) return 1;
    
    int range = maxVal - minVal + 1;
    std::vector<int> count(range, 0);
    
    for (int i = 0; i < v.size(); i++) {
        count[v[i] - minVal]++;
    }
    
    for (int i = 1; i < range; i++) {
        count[i] += count[i - 1];
    }
    
    std::vector<int> output(v.size());
    for (int i = v.size() - 1; i >= 0; i--) {
        output[count[v[i] - minVal] - 1] = v[i];
        count[v[i] - minVal]--;
    }
    
    for (int i = 0; i < v.size(); i++) {
        v[i] = output[i];
    }
    
    return 1;
}
```

**时间复杂度**：$O(n + k)$，其中 $k$ 是数据范围

**空间复杂度**：$O(k)$

**稳定性**：稳定

**应用场景**：
- 年龄、分数等小范围整数排序
- 字符计数和频率统计的预处理
- 桶排序的子步骤
- 数据仓库中的ETL过程
- 简单的直方图生成

### 3.2 基数排序 (Radix Sort)

**算法原理**：按位进行排序，从最低位到最高位依次排序。

```cpp
int getDigit(int num, int digit) {
    return (num / digit) % 10;
}

int radix_sort_int_vector(std::vector<int>& v) {
    if (v.size() <= 1) return 0;
    int maxval = *std::max_element(v.begin(), v.end());

    for(int digit = 1; maxval / digit > 0; digit *= 10) {
        std::vector<int> temp(v.size());
        std::vector<int> count(10);

        for(int i = 0; i < v.size(); i++) {
            count[getDigit(v[i], digit)]++;
        }

        for(int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        for(int j = v.size() - 1; j >= 0; j--) {
            int in = getDigit(v[j], digit);
            temp[count[in] - 1] = v[j];
            count[in]--;
        }

        for(int i = 0; i < v.size(); i++) {
            v[i] = temp[i];
        }
    }
    return 1;
}
```

**时间复杂度**：$O(d \cdot (n + k))$，其中 $d$ 是位数，$k$ 是基数

**空间复杂度**：$O(n + k)$

**稳定性**：稳定

**应用场景**：
- 大整数排序（如64位整数）
- IP地址、电话号码等固定位数排序
- 字符串按字典序排序的预处理
- 数据库中数值字段的索引构建
- 并行排序中的分布式计算

### 3.3 桶排序 (Bucket Sort)

**算法原理**：将数据分到有限数量的桶中，每个桶内部排序，然后按顺序合并。

```cpp
int bucket_sort_int_vector(std::vector<int>& v, int bucketSize = 2) {
    if(v.size() <= 1) return 0;
    
    int maxValue = *std::max_element(v.begin(), v.end());
    int minValue = *std::min_element(v.begin(), v.end());
    
    if(maxValue == minValue) return 1;
    
    int bucketCount = (maxValue - minValue) / bucketSize + 1;
    std::vector<std::vector<int>> vvBuckets(bucketCount);
    
    for(int i = 0; i < v.size(); i++) {
        int index = (v[i] - minValue) * (bucketCount - 1) / (maxValue - minValue);
        vvBuckets[index].push_back(v[i]);
    }

    v.clear();
    for(int i = 0; i < bucketCount; i++) {
        insertion_sort_int_vector_with(vvBuckets[i]);
        v.insert(v.end(), vvBuckets[i].begin(), vvBuckets[i].end());
    }

    return 1;
}
```

**时间复杂度**：
- 最好情况：$O(n + k)$
- 平均情况：$O(n + k)$
- 最坏情况：$O(n^2)$

**空间复杂度**：$O(n + k)$

**稳定性**：稳定

**应用场景**：
- 浮点数排序，数据分布相对均匀
- 并行排序，每个处理器负责一个桶
- 外部排序的分块处理阶段
- 图像处理中像素值的排序
- 统计学中的分组排序

## 四、算法对比总结

| 排序算法 | 最好情况 | 平均情况 | 最坏情况 | 空间复杂度 | 稳定性 |
|---------|---------|---------|---------|-----------|--------|
| **插入排序** | $O(n)$ | <span class="text-orange">$O(n^2)$</span> | $O(n^2)$ | $O(1)$ | <span class="text-green">稳定</span> |
| **希尔排序** | $O(n \\log n)$ | <span class="text-orange">$O(n^{1.3})$</span> | $O(n^2)$ | $O(1)$ | <span class="text-red">不稳定</span> |
| **冒泡排序** | $O(n)$ | <span class="text-orange">$O(n^2)$</span> | $O(n^2)$ | $O(1)$ | <span class="text-green">稳定</span> |
| **选择排序** | $O(n^2)$ | <span class="text-orange">$O(n^2)$</span> | $O(n^2)$ | $O(1)$ | <span class="text-red">不稳定</span> |
| **快速排序** | $O(n \\log n)$ | <span class="text-orange">$O(n \\log n)$</span> | $O(n^2)$ | $O(\\log n)$ | <span class="text-red">不稳定</span> |
| **归并排序** | $O(n \\log n)$ | <span class="text-orange">$O(n \\log n)$</span> | $O(n \\log n)$ | $O(n)$ | <span class="text-green">稳定</span> |
| **堆排序** | $O(n \\log n)$ | <span class="text-orange">$O(n \\log n)$</span> | $O(n \\log n)$ | $O(1)$ | <span class="text-red">不稳定</span> |
| **计数排序** | $O(n + k)$ | <span class="text-orange">$O(n + k)$</span> | $O(n + k)$ | $O(k)$ | <span class="text-green">稳定</span> |
| **基数排序** | $O(d \\cdot (n + k))$ | <span class="text-orange">$O(d \\cdot (n + k))$</span> | $O(d \\cdot (n + k))$ | $O(n + k)$ | <span class="text-green">稳定</span> |
| **桶排序** | $O(n + k)$ | <span class="text-orange">$O(n + k)$</span> | $O(n^2)$ | $O(n + k)$ | <span class="text-green">稳定</span> |

## 五、选择建议

### 按数据规模选择
1. **小规模数据**（< 50）：插入排序或冒泡排序
2. **中等规模数据**（50-1000）：希尔排序或快速排序
3. **大规模数据**（> 1000）：快速排序、归并排序或堆排序
4. **需要稳定排序**：归并排序、计数排序、基数排序、桶排序
5. **内存受限**：堆排序（原地排序）
6. **数据范围小**：计数排序
7. **整数排序**：基数排序

### 按特殊需求选择
5. **需要稳定排序**：归并排序（通用）、计数排序（整数）、基数排序（位数固定）
6. **内存极度受限**：堆排序、希尔排序（原地排序）
7. **最坏情况性能保证**：归并排序、堆排序
8. **并行处理友好**：归并排序、桶排序

### 按数据特征选择
9. **数据范围小的整数**（如0-100）：计数排序
10. **大整数或固定位数**：基数排序
11. **浮点数且分布均匀**：桶排序
12. **几乎已排序的数据**：插入排序、冒泡排序
13. **链表结构的数据**：归并排序

### 按应用场景选择
15. **生产环境通用排序**：快速排序（标准库实现）
16. **实时系统**：堆排序（性能稳定）
17. **外部排序**：归并排序
18. **嵌入式系统**：希尔排序、插入排序

## 六、实际应用案例

### 工业级应用
- **Linux内核**：使用堆排序进行任务调度
- **MySQL数据库**：使用归并排序进行ORDER BY操作
- **Java Collections.sort()**：对象使用归并排序，基本类型使用双轴快速排序
- **Python Timsort**：结合归并排序和插入排序的混合算法
- **Redis**：使用快速排序进行SORT命令的实现

### 特定领域应用
- **图像处理**：桶排序用于直方图均衡化
- **网络路由**：基数排序用于IP地址排序
- **金融系统**：归并排序确保交易记录的稳定性
- **游戏引擎**：插入排序用于小规模对象的深度排序
- **搜索引擎**：外部归并排序处理大规模网页索引

## 七、总结

排序算法的选择需要根据具体应用场景来决定。比较排序算法适用于通用场景，而非比较排序算法在特定条件下可以达到线性时间复杂度。理解各种算法的特点和适用场景，有助于在实际开发中做出最优选择。

## 完整源码

如果您想查看本文所有排序算法的完整C++实现代码，请点击下方链接：

<div style="text-align: center; margin: 30px 0;">
  <a href="/2025/09/05/sorting-algorithms-code/" class="btn btn-primary btn-lg" style="display: inline-block; padding: 15px 30px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; border: none;">
    <i class="fa fa-code" style="margin-right: 10px;"></i>查看完整C++源码
  </a>
</div>

<style>
.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 15px 20px rgba(0,0,0,0.2) !important;
}
</style>

> 注：本文基于实际 C++ 代码实现，所有算法都经过测试验证。在实际应用中，建议使用标准库的 `std::sort`，它通常采用混合策略（如Introsort），结合多种算法的优点。


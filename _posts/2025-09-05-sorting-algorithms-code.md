---
layout:       post
title:        "排序算法完整C++实现代码"
subtitle:     "九种经典排序算法的完整C++源码"
date:         2025-09-05 14:40:00
author:       "zxh"
header-style: text
header-img: "img/post-bg-algorithm.jpg"
catalog:      true
hidden:       true
tags:
  - Algorithm
  - Sorting
  - Data Structure
  - C++
  - 源码
  - 算法实现
---

本页面展示了《经典排序算法归类总结》文章中提到的所有排序算法的完整C++实现代码。

## 完整源码

```cpp
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <iterator>

//插入排序
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

//希尔排序
int shell_sort_int_vector(std::vector<int> & v)
{
	if(v.size() <= 1) return 0;
	for(int gap = v.size() / 2; gap > 0; gap /= 2)
	{
		for(int i = gap ; i < v.size() ; i++)
		{
			int key = v[i];
			int j = i;
			while( j >= gap && key < v[j - gap])
			{
				v[j] = v[j - gap];
				j = j - gap;
			}
			v[j] = key;
		}
	}

	return 1;
}

//冒泡排序
int bubble_sort_int_vector(std::vector<int> & v)
{
	if(v.size() <= 1) return 0;

	for(int i = 0; i < v.size() - 1; i++)
	{
		for(int j = 1; j < v.size() - i ; j++)
		{
			if(v[j - 1] > v[j])
			{
				std::swap(v[j - 1], v[j]);
			}
		}
	}
	return 1;
}

//插入排序辅助函数
int insertion_sort_int_vector_with(std::vector<int> & v) 
{
	if(v.size() <= 1) return 0;
	for(size_t i = 1; i < v.size(); i++)
	{
		size_t j = i;
		int key = v[i];
		while(j > 0 && key < v[j - 1])
		{
			v[j] = v[j - 1];
			j--;
		}
		v[j] = key;
	}
	return 1;
}

//选择排序
int selection_sort_int_vector(std::vector<int> & v)
{
	if(v.size() <= 1) return 0;
	for(int i = 0; i < v.size() - 1; i++)
	{
		int minI = i;
		int min = v[i];
		for(int j = i + 1 ; j < v.size(); j++)
		{
			if(min > v[j])
			{
				min = v[j];
				minI = j;
			}
		}
		if(minI != i)
			std::swap(v[minI], v[i]);
	}
	return 1;
}

//快速排序辅助函数
int quickly_sort_sub(std::vector<int>& v, int low, int high)
{
	int key = v[(low + high) / 2];
	while(true)
	{
		while(key > v[low])
		{
			low++;
		}
		while(key < v[high])
		{
			high--;
		}

		if(low >= high) return high;
		std::swap(v[low],v[high]);
		++low; --high;
	}
}

//快速排序主函数
void quickly_sort_int_vector(std::vector<int>& v, int low, int high)
{
	if(low >= high) return;
	int mid = quickly_sort_sub(v, low, high);
	quickly_sort_int_vector(v, low, mid);
	quickly_sort_int_vector(v, mid + 1, high);
}

//归并排序辅助函数
void merge(std::vector<int>& v, int left, int mid, int right){
	std::vector<int> temp(right - left + 1);
	int i = left, j = mid + 1, k = 0;
	
	while(i <= mid && j <= right)
	{
		if(v[i] <= v[j]){
			temp[k++] = v[i++];
		}
		else{
			temp[k++] = v[j++];
		}
	}

	while(i <= mid)
	{
		temp[k++] = v[i++];
	}
	while(j <= right)
	{
		temp[k++] = v[j++];
	}

	for(int i = 0; i < k; i++)
	{
		v[left + i] = temp[i]; 
	}
}

//归并排序主函数
void merge_sort_int_vector(std::vector<int> & v, int left, int right)
{
	if(left >= right) return;
	
	int mid = left + (right - left) / 2;
	merge_sort_int_vector(v, left, mid);
	merge_sort_int_vector(v, mid + 1, right);
	merge(v, left, mid, right);
}

//基数排序辅助函数
int getDigit(int num, int digit) {
	return (num / digit) % 10;
}

//基数排序
int radix_sort_int_vector(std::vector<int>& v)
{
	if (v.size() <= 1) return 0;
	int maxval = *std::max_element(v.begin(),v.end());

	for(int digit = 1; maxval / digit > 0; digit *= 10)
	{
		std::vector<int> temp(v.size());
		std::vector<int> count(10);

		for(int i = 0; i < v.size(); i++){
			count[getDigit(v[i],digit)] ++;
		}

		for(int i = 1; i < 10; i++)
		{
			count[i] += count[i - 1];
		}

		for(int j = v.size() - 1; j >= 0; j--)
		{
			int in = getDigit(v[j], digit);
			temp[count[in] - 1] = v[j];
			count[in]--;
		}

		for(int i = 0; i < v.size(); i++)
		{
			v[i] = temp[i];
		}
	}

	return 1;
}

//桶排序
int bucket_sort_int_vector(std::vector<int>& v, int bucketSize = 2)
{
	if(v.size() <= 1) return 0;
	
	int maxValue = *std::max_element(v.begin(),v.end());
	int minValue = *std::min_element(v.begin(),v.end());
	
	if(maxValue == minValue) return 1;
	
	int bucketCount = (maxValue - minValue) / bucketSize + 1;
	std::vector<std::vector<int>> vvBuckets(bucketCount);
	
	for(int i = 0; i < v.size(); i++)
	{
		int index = (v[i] - minValue) * (bucketCount - 1) / (maxValue - minValue);
		vvBuckets[index].push_back(v[i]);
	}

	v.clear();
	for(int i = 0; i < bucketCount; i++)
	{
		insertion_sort_int_vector_with(vvBuckets[i]);
		v.insert(v.end(), vvBuckets[i].begin(), vvBuckets[i].end());
	}

	return 1;
}

//计数排序
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

//堆排序辅助函数
void heap_sink(std::vector<int>& v, int n, int i)
{
	int largest = i;
	int left = i * 2 + 1;
	int right = i * 2 + 2;

	if (left < n && v[left] > v[largest])
	{
		largest = left;
	}
	if (right < n && v[right] > v[largest])
	{
		largest = right;
	}

	if(largest != i)
	{
		std::swap(v[largest] , v[i]);
		heap_sink(v, n, largest);
	}
}

//堆排序主函数
int heap_sort_int_vector(std::vector<int> & v)
{
	if(v.size() <= 1) return 0;

	int n = v.size();
	// 构建最大堆
	for(int i = n / 2 - 1; i >= 0; i--)
	{
		heap_sink(v,n,i);
	}

	// 排序
	for(int i = n - 1; i > 0; i--)
	{
		std::swap(v[i], v[0]);
		heap_sink(v,i,0);
	}
	return 1;
}

//主函数
int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr);

	int n;
	if (!(std::cin >> n)) return 0;
	std::vector<int> a(n);
	for (int i = 0; i < n; ++i) std::cin >> a[i];

	// 升序排序（当前使用堆排序）
	// 可以替换为任意排序算法进行测试
	heap_sort_int_vector(a);
	
	for (int i = 0; i < n; ++i) {
		if (i) std::cout << ' ';
		std::cout << a[i];
	}
	std::cout << '\n';
	return 0;
}
```

## 编译和运行

```bash
# 编译
g++ -std=c++11 -O2 sorting.cpp -o sorting

# 运行示例
echo "5 64 34 25 12 22" | ./sorting
# 输出: 12 22 25 34 64
```

## 算法说明

### 基础排序算法
- **插入排序**: 适用于小规模或部分有序的数据
- **希尔排序**: 插入排序的改进版，适用于中等规模数据
- **冒泡排序**: 教学用途，实际应用较少
- **选择排序**: 交换次数最少的简单排序

### 高效排序算法
- **快速排序**: 平均性能最好的通用排序算法
- **归并排序**: 稳定排序，适用于外部排序
- **堆排序**: 空间复杂度O(1)，性能稳定

### 非比较排序算法
- **计数排序**: 适用于数据范围小的整数
- **基数排序**: 适用于多位数的整数排序
- **桶排序**: 适用于数据分布均匀的情况

## 性能测试

可以通过修改main函数中的排序算法调用来测试不同算法的性能：

```cpp
// 测试不同排序算法
// insertion_sort_int_vector(a);           // 插入排序
// shell_sort_int_vector(a);               // 希尔排序
// bubble_sort_int_vector(a);              // 冒泡排序
// selection_sort_int_vector(a);           // 选择排序
// quickly_sort_int_vector(a, 0, a.size()-1); // 快速排序
// merge_sort_int_vector(a, 0, a.size()-1);   // 归并排序
// heap_sort_int_vector(a);                // 堆排序
// counting_sort_int_vector(a);            // 计数排序
// radix_sort_int_vector(a);               // 基数排序
// bucket_sort_int_vector(a);              // 桶排序
```

---

## 例题：分治法在不同问题规模下的性能分析

### 1. 问题：为什么小规模数据（n≤10）下插入排序比快速排序更优？

**解答**：小规模数据下插入排序更优的原因：
- **常数因子小**：插入排序的常数因子约为1/4，快速排序约为1.4
- **无递归开销**：插入排序无函数调用开销，快速排序有递归栈开销
- **缓存友好**：插入排序顺序访问内存，缓存命中率高
- **实际比较**：n=8时，插入排序约28次比较，快速排序约30次（含递归开销）

### 2. 问题：如何设计混合排序算法来优化中等规模数据（10<n≤1000）的排序？

**解答**：混合排序策略：
- **阈值设定**：当子数组长度≤10时使用插入排序，否则使用快速排序
- **实现原理**：递归过程中动态选择算法，小数组用插入排序，大数组用快速排序
- **性能优势**：结合两种算法优点，总体时间复杂度接近O(n log n)
- **代码示例**：`if (right-left <= 10) insertion_sort(); else quick_sort();`

### 3. 问题：大规模数据（n>1000）下如何避免快速排序的栈溢出问题？

**解答**：使用迭代版本避免栈溢出：
- **问题原因**：递归深度log₂(n)可能超过系统栈限制
- **解决方案**：用显式栈替代递归调用
- **实现方法**：用`std::stack`存储待处理的区间，循环处理
- **空间复杂度**：O(log n)，与递归版本相同
- **时间复杂度**：O(n log n)，性能无损失

### 4. 问题：不同规模下各排序算法的性能对比如何？

**解答**：性能对比结果：

| 规模 | 插入排序 | 快速排序 | 混合排序 | 最优选择 |
|------|----------|----------|----------|----------|
| n≤10 | <g>最优</g> | 较差 | <g>最优</g> | 插入排序 |
| 10<n≤100 | 良好 | 良好 | <g>最优</g> | 混合排序 |
| n>1000 | 较差 | <g>最优</g> | 良好 | 快速排序 |

**关键点**：
- 小规模：简单算法更优
- 中等规模：混合策略最佳  
- 大规模：分治法优势明显

---

> **返回**: [《经典排序算法归类总结》](/2025/09/05/sorting-algorithms-analysis/)

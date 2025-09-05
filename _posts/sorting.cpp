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
// （移除重复的希尔排序实现，保留下方版本）

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

//插入排序
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

//快速排序
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
void quickly_sort_int_vector(std::vector<int>& v, int low, int high)
{
	if(low >= high) return;
	int mid = quickly_sort_sub(v, low, high);
	quickly_sort_int_vector(v, low, mid);
	quickly_sort_int_vector(v, mid + 1, high);
}

//归并排序
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
void merge_sort_int_vector(std::vector<int> & v, int left, int right)
{
	if(left >= right) return;
	
	int mid = left + (right - left) / 2;
	merge_sort_int_vector(v, left, mid);
	merge_sort_int_vector(v, mid + 1, right);
	merge(v, left, mid, right);
}

// 基数排序
int getDigit(int num, int digit) {
	return (num / digit) % 10;
}
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

//堆排序
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
int heap_sort_int_vector(std::vector<int> & v)
{
	if(v.size() <= 1) return 0;

	int n = v.size();
	for(int i = n / 2 - 1; i >= 0; i--)
	{
		heap_sink(v,n,i);
	}

	for(int i = n - 1; i > 0; i--)
	{
		std::swap(v[i], v[0]);
		heap_sink(v,i,0);
	}
	return 1;
}


int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr);

	int n;
	if (!(std::cin >> n)) return 0;
	std::vector<int> a(n);
	for (int i = 0; i < n; ++i) std::cin >> a[i];

	// 升序排序（当前使用希尔排序）
	// insertionSort(a.begin(), a.end());
	// insertion_sort_int_vector_with(a);
	//radix_sort_int_vector(a);
	heap_sort_int_vector(a);
	
	for (int i = 0; i < n; ++i) {
		if (i) std::cout << ' ';
		std::cout << a[i];
	}
	std::cout << '\n';
	return 0;
}



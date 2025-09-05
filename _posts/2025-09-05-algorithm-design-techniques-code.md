---
layout:       post
title:        "算法设计技术完整C++代码实现"
subtitle:     "五大算法设计方法的详细代码示例"
date:         2025-09-05 15:30:00
author:       "zxh"
header-style: text
catalog:      true
hidden:       true
tags:
  - Algorithm
  - Design Patterns
  - C++
  - Implementation
  - 算法实现
---

本页面展示了五种经典算法设计技术的完整C++实现代码。

## 完整源码

```cpp
/**
 * 算法设计技术综合实例
 * 包含：分治法、贪心法、动态规划法、回溯法、分支界限法
 * 
 * 作者：zxh
 * 日期：2025-01-20
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <climits>
#include <string>
#include <cmath>

// =================================================================================
// 1. 分治法 (Divide and Conquer)
// =================================================================================

/**
 * 分治法：归并排序
 * 时间复杂度：O(n log n)
 * 空间复杂度：O(n)
 */
class DivideConquer {
public:
    // 归并排序主函数
    static void mergeSort(std::vector<int>& arr, int left, int right) {
        if (left >= right) return;
        
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);      // 分治：排序左半部分
        mergeSort(arr, mid + 1, right); // 分治：排序右半部分
        merge(arr, left, mid, right);   // 合并：合并两个有序部分
    }
    
    // 最大子数组问题（分治法）
    static int maxSubarraySum(const std::vector<int>& arr, int left, int right) {
        if (left == right) return arr[left];
        
        int mid = left + (right - left) / 2;
        
        // 分治：递归求解左右两部分
        int leftSum = maxSubarraySum(arr, left, mid);
        int rightSum = maxSubarraySum(arr, mid + 1, right);
        
        // 合并：求跨越中点的最大子数组
        int leftMax = INT_MIN, sum = 0;
        for (int i = mid; i >= left; i--) {
            sum += arr[i];
            leftMax = std::max(leftMax, sum);
        }
        
        int rightMax = INT_MIN;
        sum = 0;
        for (int i = mid + 1; i <= right; i++) {
            sum += arr[i];
            rightMax = std::max(rightMax, sum);
        }
        
        int crossSum = leftMax + rightMax;
        return std::max({leftSum, rightSum, crossSum});
    }
    
    // 快速幂算法（分治法）
    static long long quickPower(long long base, long long exp, long long mod = 1e9 + 7) {
        if (exp == 0) return 1;
        if (exp == 1) return base % mod;
        
        long long half = quickPower(base, exp / 2, mod);
        half = (half * half) % mod;
        
        if (exp % 2 == 1) {
            half = (half * base) % mod;
        }
        
        return half;
    }

private:
    static void merge(std::vector<int>& arr, int left, int mid, int right) {
        std::vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;
        
        while (i <= mid && j <= right) {
            temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
        }
        
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        
        for (int i = 0; i < k; i++) {
            arr[left + i] = temp[i];
        }
    }
};

// =================================================================================
// 2. 贪心法 (Greedy Algorithm)
// =================================================================================

/**
 * 贪心法：活动选择、最小生成树、哈夫曼编码等
 */
class GreedyAlgorithm {
public:
    // 活动选择问题
    struct Activity {
        int start, end, id;
        Activity(int s, int e, int i) : start(s), end(e), id(i) {}
    };
    
    static std::vector<int> activitySelection(std::vector<Activity>& activities) {
        // 贪心策略：按结束时间排序
        std::sort(activities.begin(), activities.end(), 
                 [](const Activity& a, const Activity& b) {
                     return a.end < b.end;
                 });
        
        std::vector<int> selected;
        int lastEnd = -1;
        
        for (const auto& activity : activities) {
            if (activity.start >= lastEnd) {  // 贪心选择
                selected.push_back(activity.id);
                lastEnd = activity.end;
            }
        }
        
        return selected;
    }
    
    // 找零钱问题（贪心法）
    static std::vector<int> coinChange(const std::vector<int>& coins, int amount) {
        std::vector<int> result;
        
        // 贪心策略：从大到小使用硬币
        for (int i = coins.size() - 1; i >= 0 && amount > 0; i--) {
            while (amount >= coins[i]) {
                result.push_back(coins[i]);
                amount -= coins[i];
            }
        }
        
        return amount == 0 ? result : std::vector<int>{};  // 无解返回空
    }
    
    // 分糖果问题
    static int distributeCandies(std::vector<int>& ratings) {
        int n = ratings.size();
        std::vector<int> candies(n, 1);
        
        // 从左到右扫描
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i-1]) {
                candies[i] = candies[i-1] + 1;
            }
        }
        
        // 从右到左扫描
        for (int i = n-2; i >= 0; i--) {
            if (ratings[i] > ratings[i+1]) {
                candies[i] = std::max(candies[i], candies[i+1] + 1);
            }
        }
        
        int total = 0;
        for (int candy : candies) {
            total += candy;
        }
        
        return total;
    }
};

// =================================================================================
// 3. 动态规划法 (Dynamic Programming)
// =================================================================================

/**
 * 动态规划法：最优子结构 + 重叠子问题
 */
class DynamicProgramming {
public:
    // 0-1背包问题
    static int knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
        int n = weights.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {
                // 不选择第i个物品
                dp[i][w] = dp[i-1][w];
                
                // 如果能选择第i个物品，取最大值
                if (w >= weights[i-1]) {
                    dp[i][w] = std::max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
                }
            }
        }
        
        return dp[n][capacity];
    }
    
    // 最长公共子序列（LCS）
    static int longestCommonSubsequence(const std::string& text1, const std::string& text2) {
        int m = text1.length(), n = text2.length();
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i-1] == text2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    // 最长递增子序列（LIS）
    static int lengthOfLIS(const std::vector<int>& nums) {
        if (nums.empty()) return 0;
        
        int n = nums.size();
        std::vector<int> dp(n, 1);
        
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = std::max(dp[i], dp[j] + 1);
                }
            }
        }
        
        return *std::max_element(dp.begin(), dp.end());
    }
    
    // 编辑距离（Levenshtein Distance）
    static int editDistance(const std::string& word1, const std::string& word2) {
        int m = word1.length(), n = word2.length();
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
        
        // 初始化边界条件
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i-1] == word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + std::min({dp[i-1][j],     // 删除
                                            dp[i][j-1],     // 插入
                                            dp[i-1][j-1]}); // 替换
                }
            }
        }
        
        return dp[m][n];
    }
    
    // 硬币找零问题（DP版本）
    static int coinChangeDP(const std::vector<int>& coins, int amount) {
        std::vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = std::min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }
};

// =================================================================================
// 4. 回溯法 (Backtracking)
// =================================================================================

/**
 * 回溯法：穷举搜索 + 剪枝
 */
class Backtracking {
public:
    // N皇后问题
    static std::vector<std::vector<std::string>> solveNQueens(int n) {
        std::vector<std::vector<std::string>> solutions;
        std::vector<int> queens(n, -1);  // queens[i]表示第i行皇后的列位置
        
        solveNQueensHelper(0, n, queens, solutions);
        return solutions;
    }
    
    // 数独求解
    static bool solveSudoku(std::vector<std::vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValidSudoku(board, i, j, c)) {
                            board[i][j] = c;  // 做选择
                            
                            if (solveSudoku(board)) {  // 递归求解
                                return true;
                            }
                            
                            board[i][j] = '.';  // 撤销选择（回溯）
                        }
                    }
                    return false;  // 无解
                }
            }
        }
        return true;  // 所有位置都填完了
    }
    
    // 子集生成（所有可能的子集）
    static std::vector<std::vector<int>> subsets(const std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        std::vector<int> path;
        
        backtrackSubsets(nums, 0, path, result);
        return result;
    }
    
    // 组合问题：从n个数中选k个数
    static std::vector<std::vector<int>> combine(int n, int k) {
        std::vector<std::vector<int>> result;
        std::vector<int> path;
        
        backtrackCombine(1, n, k, path, result);
        return result;
    }
    
    // 全排列问题
    static std::vector<std::vector<int>> permute(std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        backtrackPermute(nums, 0, result);
        return result;
    }

private:
    static void solveNQueensHelper(int row, int n, std::vector<int>& queens, 
                                  std::vector<std::vector<std::string>>& solutions) {
        if (row == n) {
            // 找到一个解
            std::vector<std::string> board(n, std::string(n, '.'));
            for (int i = 0; i < n; i++) {
                board[i][queens[i]] = 'Q';
            }
            solutions.push_back(board);
            return;
        }
        
        for (int col = 0; col < n; col++) {
            if (isQueenSafe(queens, row, col)) {
                queens[row] = col;  // 做选择
                solveNQueensHelper(row + 1, n, queens, solutions);  // 递归
                queens[row] = -1;   // 撤销选择（回溯）
            }
        }
    }
    
    static bool isQueenSafe(const std::vector<int>& queens, int row, int col) {
        for (int i = 0; i < row; i++) {
            // 检查列冲突和对角线冲突
            if (queens[i] == col || 
                abs(queens[i] - col) == abs(i - row)) {
                return false;
            }
        }
        return true;
    }
    
    static bool isValidSudoku(const std::vector<std::vector<char>>& board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            // 检查行
            if (board[row][i] == c) return false;
            // 检查列
            if (board[i][col] == c) return false;
            // 检查3x3方格
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false;
        }
        return true;
    }
    
    static void backtrackSubsets(const std::vector<int>& nums, int start, 
                                std::vector<int>& path, std::vector<std::vector<int>>& result) {
        result.push_back(path);  // 当前路径就是一个子集
        
        for (int i = start; i < nums.size(); i++) {
            path.push_back(nums[i]);  // 做选择
            backtrackSubsets(nums, i + 1, path, result);  // 递归
            path.pop_back();  // 撤销选择（回溯）
        }
    }
    
    static void backtrackCombine(int start, int n, int k, 
                                std::vector<int>& path, std::vector<std::vector<int>>& result) {
        if (path.size() == k) {
            result.push_back(path);
            return;
        }
        
        for (int i = start; i <= n; i++) {
            path.push_back(i);  // 做选择
            backtrackCombine(i + 1, n, k, path, result);  // 递归
            path.pop_back();  // 撤销选择（回溯）
        }
    }
    
    static void backtrackPermute(std::vector<int>& nums, int start, 
                                std::vector<std::vector<int>>& result) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }
        
        for (int i = start; i < nums.size(); i++) {
            std::swap(nums[start], nums[i]);  // 做选择
            backtrackPermute(nums, start + 1, result);  // 递归
            std::swap(nums[start], nums[i]);  // 撤销选择（回溯）
        }
    }
};

// =================================================================================
// 5. 分支界限法 (Branch and Bound)
// =================================================================================

/**
 * 分支界限法：最优化问题的搜索算法
 */
class BranchAndBound {
public:
    // 0-1背包问题的分支界限解法
    struct Node {
        int level;      // 当前层数
        int profit;     // 当前利润
        int weight;     // 当前重量
        double bound;   // 上界
        
        Node(int l, int p, int w, double b) : level(l), profit(p), weight(w), bound(b) {}
        
        // 优先队列需要的比较函数（按bound降序）
        bool operator<(const Node& other) const {
            return bound < other.bound;
        }
    };
    
    static int knapsackBB(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
        int n = weights.size();
        
        // 按价值密度排序
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (double)values[a] / weights[a] > (double)values[b] / weights[b];
        });
        
        std::priority_queue<Node> pq;
        int maxProfit = 0;
        
        // 根节点
        Node root(-1, 0, 0, calculateBound(-1, 0, 0, capacity, weights, values, indices));
        pq.push(root);
        
        while (!pq.empty()) {
            Node current = pq.top();
            pq.pop();
            
            if (current.bound <= maxProfit) {
                continue;  // 剪枝
            }
            
            if (current.level == n - 1) {
                continue;
            }
            
            int nextLevel = current.level + 1;
            int itemIndex = indices[nextLevel];
            
            // 分支1：包含当前物品
            if (current.weight + weights[itemIndex] <= capacity) {
                int newProfit = current.profit + values[itemIndex];
                int newWeight = current.weight + weights[itemIndex];
                double newBound = calculateBound(nextLevel, newProfit, newWeight, capacity, weights, values, indices);
                
                maxProfit = std::max(maxProfit, newProfit);
                
                if (newBound > maxProfit) {
                    pq.push(Node(nextLevel, newProfit, newWeight, newBound));
                }
            }
            
            // 分支2：不包含当前物品
            double newBound = calculateBound(nextLevel, current.profit, current.weight, capacity, weights, values, indices);
            if (newBound > maxProfit) {
                pq.push(Node(nextLevel, current.profit, current.weight, newBound));
            }
        }
        
        return maxProfit;
    }
    
    // 旅行商问题的分支界限解法（简化版）
    static int tsp(const std::vector<std::vector<int>>& graph) {
        int n = graph.size();
        std::vector<bool> visited(n, false);
        visited[0] = true;  // 从城市0开始
        
        return tspHelper(graph, visited, 0, 1, 0, INT_MAX);
    }

private:
    static double calculateBound(int level, int profit, int weight, int capacity,
                               const std::vector<int>& weights, const std::vector<int>& values,
                               const std::vector<int>& indices) {
        if (weight >= capacity) return 0;
        
        double bound = profit;
        int totalWeight = weight;
        
        for (int i = level + 1; i < indices.size(); i++) {
            int itemIndex = indices[i];
            if (totalWeight + weights[itemIndex] <= capacity) {
                totalWeight += weights[itemIndex];
                bound += values[itemIndex];
            } else {
                // 部分装入
                bound += (double)(capacity - totalWeight) * values[itemIndex] / weights[itemIndex];
                break;
            }
        }
        
        return bound;
    }
    
    static int tspHelper(const std::vector<std::vector<int>>& graph, std::vector<bool>& visited,
                        int currentCity, int count, int cost, int minCost) {
        int n = graph.size();
        
        if (count == n) {
            // 所有城市都访问了，返回起点
            return cost + graph[currentCity][0];
        }
        
        for (int nextCity = 0; nextCity < n; nextCity++) {
            if (!visited[nextCity] && graph[currentCity][nextCity] != 0) {
                // 剪枝：如果当前路径已经超过最优解，则不继续
                if (cost + graph[currentCity][nextCity] < minCost) {
                    visited[nextCity] = true;
                    int newCost = tspHelper(graph, visited, nextCity, count + 1,
                                          cost + graph[currentCity][nextCity], minCost);
                    minCost = std::min(minCost, newCost);
                    visited[nextCity] = false;  // 回溯
                }
            }
        }
        
        return minCost;
    }
};

// =================================================================================
// 测试和演示函数
// =================================================================================

void demonstrateAlgorithms() {
    std::cout << "=== 算法设计技术演示 ===\n\n";
    
    // 1. 分治法演示
    std::cout << "1. 分治法演示：\n";
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "原数组: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";
    
    DivideConquer::mergeSort(arr, 0, arr.size() - 1);
    std::cout << "归并排序后: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";
    
    std::vector<int> maxSubArr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int maxSum = DivideConquer::maxSubarraySum(maxSubArr, 0, maxSubArr.size() - 1);
    std::cout << "最大子数组和: " << maxSum << "\n\n";
    
    // 2. 贪心法演示
    std::cout << "2. 贪心法演示：\n";
    std::vector<GreedyAlgorithm::Activity> activities = {
        {1, 4, 0}, {3, 5, 1}, {0, 6, 2}, {5, 7, 3}, {3, 9, 4}, {5, 9, 5}, {6, 10, 6}, {8, 11, 7}
    };
    auto selected = GreedyAlgorithm::activitySelection(activities);
    std::cout << "活动选择结果: ";
    for (int id : selected) std::cout << id << " ";
    std::cout << "\n\n";
    
    // 3. 动态规划演示
    std::cout << "3. 动态规划演示：\n";
    std::vector<int> weights = {1, 3, 4, 5};
    std::vector<int> values = {1, 4, 5, 7};
    int capacity = 7;
    int maxValue = DynamicProgramming::knapsack01(weights, values, capacity);
    std::cout << "0-1背包最大价值: " << maxValue << "\n";
    
    std::string text1 = "abcde", text2 = "ace";
    int lcsLength = DynamicProgramming::longestCommonSubsequence(text1, text2);
    std::cout << "最长公共子序列长度: " << lcsLength << "\n\n";
    
    // 4. 回溯法演示
    std::cout << "4. 回溯法演示：\n";
    int n = 4;
    auto solutions = Backtracking::solveNQueens(n);
    std::cout << n << "皇后问题解的数量: " << solutions.size() << "\n";
    
    std::vector<int> nums = {1, 2, 3};
    auto subsets = Backtracking::subsets(nums);
    std::cout << "子集数量: " << subsets.size() << "\n\n";
    
    // 5. 分支界限法演示
    std::cout << "5. 分支界限法演示：\n";
    int bbResult = BranchAndBound::knapsackBB(weights, values, capacity);
    std::cout << "分支界限法背包最大价值: " << bbResult << "\n";
    
    std::cout << "\n=== 演示完成 ===\n";
}

// 主函数
int main() {
    demonstrateAlgorithms();
    return 0;
}
```

## 编译和运行

```bash
# 编译
g++ -std=c++11 -O2 algorithm-design-techniques.cpp -o algorithm_demo

# 运行
./algorithm_demo
```

## 各技术详细说明

### 1. 分治法实现要点
- **归并排序**：经典的分治算法，分解→递归→合并
- **最大子数组**：考虑跨越中点的情况
- **快速幂**：利用指数的二进制表示

### 2. 贪心法实现要点
- **活动选择**：按结束时间排序，贪心选择
- **硬币找零**：从大面额开始使用
- **分糖果**：双向扫描保证局部最优

### 3. 动态规划实现要点
- **状态定义**：dp[i][w]表示前i个物品在容量w下的最大价值
- **状态转移**：选择或不选择当前物品
- **边界条件**：空串或空数组的情况

### 4. 回溯法实现要点
- **做选择**：在当前状态下做出决定
- **递归探索**：继续搜索下一层
- **撤销选择**：回溯时恢复状态

### 5. 分支界限法实现要点
- **界限函数**：估算最优可能值
- **优先队列**：按界限值排序
- **剪枝策略**：超过已知最优解时剪枝

---

> **返回**: [《算法设计技术可视化讲解》](/2025/09/05/algorithm-design-visualization/)

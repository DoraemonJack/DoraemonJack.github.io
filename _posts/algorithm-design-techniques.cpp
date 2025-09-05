/**
 * �㷨��Ƽ����ۺ�ʵ��
 * ���������η���̰�ķ�����̬�滮�������ݷ�����֧���޷�
 * 
 * ���ߣ�zxh
 * ���ڣ�2025-01-20
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <climits>
#include <string>
#include <cmath>

// =================================================================================
// 1. ���η� (Divide and Conquer)
// =================================================================================

/**
 * ���η����鲢����
 * ʱ�临�Ӷȣ�O(n log n)
 * �ռ临�Ӷȣ�O(n)
 */
class DivideConquer {
public:
    // �鲢����������
    static void mergeSort(std::vector<int>& arr, int left, int right) {
        if (left >= right) return;
        
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);      // ���Σ�������벿��
        mergeSort(arr, mid + 1, right); // ���Σ������Ұ벿��
        merge(arr, left, mid, right);   // �ϲ����ϲ��������򲿷�
    }
    
    // ������������⣨���η���
    static int maxSubarraySum(const std::vector<int>& arr, int left, int right) {
        if (left == right) return arr[left];
        
        int mid = left + (right - left) / 2;
        
        // ���Σ��ݹ��������������
        int leftSum = maxSubarraySum(arr, left, mid);
        int rightSum = maxSubarraySum(arr, mid + 1, right);
        
        // �ϲ������Խ�е�����������
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
    
    // �������㷨�����η���
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
// 2. ̰�ķ� (Greedy Algorithm)
// =================================================================================

/**
 * ̰�ķ����ѡ����С�������������������
 */
class GreedyAlgorithm {
public:
    // �ѡ������
    struct Activity {
        int start, end, id;
        Activity(int s, int e, int i) : start(s), end(e), id(i) {}
    };
    
    static std::vector<int> activitySelection(std::vector<Activity>& activities) {
        // ̰�Ĳ��ԣ�������ʱ������
        std::sort(activities.begin(), activities.end(), 
                 [](const Activity& a, const Activity& b) {
                     return a.end < b.end;
                 });
        
        std::vector<int> selected;
        int lastEnd = -1;
        
        for (const auto& activity : activities) {
            if (activity.start >= lastEnd) {  // ̰��ѡ��
                selected.push_back(activity.id);
                lastEnd = activity.end;
            }
        }
        
        return selected;
    }
    
    // ����Ǯ���⣨̰�ķ���
    static std::vector<int> coinChange(const std::vector<int>& coins, int amount) {
        std::vector<int> result;
        
        // ̰�Ĳ��ԣ��Ӵ�Сʹ��Ӳ��
        for (int i = coins.size() - 1; i >= 0 && amount > 0; i--) {
            while (amount >= coins[i]) {
                result.push_back(coins[i]);
                amount -= coins[i];
            }
        }
        
        return amount == 0 ? result : std::vector<int>{};  // �޽ⷵ�ؿ�
    }
    
    // ���ǹ�����
    static int distributeCandies(std::vector<int>& ratings) {
        int n = ratings.size();
        std::vector<int> candies(n, 1);
        
        // ������ɨ��
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i-1]) {
                candies[i] = candies[i-1] + 1;
            }
        }
        
        // ���ҵ���ɨ��
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
// 3. ��̬�滮�� (Dynamic Programming)
// =================================================================================

/**
 * ��̬�滮���������ӽṹ + �ص�������
 */
class DynamicProgramming {
public:
    // 0-1��������
    static int knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
        int n = weights.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {
                // ��ѡ���i����Ʒ
                dp[i][w] = dp[i-1][w];
                
                // �����ѡ���i����Ʒ��ȡ���ֵ
                if (w >= weights[i-1]) {
                    dp[i][w] = std::max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1]);
                }
            }
        }
        
        return dp[n][capacity];
    }
    
    // ����������У�LCS��
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
    
    // ����������У�LIS��
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
    
    // �༭���루Levenshtein Distance��
    static int editDistance(const std::string& word1, const std::string& word2) {
        int m = word1.length(), n = word2.length();
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
        
        // ��ʼ���߽�����
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i-1] == word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + std::min({dp[i-1][j],     // ɾ��
                                            dp[i][j-1],     // ����
                                            dp[i-1][j-1]}); // �滻
                }
            }
        }
        
        return dp[m][n];
    }
    
    // Ӳ���������⣨DP�汾��
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
// 4. ���ݷ� (Backtracking)
// =================================================================================

/**
 * ���ݷ���������� + ��֦
 */
class Backtracking {
public:
    // N�ʺ�����
    static std::vector<std::vector<std::string>> solveNQueens(int n) {
        std::vector<std::vector<std::string>> solutions;
        std::vector<int> queens(n, -1);  // queens[i]��ʾ��i�лʺ����λ��
        
        solveNQueensHelper(0, n, queens, solutions);
        return solutions;
    }
    
    // �������
    static bool solveSudoku(std::vector<std::vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValidSudoku(board, i, j, c)) {
                            board[i][j] = c;  // ��ѡ��
                            
                            if (solveSudoku(board)) {  // �ݹ����
                                return true;
                            }
                            
                            board[i][j] = '.';  // ����ѡ�񣨻��ݣ�
                        }
                    }
                    return false;  // �޽�
                }
            }
        }
        return true;  // ����λ�ö�������
    }
    
    // �Ӽ����ɣ����п��ܵ��Ӽ���
    static std::vector<std::vector<int>> subsets(const std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        std::vector<int> path;
        
        backtrackSubsets(nums, 0, path, result);
        return result;
    }
    
    // ������⣺��n������ѡk����
    static std::vector<std::vector<int>> combine(int n, int k) {
        std::vector<std::vector<int>> result;
        std::vector<int> path;
        
        backtrackCombine(1, n, k, path, result);
        return result;
    }
    
    // ȫ��������
    static std::vector<std::vector<int>> permute(std::vector<int>& nums) {
        std::vector<std::vector<int>> result;
        backtrackPermute(nums, 0, result);
        return result;
    }

private:
    static void solveNQueensHelper(int row, int n, std::vector<int>& queens, 
                                  std::vector<std::vector<std::string>>& solutions) {
        if (row == n) {
            // �ҵ�һ����
            std::vector<std::string> board(n, std::string(n, '.'));
            for (int i = 0; i < n; i++) {
                board[i][queens[i]] = 'Q';
            }
            solutions.push_back(board);
            return;
        }
        
        for (int col = 0; col < n; col++) {
            if (isQueenSafe(queens, row, col)) {
                queens[row] = col;  // ��ѡ��
                solveNQueensHelper(row + 1, n, queens, solutions);  // �ݹ�
                queens[row] = -1;   // ����ѡ�񣨻��ݣ�
            }
        }
    }
    
    static bool isQueenSafe(const std::vector<int>& queens, int row, int col) {
        for (int i = 0; i < row; i++) {
            // ����г�ͻ�ͶԽ��߳�ͻ
            if (queens[i] == col || 
                abs(queens[i] - col) == abs(i - row)) {
                return false;
            }
        }
        return true;
    }
    
    static bool isValidSudoku(const std::vector<std::vector<char>>& board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            // �����
            if (board[row][i] == c) return false;
            // �����
            if (board[i][col] == c) return false;
            // ���3x3����
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false;
        }
        return true;
    }
    
    static void backtrackSubsets(const std::vector<int>& nums, int start, 
                                std::vector<int>& path, std::vector<std::vector<int>>& result) {
        result.push_back(path);  // ��ǰ·������һ���Ӽ�
        
        for (int i = start; i < nums.size(); i++) {
            path.push_back(nums[i]);  // ��ѡ��
            backtrackSubsets(nums, i + 1, path, result);  // �ݹ�
            path.pop_back();  // ����ѡ�񣨻��ݣ�
        }
    }
    
    static void backtrackCombine(int start, int n, int k, 
                                std::vector<int>& path, std::vector<std::vector<int>>& result) {
        if (path.size() == k) {
            result.push_back(path);
            return;
        }
        
        for (int i = start; i <= n; i++) {
            path.push_back(i);  // ��ѡ��
            backtrackCombine(i + 1, n, k, path, result);  // �ݹ�
            path.pop_back();  // ����ѡ�񣨻��ݣ�
        }
    }
    
    static void backtrackPermute(std::vector<int>& nums, int start, 
                                std::vector<std::vector<int>>& result) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }
        
        for (int i = start; i < nums.size(); i++) {
            std::swap(nums[start], nums[i]);  // ��ѡ��
            backtrackPermute(nums, start + 1, result);  // �ݹ�
            std::swap(nums[start], nums[i]);  // ����ѡ�񣨻��ݣ�
        }
    }
};

// =================================================================================
// 5. ��֧���޷� (Branch and Bound)
// =================================================================================

/**
 * ��֧���޷������Ż�����������㷨
 */
class BranchAndBound {
public:
    // 0-1��������ķ�֧���޽ⷨ
    struct Node {
        int level;      // ��ǰ����
        int profit;     // ��ǰ����
        int weight;     // ��ǰ����
        double bound;   // �Ͻ�
        
        Node(int l, int p, int w, double b) : level(l), profit(p), weight(w), bound(b) {}
        
        // ���ȶ�����Ҫ�ıȽϺ�������bound����
        bool operator<(const Node& other) const {
            return bound < other.bound;
        }
    };
    
    static int knapsackBB(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
        int n = weights.size();
        
        // ����ֵ�ܶ�����
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (double)values[a] / weights[a] > (double)values[b] / weights[b];
        });
        
        std::priority_queue<Node> pq;
        int maxProfit = 0;
        
        // ���ڵ�
        Node root(-1, 0, 0, calculateBound(-1, 0, 0, capacity, weights, values, indices));
        pq.push(root);
        
        while (!pq.empty()) {
            Node current = pq.top();
            pq.pop();
            
            if (current.bound <= maxProfit) {
                continue;  // ��֦
            }
            
            if (current.level == n - 1) {
                continue;
            }
            
            int nextLevel = current.level + 1;
            int itemIndex = indices[nextLevel];
            
            // ��֧1��������ǰ��Ʒ
            if (current.weight + weights[itemIndex] <= capacity) {
                int newProfit = current.profit + values[itemIndex];
                int newWeight = current.weight + weights[itemIndex];
                double newBound = calculateBound(nextLevel, newProfit, newWeight, capacity, weights, values, indices);
                
                maxProfit = std::max(maxProfit, newProfit);
                
                if (newBound > maxProfit) {
                    pq.push(Node(nextLevel, newProfit, newWeight, newBound));
                }
            }
            
            // ��֧2����������ǰ��Ʒ
            double newBound = calculateBound(nextLevel, current.profit, current.weight, capacity, weights, values, indices);
            if (newBound > maxProfit) {
                pq.push(Node(nextLevel, current.profit, current.weight, newBound));
            }
        }
        
        return maxProfit;
    }
    
    // ����������ķ�֧���޽ⷨ���򻯰棩
    static int tsp(const std::vector<std::vector<int>>& graph) {
        int n = graph.size();
        std::vector<bool> visited(n, false);
        visited[0] = true;  // �ӳ���0��ʼ
        
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
                // ����װ��
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
            // ���г��ж������ˣ��������
            return cost + graph[currentCity][0];
        }
        
        for (int nextCity = 0; nextCity < n; nextCity++) {
            if (!visited[nextCity] && graph[currentCity][nextCity] != 0) {
                // ��֦�������ǰ·���Ѿ��������Ž⣬�򲻼���
                if (cost + graph[currentCity][nextCity] < minCost) {
                    visited[nextCity] = true;
                    int newCost = tspHelper(graph, visited, nextCity, count + 1,
                                          cost + graph[currentCity][nextCity], minCost);
                    minCost = std::min(minCost, newCost);
                    visited[nextCity] = false;  // ����
                }
            }
        }
        
        return minCost;
    }
};

// =================================================================================
// ���Ժ���ʾ����
// =================================================================================

void demonstrateAlgorithms() {
    std::cout << "=== �㷨��Ƽ�����ʾ ===\n\n";
    
    // 1. ���η���ʾ
    std::cout << "1. ���η���ʾ��\n";
    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "ԭ����: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";
    
    DivideConquer::mergeSort(arr, 0, arr.size() - 1);
    std::cout << "�鲢�����: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";
    
    std::vector<int> maxSubArr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int maxSum = DivideConquer::maxSubarraySum(maxSubArr, 0, maxSubArr.size() - 1);
    std::cout << "����������: " << maxSum << "\n\n";
    
    // 2. ̰�ķ���ʾ
    std::cout << "2. ̰�ķ���ʾ��\n";
    std::vector<GreedyAlgorithm::Activity> activities = {
        {1, 4, 0}, {3, 5, 1}, {0, 6, 2}, {5, 7, 3}, {3, 9, 4}, {5, 9, 5}, {6, 10, 6}, {8, 11, 7}
    };
    auto selected = GreedyAlgorithm::activitySelection(activities);
    std::cout << "�ѡ����: ";
    for (int id : selected) std::cout << id << " ";
    std::cout << "\n\n";
    
    // 3. ��̬�滮��ʾ
    std::cout << "3. ��̬�滮��ʾ��\n";
    std::vector<int> weights = {1, 3, 4, 5};
    std::vector<int> values = {1, 4, 5, 7};
    int capacity = 7;
    int maxValue = DynamicProgramming::knapsack01(weights, values, capacity);
    std::cout << "0-1��������ֵ: " << maxValue << "\n";
    
    std::string text1 = "abcde", text2 = "ace";
    int lcsLength = DynamicProgramming::longestCommonSubsequence(text1, text2);
    std::cout << "����������г���: " << lcsLength << "\n\n";
    
    // 4. ���ݷ���ʾ
    std::cout << "4. ���ݷ���ʾ��\n";
    int n = 4;
    auto solutions = Backtracking::solveNQueens(n);
    std::cout << n << "�ʺ�����������: " << solutions.size() << "\n";
    
    std::vector<int> nums = {1, 2, 3};
    auto subsets = Backtracking::subsets(nums);
    std::cout << "�Ӽ�����: " << subsets.size() << "\n\n";
    
    // 5. ��֧���޷���ʾ
    std::cout << "5. ��֧���޷���ʾ��\n";
    int bbResult = BranchAndBound::knapsackBB(weights, values, capacity);
    std::cout << "��֧���޷���������ֵ: " << bbResult << "\n";
    
    std::cout << "\n=== ��ʾ��� ===\n";
}

// ������
int main() {
    demonstrateAlgorithms();
    return 0;
}

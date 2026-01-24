---
layout: post
title: "数据结构——树结构"
subtitle: "深入理解树结构的基本概念、遍历算法和实际应用"
date: 2025-09-06 15:00:00
author: "DoraemonJack"
header-img: "img/post-bg-algorithm.jpg"
catalog: true
mathjax:      true
tags:
    - 数据结构
    - Data Structure
    - 算法
    - Algorithm
    - 二叉树
    - 二叉排序树
    - 遍历算法
---

# 树结构：二叉树、树、二叉排序树等数据结构表示、遍历与实现

树结构是计算机科学中最重要的数据结构之一，广泛应用于文件系统、数据库索引、编译器语法分析等领域。本文将深入探讨树的基本概念、二叉树的表示与遍历、二叉排序树的实现，以及相关的算法应用。

## 1. 树的基本概念和术语

### 1.1 树和森林的定义

**树（Tree）**是由n个节点组成的有限集合，其中：

- 有且仅有一个根节点
- 其余节点可分为若干个互不相交的子树
- 每个子树本身也是一棵树

**森林（Trees）**是m棵互不相交的树的集合。

### 1.2 重要术语

- **节点（Node）**：树中的基本单位
- **根节点（Root）**：树的顶端节点，没有父节点
- **叶子节点（Leaf）**：没有子节点的节点
- **父节点（Parent）**：有子节点的节点
- **子节点（Child）**：父节点的直接后继
- **兄弟节点（Sibling）**：具有相同父节点的节点
- **度（Degree）**：节点拥有的子树个数,有几个孩子（分支），叶子结点的度=0
- **深度/高度（Depth/Height）**：从根到该节点的路径长度
- **层（Level）**：根节点为第0层，依次递增

### 1.3 <span style="color: #e74c3c;">树的性质</span>

1. **树中节点数 = 所有节点度数之和(总度数) + 1**

2. 度为m的树中第i层最多有 $m^i$个节点，(i>=0)。如果说从第一层(i>=1)算，那就是 $m^{i-1}$

2. m叉树的第i层最多有 $m^i$个节点。

4. **高度为h的m叉树最多有 $\frac{m^{h}-1}{m-1}$个节点。**

   **推导过程：**

   高度为h的m叉树总节点数 = 第0层 + 第1层 + ... + 第h层

   $$= 1 + m + m^2 + ... + m^h$$

   这是一个等比数列，首项 $a = 1$，公比 $q = m$，项数 $n = h+1$

   根据等比数列求和公式：

   $$S_n = \frac{a(1-q^n)}{1-q} = \frac{1 \cdot (1-m^{h})}{1-m} = \frac{m^{h}-1}{m-1}$$

5. **高度为h的m叉树至少有h个结点。高度为h,度为m的树至少有h+m-1个结点。**

6. **具有n个结点的m叉树的最小高度为 $\lceil \log_m(n(m-1)+1) \rceil$**

   **推导过程：**
   
   **前提条件：** 高度最小的情况——所有结点都有m个孩子（完全m叉树）
   
   **核心不等式：**
   
   对于高度为h的m叉树，节点数n满足：
   
   $$\frac{m^{h-1}-1}{m-1} < n \leq \frac{m^h-1}{m-1}$$
   
   其中：
   - $\frac{m^{h-1}-1}{m-1}$：前h-1层最多有几个结点
   - $\frac{m^h-1}{m-1}$：前h层最多有几个结点
   
   **推导步骤：**
   
   1. 对不等式两边同时乘以$(m-1)$并加1：
   
   $$m^{h-1} < n(m-1) + 1 \leq m^h$$
   
   2. 对不等式两边取以m为底的对数：
   
   $$h-1 < \log_m(n(m-1) + 1) \leq h$$
   
   3. 由于高度h必须是整数，因此：
   
   $$h_{min} = \lceil \log_m(n(m-1) + 1) \rceil$$

|          度为m的树          |             m叉树             |
| :-------------------------: | :---------------------------: |
| 不能为空树，至少有m+1个结点 |          可以为空树           |
|   至少有一个结点有m个孩子   | 没有要求存在一个结点有m个孩子 |
|            度<=3            |             度<3              |

## 2. 二叉树的结构和表示

### 2.1 二叉树的定义

**二叉树（Binary Tree）**是每个节点最多有两个子节点的树结构，分别称为左子树和右子树。

### 2.2 <span style="color: #e74c3c;">特殊二叉树类型</span>

#### 满二叉树
每一层都有最大数量的节点，一棵高度为h, 且含有$2^{h} - 1$个结点的二叉树。

#### 完全二叉树
除了最后一层，其他层都是满的，最后一层从左到右连续。编号对应满二叉树，**最多只有一个度为1的结点**。

$i <= \lfloor n/2 \rfloor$$   为分支结点     $  $i > \lfloor n/2 \rfloor$ 为叶子结点

#### 平衡二叉树
任意节点的左右子树高度差不超过1。

#### 二叉排序树

一棵二叉树或者空二叉树，左子树的所有结点关键字小于根节点，同理右子树。

### 2.3 <span style="color: #e74c3c;">二叉树的性质</span>

------

1. 设非空二叉树中度为0，1，2的结点个数为$n_0,n_1,n_2$，则<mark>$n_0 = n_2 + 1$</mark>（即叶子结点比分支结点多一个），假设树中的结点总数为$$n$$,则：

   <span style="color: #e74c3c;">$n = n_0 + n_1 + n_2$</span>

   <span style="color: #e74c3c;">$n = n_1 +2n_2 +1$ ,树的结点数 = 总度数+1</span>

2. 具有n个（n > 0）结点的<mark>完全二叉树</mark>的高度n为<mark>$\lceil log_2(n + 1) \rceil$</mark>或<mark>$\lfloor log_2(n) \rfloor + 1$</mark>。

3. 高为<mark>h-1</mark>的满二叉树共有 $2^{h-1} - 1$ 个结点

   高为h的完全二叉树<mark>至少</mark> $2^{h-1}$ (相当于$2^{h-1} - 1 + 1$)

   高为h的完全二叉树<mark>至多</mark> $2^{h} - 1$

4. 完全二叉树，最多只有一个度为1的结点：

   $$n_1 = 0 或 1$$

   $$n_0 = n_2 + 1（n_0 + n_2一定是奇数）$$

   $$=> 若完全二叉树有2k个（偶数）结点，则必有n_1 = 1, n_0 = k , n_2 = k - 1$$

   $$=> 若完全二叉树有2k - 1个（奇数）结点，则必有n_1 = 0, n_0 = k , n_2 = k - 1$$

   

### 2.4 二叉树的存储表示

#### 链式存储（推荐）
```cpp
typedef struct TreeNode {
    int val;
    TreeNode* left  = nullptr;  //左孩子指针
    TreeNode* right = nullptr;  //右孩子指针
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
}TreeNode, *BiTree;

BiTree root = nullptr;
#插入新结点
TreeNode* p = (TreeNode*)malloc(sizeof(TreeNode));
p->val = 2;
root->left = p;
```

#### <mark>数组存储（完全二叉树）</mark>
对于索引为i的节点：
- **i的左孩子**：$2i$
- **i的右孩子**：$2i+1$  
- **i的父节点**：$\lfloor i/2 \rfloor$
- **i所在的层次**：$\lceil \log_2(n + 1) \rceil$ 或 $\lfloor \log_2 n \rfloor + 1$

**若完全二叉树中共有n个结点，则：**

- **判断i是否有左孩子？**：$2i \leq n$？
- **判断i是否有右孩子？**：$2i+1 \leq n$？
- **判断i是否是叶子/分支结点？**：$i > \lfloor n/2 \rfloor$？

  > 当 $i > \lfloor n/2 \rfloor$ 时，节点i为叶子结点

```cpp
#define MaxSize 100
struct TreeNode{
	int value = 0;
  bool isEmpty = true; //结点是否为空
}
```

## 3. 树的遍历算法

遍历是树结构最重要的操作之一，分为深度优先遍历（DFS）和广度优先遍历（BFS）。

### 3.1 深度优先遍历（DFS）

#### 前序遍历（Preorder）
访问顺序：根 → 左 → 右

**递归实现：**
```cpp
void preorder(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";  // 访问根节点
    preorder(root->left);      // 遍历左子树
    preorder(root->right);     // 遍历右子树
}
```

**迭代实现：**
```cpp
vector<int> preorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> stk;
    stk.push(root);
    
    while (!stk.empty()) {
        TreeNode* node = stk.top();
        stk.pop();
        result.push_back(node->val);
        
        if (node->right) stk.push(node->right);
        if (node->left) stk.push(node->left);
    }
    
    return result;
}
```

**迭代原理：**
- 使用栈模拟递归调用栈
- 先访问根节点，再处理左右子树
- 由于栈是后进先出，先压入右子树，再压入左子树
- 这样出栈时就是左子树先出栈，符合"根→左→右"的访问顺序

#### 中序遍历（Inorder）
访问顺序：左 → 根 → 右

**递归实现：**
```cpp
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);       // 遍历左子树
    cout << root->val << " ";  // 访问根节点
    inorder(root->right);      // 遍历右子树
}
```

**迭代实现：**
```cpp
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> stk;
    TreeNode* curr = root;
    
    while (curr || !stk.empty()) {
        while (curr) {
            stk.push(curr);
            curr = curr->left;
        }
        curr = stk.top();
        stk.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }
    
    return result;
}
```

**迭代原理：**

- 使用栈和指针模拟递归过程
- 先一直向左走到<mark>最左端</mark>，<mark>沿途将节点压入栈</mark>
- 到达最左端后，弹出栈顶节点并访问
- 然后<mark>转向右子树</mark>，重复上述过程
- 这样保证了"左→根→右"的访问顺序

#### 后序遍历（Postorder）
访问顺序：左 → 右 → 根

**递归实现：**
```cpp
void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);     // 遍历左子树
    postorder(root->right);    // 遍历右子树
    cout << root->val << " ";  // 访问根节点
}
```

**迭代实现：**
```cpp
vector<int> postorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> stk;
    TreeNode* lastVisited = nullptr;
    TreeNode* curr = root;
    
    while (curr || !stk.empty()) {
        if (curr) {
            stk.push(curr);
            curr = curr->left;
        } else {
            TreeNode* peekNode = stk.top();
            if (peekNode->right && lastVisited != peekNode->right) {
                curr = peekNode->right;
            } else {
                result.push_back(peekNode->val);
                lastVisited = stk.top();
                stk.pop();
            }
        }
    }
    
    return result;
}
```

**迭代原理：**
- 后序遍历最复杂，需要记录上次访问的节点
- 先向左走到最左端，沿途压入栈
- 当左子树遍历完后，检查右子树
- 如果右子树存在且未被访问，则转向右子树
- 如果右子树不存在或已访问，则访问当前节点
- 使用`lastVisited`标记避免重复访问，确保"左→右→根"的顺序

### 3.2 广度优先遍历（BFS）

#### 层序遍历（Level Order）
```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(level);
    }
    
    return result;
}
```

## 4. 二叉排序树（BST）的实现

### 4.1 二叉排序树的定义

**二叉排序树（Binary Search Tree）**满足以下性质：
- 左子树所有节点的值 < 根节点的值
- 右子树所有节点的值 > 根节点的值
- 左右子树也分别是二叉排序树

### 4.2 基本操作

#### 插入操作
```cpp
TreeNode* insert(TreeNode* root, int val) {
    if (!root) {
        return new TreeNode(val);
    }
    
    if (val < root->val) {
        root->left = insert(root->left, val);
    } else if (val > root->val) {
        root->right = insert(root->right, val);
    }
    
    return root;
}
```

**插入原理：**
- 从根节点开始，比较要插入的值与当前节点值
- 如果值小于当前节点，递归插入到左子树
- 如果值大于当前节点，递归插入到右子树
- 如果值等于当前节点，不插入（BST不允许重复值）
- 当到达空节点时，创建新节点并返回
- 时间复杂度：O(log n)，最坏情况O(n)

#### 查找操作
```cpp
TreeNode* search(TreeNode* root, int val) {
    if (!root || root->val == val) {
        return root;
    }
    
    if (val < root->val) {
        return search(root->left, val);
    } else {
        return search(root->right, val);
    }
}
```

**查找原理：**
- 利用BST的有序性质进行二分查找
- 从根节点开始，比较目标值与当前节点值
- 如果目标值小于当前节点，在左子树中继续查找
- 如果目标值大于当前节点，在右子树中继续查找
- 如果找到匹配值或到达空节点，返回结果
- 每次比较都能排除一半的子树，效率很高
- 时间复杂度：O(log n)，最坏情况O(n)

#### 删除操作
```cpp
TreeNode* deleteNode(TreeNode* root, int val) {
    if (!root) return root;
    
    if (val < root->val) {
        root->left = deleteNode(root->left, val);
    } else if (val > root->val) {
        root->right = deleteNode(root->right, val);
    } else {
        // 找到要删除的节点
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        
        // 找到右子树的最小值节点
        TreeNode* minNode = findMin(root->right);
        root->val = minNode->val;
        root->right = deleteNode(root->right, minNode->val);
    }
    
    return root;
}

TreeNode* findMin(TreeNode* root) {
    while (root->left) {
        root = root->left;
    }
    return root;
}
```

**删除原理：**
- 删除操作最复杂，需要分三种情况处理：
  1. **叶子节点**：直接删除
  2. **只有一个子节点**：用子节点替换被删除节点
  3. **有两个子节点**：用右子树的最小值替换被删除节点
- 对于情况3，右子树的最小值一定在右子树的最左端
- 用最小值替换后，再递归删除原来的最小值节点
- 这样保证了BST的性质不变
- 时间复杂度：O(log n)，最坏情况O(n)

### 4.3 验证二叉搜索树

```cpp
bool isValidBST(TreeNode* root) {
    return isValidBST(root, nullptr, nullptr);
}

bool isValidBST(TreeNode* root, TreeNode* minNode, TreeNode* maxNode) {
    if (!root) return true;
    
    if ((minNode && root->val <= minNode->val) || 
        (maxNode && root->val >= maxNode->val)) {
        return false;
    }
    
    return isValidBST(root->left, minNode, root) && 
           isValidBST(root->right, root, maxNode);
}
```

**验证原理：**
- 使用上下界约束来验证BST性质
- 每个节点都有明确的值范围：`(minNode.val, maxNode.val)`
- 左子树：上界变为当前节点值，下界保持不变
- 右子树：下界变为当前节点值，上界保持不变
- 递归检查每个节点是否在合法范围内
- 如果任何节点违反约束，立即返回false
- 时间复杂度：O(n)，需要访问每个节点一次

## 5. 完整实现示例

> **📝 完整代码实现**：本文提供了完整的C++实现代码，包括二叉树和二叉排序树的所有基本操作。代码包含详细的注释和测试用例，可以直接编译运行。
<div style="text-align: center; margin: 30px 0;">
  <a href="/2025/09/06/tree_structures_complete_code/" class="btn btn-primary btn-lg" style="display: inline-block; padding: 15px 30px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); transition: all 0.3s ease; border: none;">
    <i class="fa fa-code" style="margin-right: 10px;"></i>查看完整C++源码
  </a>
</div>
<style>
.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 15px 20px rgba(0,0,0,0.2) !important;
}
</style>

## 6. 相关练习题

### 基础练习题

1. **[LeetCode 144: 二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)**
2. **[LeetCode 94: 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)**  
3. **[LeetCode 145: 二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)**
4. **[LeetCode 102: 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)**
5. **[LeetCode 104: 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)**
6. **[LeetCode 111: 二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)**
7. **[LeetCode 222: 完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)**
8. **[LeetCode 110: 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)**

### 二叉排序树相关

9. **[LeetCode 98: 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)**
10. **[LeetCode 701: 二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)**
11. **[LeetCode 450: 删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/)**
12. **[LeetCode 700: 二叉搜索树中的搜索](https://leetcode.cn/problems/search-in-a-binary-search-tree/)**

### 进阶练习题

13. **[LeetCode 105: 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)**
14. **[LeetCode 106: 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)**
15. **[LeetCode 108: 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)**
16. **[LeetCode 112: 路径总和](https://leetcode.cn/problems/path-sum/)**
17. **[LeetCode 113: 路径总和 II](https://leetcode.cn/problems/path-sum-ii/)**
18. **[LeetCode 124: 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)**
19. **[LeetCode 297: 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)**

## 7. 时间复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| 遍历操作 | O(n) | O(h) |
| BST查找 | O(log n) | O(h) |
| BST插入 | O(log n) | O(h) |
| BST删除 | O(log n) | O(h) |

其中：
- n：树中节点总数
- h：树的高度
- 最坏情况下（退化为链表），时间复杂度为O(n)

## 8. 总结

树结构是计算机科学中的基础数据结构，掌握树的基本概念、遍历算法和二叉排序树的实现对于算法学习至关重要。通过本文的学习，您应该能够：

1. 理解树的基本概念和术语
2. 掌握二叉树的存储表示方法
3. 熟练实现四种遍历算法（递归和迭代）
4. 实现二叉排序树的基本操作
5. 分析算法的时间复杂度
6. 解决相关的算法问题

树结构在实际应用中非常广泛，从文件系统到数据库索引，从编译器到人工智能，都离不开树的概念。深入理解树结构将为您的算法学习打下坚实的基础。

---

*本文涵盖了树结构的核心知识点，包括基本概念、遍历算法、二叉排序树实现等。建议结合实际编程练习来加深理解。*

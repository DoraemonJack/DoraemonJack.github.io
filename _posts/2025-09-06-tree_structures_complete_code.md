---
layout:       post
title:        "树结构完整C++代码实现"
subtitle:     "二叉树、二叉排序树的完整C++实现代码"
date:         2025-09-06 10:00:00
author:       "DoraemonJack"
header-style: text
catalog:      false
header-img: "img/post-bg-algorithm.jpg"
mathjax:      false
hidden:       true
tags:
    - Algorithm
    - Data Structure
    - C++
    - Tree
    - Binary Tree
    - BST
    - 算法实现
    - 二叉树
    - 二叉排序树
---

# 树结构完整代码实现

本文提供树结构的完整C++实现代码，包括二叉树和二叉排序树的所有基本操作。

## 完整代码实现

```cpp
#include <iostream>
#include <queue>
#include <stack>
#include <vector>
using namespace std;

// 二叉树节点定义
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 二叉树类
class BinaryTree {
private:
    TreeNode* root;
    
public:
    BinaryTree() : root(nullptr) {}
    
    // 插入节点（按层序插入）
    void insert(int val) {
        if (!root) {
            root = new TreeNode(val);
            return;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (!node->left) {
                node->left = new TreeNode(val);
                return;
            } else if (!node->right) {
                node->right = new TreeNode(val);
                return;
            } else {
                q.push(node->left);
                q.push(node->right);
            }
        }
    }
    
    // 前序遍历（递归）
    void preorder(TreeNode* node) {
        if (!node) return;
        cout << node->val << " ";
        preorder(node->left);
        preorder(node->right);
    }
    
    // 前序遍历（迭代）
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
    
    // 中序遍历（递归）
    void inorder(TreeNode* node) {
        if (!node) return;
        inorder(node->left);
        cout << node->val << " ";
        inorder(node->right);
    }
    
    // 中序遍历（迭代）
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
    
    // 后序遍历（递归）
    void postorder(TreeNode* node) {
        if (!node) return;
        postorder(node->left);
        postorder(node->right);
        cout << node->val << " ";
    }
    
    // 后序遍历（迭代）
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
    
    // 层序遍历
    void levelOrder() {
        if (!root) return;
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            cout << node->val << " ";
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    
    // 计算树的高度
    int height(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(height(root->left), height(root->right));
    }
    
    // 计算节点总数
    int countNodes(TreeNode* root) {
        if (!root) return 0;
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
    
    // 验证是否为平衡二叉树
    bool isBalanced(TreeNode* root) {
        return checkHeight(root) != -1;
    }
    
    int checkHeight(TreeNode* root) {
        if (!root) return 0;
        
        int leftHeight = checkHeight(root->left);
        if (leftHeight == -1) return -1;
        
        int rightHeight = checkHeight(root->right);
        if (rightHeight == -1) return -1;
        
        if (abs(leftHeight - rightHeight) > 1) return -1;
        
        return 1 + max(leftHeight, rightHeight);
    }
    
    TreeNode* getRoot() { return root; }
    
    ~BinaryTree() {
        destroyTree(root);
    }
    
private:
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
};

// 二叉排序树类
class BinarySearchTree {
private:
    TreeNode* root;
    
public:
    BinarySearchTree() : root(nullptr) {}
    
    // 插入节点
    void insert(int val) {
        root = insert(root, val);
    }
    
    // 查找节点
    bool search(int val) {
        return search(root, val) != nullptr;
    }
    
    // 删除节点
    void deleteNode(int val) {
        root = deleteNode(root, val);
    }
    
    // 中序遍历（BST的中序遍历是有序的）
    void inorder() {
        inorder(root);
        cout << endl;
    }
    
    // 验证是否为有效的二叉搜索树
    bool isValidBST() {
        return isValidBST(root, nullptr, nullptr);
    }
    
    // 查找最小值
    int findMin() {
        TreeNode* minNode = findMin(root);
        return minNode ? minNode->val : -1;
    }
    
    // 查找最大值
    int findMax() {
        TreeNode* maxNode = findMax(root);
        return maxNode ? maxNode->val : -1;
    }
    
    TreeNode* getRoot() { return root; }
    
    ~BinarySearchTree() {
        destroyTree(root);
    }
    
private:
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
    
    TreeNode* deleteNode(TreeNode* root, int val) {
        if (!root) return root;
        
        if (val < root->val) {
            root->left = deleteNode(root->left, val);
        } else if (val > root->val) {
            root->right = deleteNode(root->right, val);
        } else {
            if (!root->left) return root->right;
            if (!root->right) return root->left;
            
            TreeNode* minNode = findMin(root->right);
            root->val = minNode->val;
            root->right = deleteNode(root->right, minNode->val);
        }
        
        return root;
    }
    
    TreeNode* findMin(TreeNode* root) {
        while (root && root->left) {
            root = root->left;
        }
        return root;
    }
    
    TreeNode* findMax(TreeNode* root) {
        while (root && root->right) {
            root = root->right;
        }
        return root;
    }
    
    void inorder(TreeNode* root) {
        if (!root) return;
        inorder(root->left);
        cout << root->val << " ";
        inorder(root->right);
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
    
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
};

// 测试函数
int main() {
    cout << "=== 二叉树测试 ===" << endl;
    BinaryTree bt;
    
    vector<int> values = {1, 2, 3, 4, 5, 6, 7};
    for (int val : values) {
        bt.insert(val);
    }
    
    cout << "前序遍历（递归）: ";
    bt.preorder(bt.getRoot());
    cout << endl;
    
    cout << "前序遍历（迭代）: ";
    vector<int> preorderResult = bt.preorderIterative(bt.getRoot());
    for (int val : preorderResult) {
        cout << val << " ";
    }
    cout << endl;
    
    cout << "中序遍历（递归）: ";
    bt.inorder(bt.getRoot());
    cout << endl;
    
    cout << "中序遍历（迭代）: ";
    vector<int> inorderResult = bt.inorderIterative(bt.getRoot());
    for (int val : inorderResult) {
        cout << val << " ";
    }
    cout << endl;
    
    cout << "后序遍历（递归）: ";
    bt.postorder(bt.getRoot());
    cout << endl;
    
    cout << "后序遍历（迭代）: ";
    vector<int> postorderResult = bt.postorderIterative(bt.getRoot());
    for (int val : postorderResult) {
        cout << val << " ";
    }
    cout << endl;
    
    cout << "层序遍历: ";
    bt.levelOrder();
    cout << endl;
    
    cout << "树的高度: " << bt.height(bt.getRoot()) << endl;
    cout << "节点总数: " << bt.countNodes(bt.getRoot()) << endl;
    cout << "是否平衡: " << (bt.isBalanced(bt.getRoot()) ? "是" : "否") << endl;
    
    cout << "\n=== 二叉排序树测试 ===" << endl;
    BinarySearchTree bst;
    
    vector<int> bstValues = {5, 3, 7, 2, 4, 6, 8, 1, 9};
    for (int val : bstValues) {
        bst.insert(val);
    }
    
    cout << "BST中序遍历: ";
    bst.inorder();
    
    cout << "查找节点4: " << (bst.search(4) ? "找到" : "未找到") << endl;
    cout << "查找节点9: " << (bst.search(9) ? "找到" : "未找到") << endl;
    cout << "查找节点10: " << (bst.search(10) ? "找到" : "未找到") << endl;
    
    cout << "最小值: " << bst.findMin() << endl;
    cout << "最大值: " << bst.findMax() << endl;
    
    cout << "是否为有效BST: " << (bst.isValidBST() ? "是" : "否") << endl;
    
    cout << "删除节点3后中序遍历: ";
    bst.deleteNode(3);
    bst.inorder();
    
    cout << "删除节点7后中序遍历: ";
    bst.deleteNode(7);
    bst.inorder();
    
    return 0;
}
```

## 代码说明

### 主要功能

1. **二叉树类 (BinaryTree)**
   - 层序插入节点
   - 四种遍历方式（前序、中序、后序、层序）
   - 递归和迭代两种实现方式
   - 计算树高度和节点总数
   - 验证平衡二叉树

2. **二叉排序树类 (BinarySearchTree)**
   - 插入、查找、删除操作
   - 查找最小值和最大值
   - 验证BST有效性
   - 中序遍历（结果有序）

### 编译和运行

```bash
g++ -o tree_structures tree_structures_complete_code.cpp
./tree_structures
```

### 预期输出

```
**二叉树测试**
前序遍历（递归）: 1 2 4 5 3 6 7 
前序遍历（迭代）: 1 2 4 5 3 6 7 
中序遍历（递归）: 4 2 5 1 6 3 7 
中序遍历（迭代）: 4 2 5 1 6 3 7 
后序遍历（递归）: 4 5 2 6 7 3 1 
后序遍历（迭代）: 4 5 2 6 7 3 1 
层序遍历: 1 2 3 4 5 6 7 
树的高度: 3
节点总数: 7
是否平衡: 是

**二叉排序树测试**
BST中序遍历: 1 2 3 4 5 6 7 8 9 
查找节点4: 找到
查找节点9: 找到
查找节点10: 未找到
最小值: 1
最大值: 9
是否为有效BST: 是
删除节点3后中序遍历: 1 2 4 5 6 7 8 9 
删除节点7后中序遍历: 1 2 4 5 6 8 9 
```

## 时间复杂度分析

| 操作 | 二叉树 | 二叉排序树 |
|------|--------|------------|
| 插入 | O(n) | O(log n) |
| 查找 | O(n) | O(log n) |
| 删除 | O(n) | O(log n) |
| 遍历 | O(n) | O(n) |

*注：BST的时间复杂度在平衡情况下为O(log n)，最坏情况为O(n)*

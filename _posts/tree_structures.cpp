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
    void preorderRecursive(TreeNode* node) {
        if (!node) return;
        cout << node->val << " ";
        preorderRecursive(node->left);
        preorderRecursive(node->right);
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
    void inorderRecursive(TreeNode* node) {
        if (!node) return;
        inorderRecursive(node->left);
        cout << node->val << " ";
        inorderRecursive(node->right);
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
    void postorderRecursive(TreeNode* node) {
        if (!node) return;
        postorderRecursive(node->left);
        postorderRecursive(node->right);
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
    
    // 判断是否为完全二叉树
    bool isCompleteTree(TreeNode* root) {
        if (!root) return true;
        
        queue<TreeNode*> q;
        q.push(root);
        bool foundNull = false;
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (!node) {
                foundNull = true;
            } else {
                if (foundNull) return false;
                q.push(node->left);
                q.push(node->right);
            }
        }
        
        return true;
    }
    
    // 获取根节点
    TreeNode* getRoot() { return root; }
    
    // 销毁树
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
    
    ~BinaryTree() {
        destroyTree(root);
    }
};

// 二叉排序树类
class BinarySearchTree {
private:
    TreeNode* root;
    
public:
    BinarySearchTree() : root(nullptr) {}
    
    // 插入节点
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
    
    void insert(int val) {
        root = insert(root, val);
    }
    
    // 查找节点
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
    
    bool search(int val) {
        return search(root, val) != nullptr;
    }
    
    // 找到最小值节点
    TreeNode* findMin(TreeNode* root) {
        while (root && root->left) {
            root = root->left;
        }
        return root;
    }
    
    // 删除节点
    TreeNode* deleteNode(TreeNode* root, int val) {
        if (!root) return root;
        
        if (val < root->val) {
            root->left = deleteNode(root->left, val);
        } else if (val > root->val) {
            root->right = deleteNode(root->right, val);
        } else {
            // 找到要删除的节点
            if (!root->left) {
                TreeNode* temp = root->right;
                delete root;
                return temp;
            } else if (!root->right) {
                TreeNode* temp = root->left;
                delete root;
                return temp;
            }
            
            // 节点有两个子节点，找到右子树的最小值
            TreeNode* temp = findMin(root->right);
            root->val = temp->val;
            root->right = deleteNode(root->right, temp->val);
        }
        
        return root;
    }
    
    void deleteNode(int val) {
        root = deleteNode(root, val);
    }
    
    // 验证是否为有效的BST
    bool isValidBST(TreeNode* root) {
        return isValidBST(root, nullptr, nullptr);
    }
    
private:
    bool isValidBST(TreeNode* root, TreeNode* minNode, TreeNode* maxNode) {
        if (!root) return true;
        
        if ((minNode && root->val <= minNode->val) || 
            (maxNode && root->val >= maxNode->val)) {
            return false;
        }
        
        return isValidBST(root->left, minNode, root) && 
               isValidBST(root->right, root, maxNode);
    }
    
public:
    // 获取根节点
    TreeNode* getRoot() { return root; }
    
    // 销毁树
    void destroyTree(TreeNode* node) {
        if (node) {
            destroyTree(node->left);
            destroyTree(node->right);
            delete node;
        }
    }
    
    ~BinarySearchTree() {
        destroyTree(root);
    }
};

// 测试函数
void testBinaryTree() {
    cout << "=== 二叉树测试 ===" << endl;
    BinaryTree bt;
    
    // 插入节点
    vector<int> values = {1, 2, 3, 4, 5, 6, 7};
    for (int val : values) {
        bt.insert(val);
    }
    
    cout << "前序遍历（递归）: ";
    bt.preorderRecursive(bt.getRoot());
    cout << endl;
    
    cout << "中序遍历（递归）: ";
    bt.inorderRecursive(bt.getRoot());
    cout << endl;
    
    cout << "后序遍历（递归）: ";
    bt.postorderRecursive(bt.getRoot());
    cout << endl;
    
    cout << "树的高度: " << bt.height(bt.getRoot()) << endl;
    cout << "节点总数: " << bt.countNodes(bt.getRoot()) << endl;
    cout << "是否为完全二叉树: " << (bt.isCompleteTree(bt.getRoot()) ? "是" : "否") << endl;
}

void testBinarySearchTree() {
    cout << "\n=== 二叉排序树测试 ===" << endl;
    BinarySearchTree bst;
    
    // 插入节点
    vector<int> values = {5, 3, 7, 2, 4, 6, 8};
    for (int val : values) {
        bst.insert(val);
    }
    
    cout << "中序遍历（BST的中序遍历是有序的）: ";
    bst.inorderRecursive(bst.getRoot());
    cout << endl;
    
    cout << "查找节点4: " << (bst.search(4) ? "找到" : "未找到") << endl;
    cout << "查找节点9: " << (bst.search(9) ? "找到" : "未找到") << endl;
    
    cout << "删除节点3后中序遍历: ";
    bst.deleteNode(3);
    bst.inorderRecursive(bst.getRoot());
    cout << endl;
}

int main() {
    testBinaryTree();
    testBinarySearchTree();
    return 0;
}

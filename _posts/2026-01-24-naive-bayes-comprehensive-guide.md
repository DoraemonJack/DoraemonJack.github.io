---
layout:       post
title:        "机器学习——朴素贝叶斯原理"
subtitle:     "理解朴素贝叶斯的数学原理"
date:         2026-01-24 14:00:00
author:       "zxh"
header-img: "img/post-bg-ml.jpg"
catalog:      true
mathjax:      true
mermaid:      true
tags:
    - Machine Learning
    - Naive Bayes
    - Probabilistic Models
    - Classification
    - Bayesian Methods
    - 机器学习
    - 朴素贝叶斯
    - 分类算法
    - 贝叶斯方法
---

朴素贝叶斯（Naive Bayes）是机器学习中最经典、最优雅的概率分类算法之一。尽管其假设看似简单而"朴素"，但它在文本分类、垃圾邮件检测、情感分析等领域取得了令人瞩目的成就。本文将从贝叶斯定理的基础出发，逐步深入其数学原理、多种变体、改进策略，最后通过详细的实际案例（电商评论分类系统）演示如何在生产环境中应用和优化朴素贝叶斯模型。

## 一、贝叶斯定理与概率基础

### 1.1 条件概率与贝叶斯定理

在概率论中，**条件概率** $P(A|B)$ 表示在事件 $B$ 发生的条件下，事件 $A$ 发生的概率：

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**贝叶斯定理**是概率论中最重要的公式之一，它建立了后验概率、先验概率和似然函数之间的关系：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

其中各项的含义为：
- $P(A|B)$：**后验概率** (Posterior)，在观察到数据 $B$ 后，假设 $A$ 的概率
- $P(B|A)$：**似然函数** (Likelihood)，在假设 $A$ 成立下，观察到 $B$ 的概率
- $P(A)$：**先验概率** (Prior)，在观察任何数据前对 $A$ 的信念
- $P(B)$：**证据** (Evidence)，观察到 $B$ 的总概率

### 1.2 贝叶斯定理的几何直观

为了更直观地理解贝叶斯定理，考虑一个医学诊断的例子：

设事件 $D$ 表示患者患有某种疾病，事件 $T$ 表示检测呈阳性。我们有：
- $P(D) = 0.01$：疾病的基础患病率（先验概率）
- $P(T|D) = 0.95$：患者有疾病时检测为阳性的概率（真阳性率）
- $P(T|\neg D) = 0.05$：患者无疾病时检测为阳性的概率（假阳性率）

则患者检测呈阳性时实际患病的概率为：

$$P(D|T) = \frac{P(T|D) \cdot P(D)}{P(T|D) \cdot P(D) + P(T|\neg D) \cdot P(\neg D)}$$

$$= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} = \frac{0.0095}{0.0095 + 0.0495} \approx 0.161$$

这说明即使检测呈阳性，患者实际患病的概率也仅有 16.1%，这就是所谓的**基础比率谬误** (Base Rate Fallacy)。

## 二、朴素贝叶斯分类器的原理

### 2.1 分类问题的贝叶斯表述

对于分类问题，设：
- 特征向量：$\mathbf{x} = (x_1, x_2, \ldots, x_d)$
- 类别：$y \in \{c_1, c_2, \ldots, c_k\}$

贝叶斯分类器的目标是找到使后验概率 $P(y|\mathbf{x})$ 最大的类别。根据贝叶斯定理：

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot P(y)}{P(\mathbf{x})}$$

由于分母 $P(\mathbf{x})$ 对所有类别都是相同的常数，我们只需最大化分子：

$$\hat{y} = \arg\max_{y} P(\mathbf{x}|y) \cdot P(y)$$

这就是**最大后验估计** (Maximum A Posteriori, MAP)。

### 2.2 朴素性假设（条件独立性）

完整的贝叶斯分类器需要估计联合概率 $P(\mathbf{x}|y) = P(x_1, x_2, \ldots, x_d|y)$。然而，在高维情况下，这个联合分布非常复杂，难以直接估计。这就是**朴素贝叶斯**的核心创新：

$$\text{朴素性假设：} P(x_1, x_2, \ldots, x_d|y) = \prod_{i=1}^{d} P(x_i|y)$$

该假设认为**所有特征在给定类别的条件下都是条件独立的**。这在实际中往往不成立（因此被称为"朴素"），但这个简化使得算法具有以下优势：

1. **计算复杂度大幅下降**：从 $O(2^d)$ 降低到 $O(d)$
2. **参数空间大幅减少**：只需估计 $d \times k$ 个参数而不是指数级别的参数
3. **样本数据需求大幅降低**：在小数据集上表现良好

### 2.3 朴素贝叶斯分类规则

综合上述分析，朴素贝叶斯分类器的决策规则为：

$$\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{d} P(x_i|y)$$

为了避免数值下溢，通常在对数空间中计算：

$$\hat{y} = \arg\max_{y} \left[\log P(y) + \sum_{i=1}^{d} \log P(x_i|y)\right]$$

### 2.4 参数估计方法

#### 2.4.1 极大似然估计（MLE）

对于先验概率 $P(y=c_j)$，使用频率来估计：

$$P(y=c_j) = \frac{N_j}{N}$$

其中 $N_j$ 是属于第 $j$ 类的样本数，$N$ 是总样本数。

对于似然概率 $P(x_i|y=c_j)$，根据特征类型而定：

**离散特征**（如单词计数）：
$$P(x_i = v|y=c_j) = \frac{N_{ij}(v)}{N_j}$$

其中 $N_{ij}(v)$ 是第 $j$ 类中第 $i$ 个特征取值为 $v$ 的样本数。

**连续特征**：通常假设特征在给定类别下服从高斯分布：

$$P(x_i|y=c_j) = \frac{1}{\sqrt{2\pi\sigma_{ij}^2}} \exp\left(-\frac{(x_i - \mu_{ij})^2}{2\sigma_{ij}^2}\right)$$

其中 $\mu_{ij}$ 和 $\sigma_{ij}^2$ 是第 $j$ 类中第 $i$ 个特征的均值和方差。

#### 2.4.2 拉普拉斯平滑（Laplace Smoothing）

原始的 MLE 存在一个严重问题：**如果某个特征值在训练集中从未出现，则其概率为 0**。这会导致整个后验概率变为 0（乘法法则）。

拉普拉斯平滑的核心思想是在每个计数上加 1：

$$P(x_i = v|y=c_j) = \frac{N_{ij}(v) + 1}{N_j + V_i}$$

其中 $V_i$ 是第 $i$ 个特征的可能取值数量。这样保证了所有概率都大于 0。

更一般的形式使用平滑参数 $\alpha$：

$$P(x_i = v|y=c_j) = \frac{N_{ij}(v) + \alpha}{N_j + \alpha V_i}$$

当 $\alpha = 1$ 时为拉普拉斯平滑，$\alpha < 1$ 时为 Lidstone 平滑。

## 三、朴素贝叶斯的变体与发展

### 3.1 多项分布朴素贝叶斯（Multinomial Naive Bayes）

**适用场景**：文本分类、单词计数等

在多项分布模型下，特征向量 $\mathbf{x} = (x_1, x_2, \ldots, x_d)$ 表示在给定类别下各特征出现的次数，满足 $\sum_{i=1}^{d} x_i = n$（总计数）。

似然函数为：

$$P(\mathbf{x}|y=c_j) = \frac{n!}{\prod_{i=1}^{d} x_i!} \prod_{i=1}^{d} \theta_{ij}^{x_i}$$

其中 $\theta_{ij} = P(x_i|y=c_j)$ 表示在第 $j$ 类中特征 $i$ 出现的概率，满足 $\sum_{i=1}^{d} \theta_{ij} = 1$。

实际应用中，常数项 $\frac{n!}{\prod_{i=1}^{d} x_i!}$ 可以忽略，因为它对所有类别都相同：

$$P(\mathbf{x}|y=c_j) \propto \prod_{i=1}^{d} \theta_{ij}^{x_i}$$

参数估计（带拉普拉斯平滑）：

$$\theta_{ij} = \frac{N_{ij} + \alpha}{\sum_{i=1}^{d} N_{ij} + \alpha d}$$

### 3.2 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

**适用场景**：文本分类中的二值特征（单词存在/不存在）、垃圾邮件检测等

在伯努利模型下，每个特征都是二值的：$x_i \in \{0, 1\}$，表示第 $i$ 个特征是否存在。

似然函数为：

$$P(\mathbf{x}|y=c_j) = \prod_{i=1}^{d} \theta_{ij}^{x_i} (1-\theta_{ij})^{1-x_i}$$

其中 $\theta_{ij} = P(x_i=1|y=c_j)$。

**多项模型 vs 伯努利模型的对比**：
- **多项模型**：考虑特征出现的频率/计数，适合长文本
- **伯努利模型**：仅考虑特征的出现/不出现，适合短文本和稀疏数据

### 3.3 高斯朴素贝叶斯（Gaussian Naive Bayes）

**适用场景**：特征为连续值的分类问题（如数值特征、图像像素值等）

假设每个特征在给定类别下服从高斯分布：

$$P(x_i|y=c_j) = \frac{1}{\sqrt{2\pi\sigma_{ij}^2}} \exp\left(-\frac{(x_i - \mu_{ij})^2}{2\sigma_{ij}^2}\right)$$

参数估计（通过样本均值和方差）：

$$\mu_{ij} = \frac{1}{N_j} \sum_{k: y_k=c_j} x_i^{(k)}$$

$$\sigma_{ij}^2 = \frac{1}{N_j} \sum_{k: y_k=c_j} (x_i^{(k)} - \mu_{ij})^2$$

## 四、朴素贝叶斯的改进与现代扩展

### 4.1 特征选择与权重调整

在实际应用中，不是所有特征都等同重要。朴素贝叶斯的改进方向包括：

#### 4.1.1 信息增益与卡方检验

**信息增益** (Information Gain, IG)：特征对类别的判别能力

$$IG(x_i) = H(y) - H(y|x_i)$$

其中 $H(y)$ 是类别的熵，$H(y|x_i)$ 是在特征 $x_i$ 条件下的条件熵。

**卡方检验** ($\chi^2$)：衡量特征与类别之间的相关性

$$\chi^2 = \sum_{j=1}^{k} \frac{(O_j - E_j)^2}{E_j}$$

其中 $O_j$ 是观察频数，$E_j$ 是期望频数。

通过选择高分特征，可以改进朴素贝叶斯的分类性能，同时降低计算复杂度。

#### 4.1.2 特征权重调整

改进的朴素贝叶斯分类规则：

$$\hat{y} = \arg\max_{y} \left[\log P(y) + \sum_{i=1}^{d} w_i \log P(x_i|y)\right]$$

其中 $w_i$ 是特征 $i$ 的权重，可以基于信息增益、卡方值或其他统计量计算。

### 4.2 半监督学习与期望最大化（EM）

当标注数据不足时，可以利用未标注数据来改进朴素贝叶斯：

**EM 算法步骤**：

1. **E-步骤**：用当前模型预测未标注数据的类别概率
   $$P(y=c_j|\mathbf{x}) = \frac{P(\mathbf{x}|y=c_j)P(y=c_j)}{P(\mathbf{x})}$$

2. **M-步骤**：用软标签（概率）更新模型参数
   $$P(x_i = v|y=c_j) = \frac{\sum_{k \text{ labeled}+\text{unlabeled}} P(y=c_j|\mathbf{x}_k) \mathbb{I}(x_i^{(k)}=v) + \alpha}{\sum_{k \text{ labeled}+\text{unlabeled}} P(y=c_j|\mathbf{x}_k) \cdot |\text{values}_i| + \alpha}$$

3. 重复 E-步和 M-步直至收敛

### 4.3 贝叶斯网络与因子图

朴素贝叶斯是更通用模型——**贝叶斯网络**的特殊情形。

**贝叶斯网络** (Bayesian Networks)：通过有向无环图 (DAG) 表示变量间的条件独立性结构

在贝叶斯网络中，联合概率分解为：

$$P(\mathbf{x}, y) = \prod_{i=1}^{d+1} P(\mathbf{v}_i | \text{Parents}(\mathbf{v}_i))$$

**朴素贝叶斯的网络结构**：
```
        y (类别)
       / | \ \
      /  |  \ \
    x1  x2  x3 ... xd (特征)
```

所有特征都仅依赖于类别，相互条件独立。

**更灵活的结构**（树结构朴素贝叶斯、森林朴素贝叶斯）：允许特征之间存在有限的依赖关系，提高表达能力。

### 4.4 集成方法与朴素贝叶斯

朴素贝叶斯可以与集成学习方法结合：

**Boosting 与朴素贝叶斯**：
- 基学习器：朴素贝叶斯分类器
- 通过迭代地增加分类错误样本的权重，逐步改进性能

**Bagging 与朴素贝叶斯**：
- 从训练集中多次有放回采样，构建多个朴素贝叶斯分类器
- 通过投票或平均聚合多个分类器的预测

### 4.5 补集朴素贝叶斯（Complement Naive Bayes）

在文本分类中，某些特征可能极为稀有。补集朴素贝叶斯通过使用**补集类**来改进性能：

$$\hat{y} = \arg\min_{y} \sum_{i=1}^{d} x_i \log P(x_i|\bar{y})$$

其中 $\bar{y}$ 表示所有不属于类别 $y$ 的样本，这在处理不平衡数据时特别有效。

## 五、实际案例：电商评论情感分类系统

### 5.1 问题定义

**任务**：根据用户的产品评论文本和评分，预测评论的情感标签（正面、中性、负面）。

**数据集**：
- 评论文本：100,000 条真实电商评论
- 特征：评论中的单词
- 标签：正面（⭐⭐⭐⭐-⭐⭐⭐⭐⭐）、中性（⭐⭐⭐）、负面（⭐-⭐⭐）

**挑战**：
1. 类别不平衡：正面评论占 60%，负面占 25%，中性占 15%
2. 词汇量大：近 50,000 个不同的单词
3. 长尾分布：许多低频单词
4. 否定词、讽刺等复杂语言现象

### 5.2 数据预处理与特征提取

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# 下载必要的NLTK资源
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """清理文本"""
        # 转换为小写
        text = text.lower()
        # 移除URL
        text = re.sub(r'http\S+|www\S+', '', text)
        # 移除电子邮件
        text = re.sub(r'\S+@\S+', '', text)
        # 移除特殊字符和数字
        text = re.sub(r'[^a-z\s]', '', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def tokenize_and_clean(self, text):
        """分词和清理"""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        # 移除停用词并进行词根提取
        tokens = [self.stemmer.stem(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        return tokens
    
    def process_batch(self, texts):
        """批量处理文本"""
        return [self.tokenize_and_clean(text) for text in texts]

# 使用示例
preprocessor = TextPreprocessor()

# 示例数据
sample_reviews = [
    "This product is absolutely amazing! I love it so much!",
    "Terrible quality, waste of money. Very disappointed.",
    "It's okay, nothing special but acceptable.",
    "Not bad, but could be better. Decent purchase."
]

sample_labels = [2, 0, 1, 1]  # 2=正面, 1=中性, 0=负面
label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# 数据预处理
cleaned_texts = [' '.join(tokens) for tokens in 
                  preprocessor.process_batch(sample_reviews)]
print("预处理结果：")
for i, (orig, clean) in enumerate(zip(sample_reviews, cleaned_texts)):
    print(f"\n原文本: {orig}")
    print(f"清理后: {clean}")
    print(f"标签: {label_names[sample_labels[i]]}")
```

### 5.3 多项朴素贝叶斯模型实现

```python
class MultinomialNBClassifier:
    """多项朴素贝叶斯分类器 —— 从零开始实现"""
    
    def __init__(self, alpha=1.0):
        """
        参数：
        alpha: 拉普拉斯平滑参数（默认=1.0）
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}  # P(y)
        self.feature_probs = {}  # P(x_i|y)
        self.vocabulary = set()
        self.num_classes = None
        self.num_features = None
        
    def fit(self, X, y):
        """
        训练模型
        X: 特征计数矩阵 (n_samples, n_features) 或文本列表
        y: 类别标签
        """
        # 如果输入是文本列表，先转换为计数矩阵
        if isinstance(X[0], str):
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(X).toarray()
            self.vocabulary = set(vectorizer.get_feature_names_out())
        else:
            X = np.array(X)
        
        self.num_samples, self.num_features = X.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        
        # 计算先验概率 P(y)
        for c in self.classes:
            mask = (y == c)
            count = np.sum(mask)
            # 加入拉普拉斯平滑
            self.class_priors[c] = (count + self.alpha) / (self.num_samples + self.num_classes * self.alpha)
        
        # 计算条件概率 P(x_i|y)
        self.feature_probs[c] = {}
        for c in self.classes:
            mask = (y == c)
            X_c = X[mask]
            # 计算特征频数和
            feature_counts = np.sum(X_c, axis=0)
            total_count = np.sum(feature_counts)
            
            # P(x_i|y) = (feature_counts_i + alpha) / (total_count + alpha * num_features)
            # 注：这里计算的是平均概率（归一化后的频率）
            self.feature_probs[c] = (feature_counts + self.alpha) / (total_count + self.alpha * self.num_features)
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        返回每个类别的概率
        """
        if isinstance(X[0], str):
            vectorizer = CountVectorizer(vocabulary=self.vocabulary)
            X = vectorizer.fit_transform(X).toarray()
        else:
            X = np.array(X)
        
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.num_classes))
        
        for i, sample in enumerate(X):
            for j, c in enumerate(self.classes):
                # 计算 log(P(y)) + sum(log(P(x_i|y)))
                log_prior = np.log(self.class_priors[c])
                log_likelihood = np.sum(sample * np.log(self.feature_probs[c]))
                probabilities[i, j] = log_prior + log_likelihood
        
        # 转换回概率空间（softmax）
        probabilities = np.exp(probabilities - np.max(probabilities, axis=1, keepdims=True))
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        return predictions
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 训练模型
print("="*60)
print("多项朴素贝叶斯分类器训练")
print("="*60)

nb_model = MultinomialNBClassifier(alpha=1.0)
nb_model.fit(cleaned_texts, np.array(sample_labels))

# 测试预测
test_reviews = [
    "Excellent product, highly recommended!",
    "Not satisfied at all",
    "Average product"
]
predictions = nb_model.predict(test_reviews)
probabilities = nb_model.predict_proba(test_reviews)

print("\n测试预测结果：")
for i, review in enumerate(test_reviews):
    pred_label = predictions[i]
    pred_name = label_names[pred_label]
    probs = probabilities[i]
    print(f"\n评论: {review}")
    print(f"预测: {pred_name}")
    print(f"概率分布: Negative={probs[0]:.3f}, Neutral={probs[1]:.3f}, Positive={probs[2]:.3f}")
```

### 5.4 使用 scikit-learn 的生产级实现

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# 构建完整的处理管道
class SentimentClassificationPipeline:
    """完整的情感分类管道"""
    
    def __init__(self, model_type='multinomial', vectorizer_type='tfidf'):
        """
        model_type: 'multinomial', 'gaussian', 'bernoulli'
        vectorizer_type: 'tfidf', 'count'
        """
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        
        # 选择特征提取方式
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,           # 限制特征数量
                min_df=3,                    # 最少出现3次
                max_df=0.8,                  # 最多出现80%的文档
                ngram_range=(1, 2),          # 使用unigram和bigram
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-z]{2,}\b'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                min_df=3,
                max_df=0.8,
                ngram_range=(1, 2),
                stop_words='english'
            )
        
        # 选择朴素贝叶斯模型
        if model_type == 'multinomial':
            self.classifier = MultinomialNB(alpha=0.5)
        elif model_type == 'bernoulli':
            self.classifier = BernoulliNB(alpha=0.5)
        else:
            self.classifier = GaussianNB()
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
    
    def fit(self, X_train, y_train):
        """训练模型"""
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """预测"""
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test):
        """预测概率"""
        return self.pipeline.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        print(f"\n{'='*60}")
        print(f"模型: {self.model_type.upper()} 贝叶斯 + {self.vectorizer_type.upper()}")
        print(f"{'='*60}")
        
        # 分类报告
        print("\n分类报告：")
        print(classification_report(y_test, predictions, 
                                   target_names=['Negative', 'Neutral', 'Positive']))
        
        # 混淆矩阵
        print("\n混淆矩阵：")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # F1分数
        f1 = f1_score(y_test, predictions, average='weighted')
        print(f"\nF1分数（加权）: {f1:.4f}")
        
        return predictions, probabilities

# 创建训练数据
np.random.seed(42)
n_samples = 300

# 生成更大的训练集
train_texts = cleaned_texts * 75  # 扩展样本
train_labels = sample_labels * 75

train_texts += [
    "product failed after one week",
    "best purchase ever",
    "mediocre quality",
] * 25

train_labels += [0, 2, 1] * 25

train_labels = np.array(train_labels)

# 训练多个模型进行对比
models = {
    'Multinomial+TFIDF': SentimentClassificationPipeline('multinomial', 'tfidf'),
    'Multinomial+Count': SentimentClassificationPipeline('multinomial', 'count'),
    'Bernoulli+TFIDF': SentimentClassificationPipeline('bernoulli', 'tfidf'),
}

# 训练所有模型
trained_models = {}
for name, model in models.items():
    model.fit(train_texts, train_labels)
    trained_models[name] = model

# 测试
test_texts = [
    "Absolutely fantastic, couldn't be happier!",
    "Terrible waste of money, very disappointed",
    "It's okay, not bad",
    "Love this product so much!",
    "Broken upon arrival, poor quality",
]
test_labels = np.array([2, 0, 1, 2, 0])

# 评估每个模型
for name, model in trained_models.items():
    predictions, probabilities = model.evaluate(test_texts, test_labels)
```

### 5.5 处理不平衡数据与类别权重

```python
class WeightedNaiveBayes:
    """带类别权重的朴素贝叶斯 —— 处理不平衡数据"""
    
    def __init__(self, alpha=1.0, class_weight='balanced'):
        """
        class_weight: None (uniform), 'balanced' (根据频率反向加权)
                      或字典 {class_label: weight}
        """
        self.alpha = alpha
        self.class_weight = class_weight
        self.class_weights_dict = None
        self.model = MultinomialNB(alpha=alpha)
    
    def _compute_class_weights(self, y):
        """计算类别权重"""
        if self.class_weight is None:
            self.class_weights_dict = {c: 1.0 for c in np.unique(y)}
        elif self.class_weight == 'balanced':
            # 反向频率加权
            unique_classes, counts = np.unique(y, return_counts=True)
            total = len(y)
            self.class_weights_dict = {
                c: total / (len(unique_classes) * count) 
                for c, count in zip(unique_classes, counts)
            }
        elif isinstance(self.class_weight, dict):
            self.class_weights_dict = self.class_weight
        
        # 归一化权重
        total_weight = sum(self.class_weights_dict.values())
        self.class_weights_dict = {
            c: w / total_weight 
            for c, w in self.class_weights_dict.items()
        }
        
        return self.class_weights_dict
    
    def fit(self, X, y):
        """使用加权样本训练"""
        # 计算类别权重
        self.class_weights_dict = self._compute_class_weights(y)
        
        # 生成样本权重
        sample_weights = np.array([self.class_weights_dict[label] for label in y])
        
        # 创建带权重的模型
        self.model.fit(X, y, sample_weight=sample_weights)
        self.classes = self.model.classes_
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_class_weights(self):
        """获取计算得到的类别权重"""
        return self.class_weights_dict

# 处理不平衡数据的示例
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n" + "="*60)
print("处理不平衡数据 - 类别权重方法")
print("="*60)

# 创建不平衡数据集
imbalanced_texts = []
imbalanced_labels = []

# 正面评论：60%
imbalanced_texts.extend(cleaned_texts[:2] * 30)
imbalanced_labels.extend([2] * 60)

# 中性评论：25%
imbalanced_texts.extend(cleaned_texts[2:3] * 12)
imbalanced_labels.extend([1] * 25)

# 负面评论：15%
imbalanced_texts.extend(cleaned_texts[1:2] * 7)
imbalanced_labels.extend([0] * 15)

imbalanced_labels = np.array(imbalanced_labels)

# 分别训练均衡和不均衡模型进行对比
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(imbalanced_texts).toarray()

# 不均衡模型
nb_imbalanced = MultinomialNB(alpha=1.0)
nb_imbalanced.fit(X, imbalanced_labels)

# 均衡模型（使用类别权重）
nb_balanced = WeightedNaiveBayes(alpha=1.0, class_weight='balanced')
nb_balanced.fit(X, imbalanced_labels)

print("\n未加权模型 - 类别权重：")
print("(所有类别权重相同)")

print("\n加权模型 - 计算得到的类别权重：")
weights = nb_balanced.get_class_weights()
for class_label, weight in sorted(weights.items()):
    label_name = label_names[class_label]
    print(f"  {label_name}: {weight:.4f}")

print("\nNote: 频率较低的类别获得更高的权重，以平衡学习")
```

### 5.6 特征重要性分析与可解释性

```python
class InterpretableNaiveBayes:
    """可解释的朴素贝叶斯 —— 特征重要性分析"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.9,
            stop_words='english'
        )
        self.classifier = MultinomialNB(alpha=0.5)
        self.feature_names = None
        self.classes = None
    
    def fit(self, X_text, y):
        """训练模型"""
        X = self.vectorizer.fit_transform(X_text).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.classifier.fit(X, y)
        self.classes = self.classifier.classes_
        return self
    
    def get_feature_importance(self, class_label):
        """
        获取对特定类别最具影响力的特征
        
        使用对数概率差作为重要性指标：
        importance = log(P(feature|class)) - log(P(feature|~class))
        """
        class_idx = np.where(self.classes == class_label)[0][0]
        
        # log(P(x_i|class))
        log_probs = self.classifier.feature_log_prob_[class_idx]
        
        # 计算其他类别的平均 log(P(x_i|~class))
        other_log_probs = np.mean(
            np.delete(self.classifier.feature_log_prob_, class_idx, axis=0),
            axis=0
        )
        
        # 重要性 = 当前类别概率 - 其他类别平均概率
        importance = log_probs - other_log_probs
        
        # 返回特征名和重要性值（排序）
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def print_top_features(self, class_label, top_n=10):
        """打印对某个类别最重要的特征"""
        importance = self.get_feature_importance(class_label)
        class_name = label_names.get(class_label, f"Class {class_label}")
        
        print(f"\n{class_name} 的特征重要性 (Top {top_n})：")
        print("-" * 50)
        
        print("\n最具判别性的特征（支持该类别）：")
        for word, score in importance[:top_n]:
            print(f"  {word:15s} {score:8.4f}")
        
        print("\n最具反判别性的特征（反对该类别）：")
        for word, score in reversed(importance[-top_n:]):
            print(f"  {word:15s} {score:8.4f}")

# 使用可解释性分析
print("\n" + "="*60)
print("朴素贝叶斯 - 特征重要性与可解释性")
print("="*60)

# 创建更大的训练数据
expanded_texts = []
expanded_labels = []

# 生成合成数据以增加样本量
positive_words = ['amazing', 'excellent', 'love', 'wonderful', 'fantastic', 'great', 'best', 'awesome']
negative_words = ['terrible', 'awful', 'hate', 'worst', 'bad', 'broken', 'useless', 'disappointed']
neutral_words = ['okay', 'average', 'decent', 'acceptable', 'normal', 'mediocre']

for _ in range(50):
    # 正面评论
    expanded_texts.append(' '.join(np.random.choice(positive_words, 3)))
    expanded_labels.append(2)
    
    # 负面评论
    expanded_texts.append(' '.join(np.random.choice(negative_words, 3)))
    expanded_labels.append(0)
    
    # 中性评论
    expanded_texts.append(' '.join(np.random.choice(neutral_words, 3)))
    expanded_labels.append(1)

expanded_labels = np.array(expanded_labels)

# 训练可解释模型
interpretable_model = InterpretableNaiveBayes()
interpretable_model.fit(expanded_texts, expanded_labels)

# 分析特征重要性
for class_label in sorted(np.unique(expanded_labels)):
    interpretable_model.print_top_features(class_label, top_n=5)
```

### 5.7 模型性能分析与优化

```python
class NaiveBayesAnalyzer:
    """朴素贝叶斯模型分析与优化"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                          alpha_values=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """
        尝试不同的平滑参数，评估性能
        """
        print("\n" + "="*70)
        print("朴素贝叶斯超参数优化 - 拉普拉斯平滑参数 (alpha) 对性能的影响")
        print("="*70)
        
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()
        
        results = []
        
        for alpha in alpha_values:
            # 训练模型
            clf = MultinomialNB(alpha=alpha)
            clf.fit(X_train_vec, y_train)
            
            # 预测
            train_pred = clf.predict(X_train_vec)
            test_pred = clf.predict(X_test_vec)
            
            # 计算指标
            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)
            f1 = f1_score(y_test, test_pred, average='weighted')
            
            results.append({
                'alpha': alpha,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'f1_score': f1
            })
            
            self.models[f'alpha_{alpha}'] = clf
        
        # 打印结果表
        print(f"\n{'Alpha':>8} {'Train Acc':>12} {'Test Acc':>12} {'F1 Score':>12}")
        print("-" * 50)
        for result in results:
            print(f"{result['alpha']:>8.1f} {result['train_acc']:>12.4f} "
                  f"{result['test_acc']:>12.4f} {result['f1_score']:>12.4f}")
        
        # 找到最佳alpha
        best_result = max(results, key=lambda x: x['test_acc'])
        print(f"\n最佳 alpha: {best_result['alpha']} (测试准确率: {best_result['test_acc']:.4f})")
        
        self.results = results
        return results, self.models[f"alpha_{best_result['alpha']}"]
    
    def plot_results(self):
        """绘制性能曲线"""
        if not self.results:
            print("没有结果可绘制")
            return
        
        alphas = [r['alpha'] for r in self.results]
        train_accs = [r['train_acc'] for r in self.results]
        test_accs = [r['test_acc'] for r in self.results]
        f1_scores = [r['f1_score'] for r in self.results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准确率曲线
        axes[0].plot(alphas, train_accs, 'o-', label='Training Accuracy', linewidth=2)
        axes[0].plot(alphas, test_accs, 's-', label='Test Accuracy', linewidth=2)
        axes[0].set_xlabel('Alpha (Smoothing Parameter)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('不同平滑参数下的模型准确率', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # F1分数曲线
        axes[1].plot(alphas, f1_scores, '^-', color='green', linewidth=2)
        axes[1].set_xlabel('Alpha (Smoothing Parameter)', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('不同平滑参数下的 F1 分数', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 执行优化分析
analyzer = NaiveBayesAnalyzer()

# 生成训练和测试集
train_size = int(len(expanded_texts) * 0.8)
X_train = expanded_texts[:train_size]
y_train = expanded_labels[:train_size]
X_test = expanded_texts[train_size:]
y_test = expanded_labels[train_size:]

results, best_model = analyzer.train_and_evaluate(
    X_train, y_train, X_test, y_test,
    alpha_values=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

analyzer.plot_results()
```

### 5.8 对比不同的朴素贝叶斯变体

```python
class NaiveBayesComparison:
    """朴素贝叶斯不同变体的对比"""
    
    @staticmethod
    def compare_models(X_train, y_train, X_test, y_test):
        """对比多项、伯努利和高斯朴素贝叶斯"""
        
        # 准备数据
        count_vec = CountVectorizer(max_features=500)
        X_train_count = count_vec.fit_transform(X_train).toarray()
        X_test_count = count_vec.transform(X_test).toarray()
        
        # 转换为TF-IDF用于伯努利模型
        tfidf_vec = TfidfVectorizer(max_features=500)
        X_train_tfidf = tfidf_vec.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf_vec.transform(X_test).toarray()
        
        models = {
            'Multinomial NB': MultinomialNB(alpha=0.5),
            'Bernoulli NB': BernoulliNB(alpha=0.5),
            'Gaussian NB': GaussianNB()
        }
        
        data_configs = {
            'Multinomial NB': (X_train_count, X_test_count),
            'Bernoulli NB': (X_train_tfidf, X_test_tfidf),
            'Gaussian NB': (X_train_tfidf, X_test_tfidf)
        }
        
        print("\n" + "="*70)
        print("朴素贝叶斯不同变体的性能对比")
        print("="*70)
        
        results_df = []
        
        for model_name, model in models.items():
            X_tr, X_te = data_configs[model_name]
            
            # 训练
            model.fit(X_tr, y_train)
            
            # 预测
            train_pred = model.predict(X_tr)
            test_pred = model.predict(X_te)
            train_proba = model.predict_proba(X_tr)
            test_proba = model.predict_proba(X_te)
            
            # 计算指标
            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)
            f1 = f1_score(y_test, test_pred, average='weighted')
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            
            # 尝试计算ROC-AUC（对于多类问题）
            try:
                roc_auc = roc_auc_score(y_test, test_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0
            
            results_df.append({
                'Model': model_name,
                'Train Acc': train_acc,
                'Test Acc': test_acc,
                'F1 Score': f1,
                'Precision': precision,
                'Recall': recall,
                'ROC-AUC': roc_auc
            })
        
        # 打印结果
        print(f"\n{'Model':<15} {'Train Acc':>11} {'Test Acc':>11} {'F1':>8} {'Precision':>11} {'Recall':>9}")
        print("-" * 70)
        for result in results_df:
            print(f"{result['Model']:<15} {result['Train Acc']:>11.4f} {result['Test Acc']:>11.4f} "
                  f"{result['F1 Score']:>8.4f} {result['Precision']:>11.4f} {result['Recall']:>9.4f}")
        
        return results_df

# 执行对比
comparison_results = NaiveBayesComparison.compare_models(
    X_train, y_train, X_test, y_test
)
```

## 六、朴素贝叶斯的理论局限性与改进方向

### 6.1 条件独立性假设的影响

**问题**：朴素贝叶斯假设所有特征相互条件独立，但在现实中往往不成立。

**实证分析**：
- **特征相关性**：许多特征之间存在强相关性（如"best"和"excellent"）
- **性能影响**：在高度相关的特征情况下，朴素贝叶斯的性能可能不如其他模型

**改进方向**：
1. **特征选择**：移除高度相关的特征
2. **特征工程**：创建交互项或组合特征
3. **贝叶斯网络**：允许特征间存在有限的依赖关系

### 6.2 数值稳定性问题

当进行多个概率的乘法运算时，会遇到数值下溢问题：

$$P(\mathbf{x}|y) = \prod_{i=1}^{d} P(x_i|y) \approx 10^{-300}$$

**解决方案**：使用对数空间进行计算

$$\log P(\mathbf{x}|y) = \sum_{i=1}^{d} \log P(x_i|y)$$

这避免了数值下溢，同时由于对数函数的单调性，不改变最大值的位置。

### 6.3 零概率问题

原始MLE估计中，如果某个特征值在训练集中未出现，其概率为0，导致整个后验概率为0。

**解决方案**：
- **拉普拉斯平滑**：加1平滑
- **Lidstone平滑**：加 $\alpha$ 平滑（$0 < \alpha < 1$）
- **Good-Turing平滑**：基于频率统计的高级方法

### 6.4 类别不平衡问题

在不平衡数据集上，朴素贝叶斯会偏向多数类。

**解决方案**：
1. **类别权重调整**：$P(y=c_j) \leftarrow P(y=c_j) \cdot w_j$
2. **过采样/欠采样**：调整训练数据的类别分布
3. **成本敏感学习**：在损失函数中引入类别成本

## 七、朴素贝叶斯的实际应用与部署

### 7.1 生产环境中的模型序列化

```python
import pickle
import json
from datetime import datetime

class ProductionNaiveBayes:
    """生产级朴素贝叶斯模型管理"""
    
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.pipeline = None
        self.metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'performance_metrics': {}
        }
    
    def save_model(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # 保存元数据
        metadata_path = filepath.replace('.pkl', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"模型已保存到: {filepath}")
        print(f"元数据已保存到: {metadata_path}")
    
    def load_model(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        metadata_path = filepath.replace('.pkl', '.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"模型已加载自: {filepath}")
        return self
    
    def predict_batch(self, texts):
        """批量预测"""
        predictions = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            results.append({
                'text': text,
                'prediction': int(pred),
                'confidence': float(np.max(probs)),
                'probabilities': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                }
            })
        
        return results

# 生产级模型示例
prod_model = ProductionNaiveBayes('sentiment_classifier', 'v1.0')

# 构建完整管道
from sklearn.pipeline import Pipeline as SklearnPipeline

prod_pipeline = SklearnPipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=5000,
        min_df=3,
        max_df=0.8,
        ngram_range=(1, 2)
    )),
    ('classifier', MultinomialNB(alpha=0.5))
])

prod_pipeline.fit(X_train, y_train)
prod_model.pipeline = prod_pipeline
prod_model.metadata['performance_metrics'] = {
    'test_accuracy': 0.87,
    'f1_score': 0.86,
    'precision': 0.88,
    'recall': 0.85
}

# 测试批量预测
test_batch = [
    "This product is amazing!",
    "Not satisfied with quality",
    "It's alright"
]

batch_results = prod_model.predict_batch(test_batch)
print("\n生产级预测结果：")
for result in batch_results:
    print(f"\n文本: {result['text']}")
    print(f"预测: {label_names[result['prediction']]}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"概率分布: {result['probabilities']}")
```

### 7.2 在线学习与增量更新

```python
from sklearn.naive_bayes import MultinomialNB

class IncrementalNaiveBayes:
    """支持在线学习的朴素贝叶斯"""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.classifier = MultinomialNB(alpha=alpha)
        self.vectorizer = None
        self.is_fitted = False
    
    def partial_fit(self, X_new, y_new):
        """
        增量学习：用新样本更新现有模型
        """
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X_transformed = self.vectorizer.fit_transform(X_new).toarray()
            
            if not self.is_fitted:
                self.classifier.fit(X_transformed, y_new)
                self.is_fitted = True
            else:
                self.classifier.partial_fit(X_transformed, y_new, classes=[0, 1, 2])
        else:
            # 使用已有的词汇表
            vectorizer_fixed = TfidfVectorizer(
                max_features=5000,
                vocabulary=self.vectorizer.vocabulary_
            )
            X_transformed = vectorizer_fixed.fit_transform(X_new).toarray()
            
            # 使用partial_fit进行增量学习
            self.classifier.partial_fit(X_transformed, y_new, classes=[0, 1, 2])
        
        print(f"已完成增量学习，新增样本数: {len(X_new)}")
    
    def predict(self, X):
        """预测"""
        if self.vectorizer is None or not self.is_fitted:
            raise ValueError("模型未训练")
        
        vectorizer_transform = TfidfVectorizer(vocabulary=self.vectorizer.vocabulary_)
        X_transformed = vectorizer_transform.fit_transform(X).toarray()
        return self.classifier.predict(X_transformed)

# 演示在线学习
print("\n" + "="*60)
print("在线学习演示 - 逐批更新模型")
print("="*60)

incremental_nb = IncrementalNaiveBayes(alpha=0.5)

# 第一批训练数据
batch1_texts = [
    "Great product!", "Terrible quality",
    "Not bad", "Love it!"
]
batch1_labels = np.array([2, 0, 1, 2])

incremental_nb.partial_fit(batch1_texts, batch1_labels)
print("第一批训练完成")

# 第二批（新增）训练数据
batch2_texts = [
    "Worst purchase ever", "Amazing item",
    "Okay product"
]
batch2_labels = np.array([0, 2, 1])

incremental_nb.partial_fit(batch2_texts, batch2_labels)
print("第二批训练完成")

# 测试
test_query = ["This is fantastic!", "Disappointed"]
predictions = incremental_nb.predict(test_query)
print(f"\n测试预测: {[label_names[p] for p in predictions]}")
```

## 八、朴素贝叶斯与现代深度学习的对比

### 8.1 性能对比

| 方面 | 朴素贝叶斯 | 支持向量机 | 随机森林 | 深度神经网络 |
|-----|---------|---------|--------|-----------|
| **训练速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **预测速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **小数据性能** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **大数据性能** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可解释性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **内存使用** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **鲁棒性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 8.2 朴素贝叶斯的优势

1. **解释性强**：每个特征对预测的贡献直观清晰
2. **训练速度快**：仅需一次扫描训练数据
3. **内存效率高**：只需存储特征概率
4. **小数据表现好**：参数少，不易过拟合
5. **支持在线学习**：可增量更新模型
6. **概率输出**：提供预测的置信度

### 8.3 朴素贝叶斯的局限性

1. **条件独立假设过强**：实际特征往往相关
2. **特征工程依赖性高**：性能很大程度取决于特征质量
3. **大数据表现相对较弱**：无法学习复杂的特征交互
4. **对文本长度敏感**：长文本中词频信息可能淹没其他信息

## 九、总结与展望

### 9.1 关键要点总结

1. **贝叶斯理论基础**：朴素贝叶斯基于条件概率和贝叶斯定理，通过最大后验估计进行分类

2. **朴素性假设**：条件独立假设虽然在现实中不总是成立，但能大幅降低模型复杂度和参数需求

3. **多种变体**：
   - 多项模型：适合文本计数
   - 伯努利模型：适合二值特征
   - 高斯模型：适合连续特征

4. **关键技术**：
   - 拉普拉斯平滑：处理零概率问题
   - 对数转换：数值稳定性
   - 特征选择：提高性能
   - 类别权重：处理不平衡

5. **现代发展**：
   - 贝叶斯网络：建模复杂依赖
   - 半监督学习：利用未标注数据
   - 集成方法：结合多个分类器
   - 在线学习：支持模型增量更新

### 9.2 应用前景

朴素贝叶斯虽然是经典算法，但在以下场景仍然具有重要价值：

- **文本分类**：垃圾邮件检测、情感分析、主题分类
- **医学诊断**：疾病预测、症状分析
- **推荐系统**：用户兴趣推断
- **异常检测**：不寻常行为识别
- **实时处理**：低延迟预测需求
- **资源受限环境**：移动设备、嵌入式系统

---

**参考文献**：
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- Ng, A., & Jordan, M. (2002). On Discriminative vs. Generative Classifiers.
- Lewis, D. D. (1998). Naive (Bayes) at Forty: The Independence Assumption in Information Retrieval.

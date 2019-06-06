# Mining Notes
> 一些个人笔记

## 测试

### 回归随机集

- sklearn.datasets.make_regression(*n_samples=100*, *n_features=100*, *n_informative=10*, *n_targets=1*, *bias=0.0*, *effective_rank=None*, *tail_strength=0.5*, *noise=0.0*, *shuffle=True*, *coef=False*, *random_state=None*)
- 源：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression

```
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression

# X为样本特征，Y为样本输出，coef为回归系数，共1000个样本，每个样本1个特征
X, Y, coef = make_regression(n_samples=1000, n_features=1,noise=10, coef=True)
plt.scatter(X, Y, color='black')
plt.plot(X, X*coef, color='blue',linewidth=3)
```



### 分类随机集

- sklearn.datasets.make_classification(*n_samples=100*, *n_features=20*, *n_informative=2*, *n_redundant=2*, *n_repeated=0*, *n_classes=2*, *n_clusters_per_class=2*, *weights=None*, *flip_y=0.01*, *class_sep=1.0*, *hypercube=True*, *shift=0.0*, *scale=1.0*, *shuffle=True*, *random_state=None*)
- 源：https://scikit-learn.org/stable/modules/generated/ sklearn.datasets.make_classification.html

```
from sklearn.datasets import make_classification

# X为样本特征，Y为样本类别输出，共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X, Y = make_classification(n_samples=400, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
```



### 聚类随机集

- sklearn.datasets.make_blobs(*n_samples=100*, *n_features=2*, *centers=None*, *cluster_std=1.0*, *center_box=(-10.0*, *10.0)*, *shuffle=True*, *random_state=None*)
- 源：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

```
from sklearn.datasets import make_blobs

# X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]，簇方差分别为[0.4, 0.5, 0.2]
X, Y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
```



### 正太分布随机集

- sklearn.datasets.make_gaussian_quantiles(*mean=None*, *cov=1.0*, *n_samples=100*, *n_features=2*, *n_classes=3*, *shuffle=True*, *random_state=None*)
- 源：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html

```
from sklearn.datasets import make_gaussian_quantiles

#生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X, Y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
```



## 回归

### 梯度下降

- 算法原理：
  - 偏导数 --> 梯度 --> 梯度下降（/上升）方向就是最优解存在方向 --> 梯度向量——数学的诠释方式
  - https://www.jianshu.com/p/c7e642877b0e
  - https://www.cnblogs.com/pinard/p/5970503.html
- 分类：
  - 批量梯度下降（BGD）
  - 随机梯度下降（SGD）
  - 小批量梯度下降

```
def error_function(theta, X, y):
	diff = np.dot(X, theta) - y
	m = len(X)
	return (1.0 / (2 * m)) * np.dot(np.transpose(diff), diff)

def BGD_gradient_function(theta, X, y):
	diff = np.dot(X, theta) - y
	m = len(X)
	return (1.0 / m) * np.dot(np.transpose(X),diff)

def SGD_gradient_function(theta, X, y):
	m = len(X)
	i = np.random.randint(0, m)
	diff = np.dot(X[i], theta) - y[i]
	return (np.transpose(X[i]) * diff).reshape(-1,1)

def gradient_descent(init_theta, X, y, alpha, BGD_SGD = False):
	theta = init_theta
	gradient_function = lambda theta, X, y: SGD_gradient_function(theta, X, y) if BGD_SGD else BGD_gradient_function(theta, X, y)
	gradient = gradient_function(theta, X, y)
	while not np.all(np.absolute(gradient) <= EPSILON):
		theta = theta - alpha * gradient
		gradient = gradient_function(theta, X, y)
	return theta
```



### 最小二乘法

- 用途：
  - 函数拟合、求函数极值
  - 优点：快
  - 缺点：特征值过大时（阈值：10000）将消耗大量计算资源
  - 局限性：
    - 拟合函数若不是线性的，需要一些技巧转化成线性
    - $X^T * X$的逆矩阵有可能不存在
- 算法原理：
  - <https://www.cnblogs.com/pinard/p/5976811.html>
  - 矩阵求导：https://www.jianshu.com/p/4128e5b31fb4

```
def least_squares(X,Y):
	X = np.mat(X)
	Y = np.mat(Y)
	theta = (X.T * X).I * X.T * Y
	# Alternative: theta = X.I * X.T.I * X.T * Y
	# 求逆矩阵的操作是计算瓶颈，经测试，原式子可能会比Alternative更快，但不明显
	return theta
```

- 非线性转化技巧：
  - 若拟合曲线为$y = \theta_0 + \theta_1 * x + \theta_2 * x^2$，$X$矩阵可转换成$(x_0, x_1, x_2), x1 = x, x_2 = x^2$，$\theta$矩阵为$(\theta_0, \theta_1, \theta_2)$，如此类推。



### 线性回归

- 算法原理：https://www.cnblogs.com/pinard/p/6004041.html
- 正则化：
  - L1正则化：Lasso回归，$J(θ)=\frac{1}{2}(Xθ−Y)^{T}(Xθ−Y)+α||θ||_{1}$
  - L2正则化：Ridge回归，$J(θ)=\frac{1}{2}(Xθ−Y)^{T}(Xθ−Y)+\frac{1}{2}α||θ||^{2}_{2}$
- Python库：sklearn.linear_model.LinearRegression

```
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_, linreg.coef_
```



### Lasso回归





### Ridge回归





### 交叉验证

- 用途：小规模数据训练优化选择模型
- 分类：
  - 简单交叉验证：多次随机切割样本，7成训练集/3成测试集
  - S折交叉验证：n次(n<S)随机样本为S份，S-1份训练集/1份测试集
  - 留一交叉验证：S折交叉验证的特例，S=样本数N，适用于样本数极少的情况

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)	# 简单交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/S, random_state=2)	# S折交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=101)	# 留一交叉验证
```







## 关联

### Apriori

- 用途：发觉频繁模式和关联规则
- 算法原理：
  - 核心是频繁项集和减枝
  - https://www.cnblogs.com/en-heng/p/5719101.html
  - https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
- Python库：Apyori
  - https://github.com/ymoch/apyori

```Python
from apyori import apriori

transactions = [['Bread','Milk'],
        ['Bread','Diapers','Beer','Eggs'],
        ['Milk','Diapers','Beer','Cola'],
        ['Bread','Milk','Diapers','Beer'],
        ['Bread','Milk','Beer','Cola']]

result = list(apriori(transactions, min_support=0.6, min_confidence=1.0, max_length=2))

for row in result:
    print list(row.items)
    for rule in row.ordered_statistics:
        print list(rule.items_base), "-->", list(rule.items_add), "support:", row.support, "confidence:", rule.confidence, "lift:", rule.lift
```

```std
['Beer', 'Diapers']
['Diapers'] --> ['Beer'] support: 0.6 confidence: 1.0 lift: 1.25
```



### FP-growth

- 用途：发掘频繁模式
- 算法原理：
  - 核心是构建频繁模式树
  - https://blog.csdn.net/baixiangxue/article/details/80335469
- Python库：fp-growth
  - https://pypi.org/project/fp-growth/#description
  - https://github.com/enaeseth/python-fp-growth/

```python
from fp_growth import find_frequent_itemsets

for itemset in find_frequent_itemsets(transactions, minsup):
    print itemset
```



### Eclat

- 用途：对特定类型数据源（低频繁度）效率会更佳
- 算法原理：
  - 核心是倒排思想，从水平数据格式转化成垂直数据格式，适合低频率密度场景，交运算是算法速率瓶颈
  - https://blog.csdn.net/my_learning_road/article/details/79728389
- 算法实现：
  - http://adrem.uantwerpen.be/~goethals/software/

```python
def eclat(prefix, items, minsup,frequent_itemsets):
	while items:
		i,itids = items.pop()
		isupp = len(itids)
		if isupp >= minsup:
			frequent_itemsets.append(prefix+[i])
			suffix = [] 
			for j, ojtids in items:
				jtids = itids & ojtids
				if len(jtids) >= minsup:
					suffix.append((j,jtids))
			eclat(prefix+[i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), minsup, frequent_itemsets)

def eclat_frequent_itemsets(transactions, minsup):
	frequent_itemsets = []
	data = {}
	trans = 0
	for row in transactions:
		trans += 1
		for item in row:
			if item not in data:
				data[item] = set()
			data[item].add(trans)
	eclat([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), minsup, frequent_itemsets)
	return frequent_itemsets
```



### CCA

- 用途：挖掘多维数据的关联关系
- 算法原理：
  - 核心是协方差和奇异值分解
  - https://www.cnblogs.com/pinard/p/6288716.html
- Python库：sklearn
  - sklearn.cross_decomposition.CCA
  - https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html



## 分类

### 决策树

- 用途：可以用作分类，也可以用作回归
- 算法原理：
  - 核心是信息的度量——信息熵&基尼指数
  - https://www.cnblogs.com/pinard/p/6050306.html
  - 大杂烩：https://github.com/ljpzzz/machinelearning#8
- 演进：
  - ID3 --> C4.5 --> CART
- Python库：sklearn
  - BUG：sklearn只能进行数值型运算，不能处理字符串样本和结果，必须先进行样本序列化。
  - sklearn.tree.DecisionTreeClassifier
  - sklearn.tree.DecisionTreeRegressor
- 实践心得：
  - DTC与DTR的区别：DTC是基于gini指数，DTR是基于MSE（均方误差），DTR能很好地处理连续值，DTC实践中发现不能处理连续值
  - 控制决策树的深度：在探索过程中，max_depth最好控制在3以下
- 基础版本实现：

```
from sklearn import tree
from pydotplus import graph_from_dot_data as dot2graph

X = [[28],[25],[19],[18],[40],[45],[60],[33],[20]]	#learning data
Y = [1,1,0,0,1,0,0,1,0]

clf = tree.DecisionTreeClassifier()	#default criterion='gini'
#clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X,Y)

test = [[22],[16],[23],[42],[43],[59]]	#testing data
print test,"\n",clf.predict(test)

dot = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
graph = dot2graph(dot)
graph.write_pdf("tree_test.pdf")
```

- 序列化实现：

```
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from pydotplus import graph_from_dot_data as dot2graph

df = pd.DataFrame({
		'Outlook':['Sunny','Sunny','Cloudy','Sunny','Cloudy','Cloudy','Sunny'],
		'Temp':['Cold','Warm','Warm','Warm','Cold','Cold','Cold'],
		'Routine':['InDoor','OutDoor','InDoor','InDoor','InDoor','OutDoor','OutDoor'],
		'WearCoat':['No','No','No','No','Yes','Yes','Yes']
		})	#testing data

le = dict()
for col in ['Outlook','Temp','Routine']:
	le[col] = LabelEncoder().fit_transform(df[col])
le = pd.DataFrame(le)
X = list(le.values)
Y = list(LabelEncoder().fit_transform(df['WearCoat']))	#Serializing

clf = DecisionTreeClassifier()	#default criterion='gini'
#clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

dot = export_graphviz(clf, out_file=None, 
		feature_names=le.columns, 
		class_names=LabelEncoder().fit(df['WearCoat']).classes_, 
		filled=True, rounded=True, special_characters=True)
graph = dot2graph(dot)
graph.write_pdf("gini_tree_test2.pdf")
#graph.write_pdf("entropy_tree_test2.pdf")
```

- 回归实现：

```
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf = clf.fit(X,Y) 
predict = clf.predict(test)
```



### 朴素Bayes

python包：sklearn

```
from sklearn.naive_bayes import GaussianNB
```



### K临近算法(KNN)

python包：sklearn

```
from sklearn import cross_validation, neighbors
neighbors.KNeighborsClassifier()
```



### Logistic回归

python包：sklearn

```
from sklearn.linear_model import LogisticRegression
```



### SVM

```
from sklearn import svm
```



### 随机森林

```
from sklearn.ensemble import RandomForestClassifier
```



### 回归

```
from sklearn import linear_model
```



### K-means

```
from sklearn.cluster import Kmeans
```



## 降维

### 主成分分析（PCA）

- 用途：从高维数据到低维数据的转换，如从二维数据到一维数据
- 算法原理：
  - 降维的本质：高维度样本点到低维度超平面的距离足够近,或者说样本点在这个超平面上的投影能尽可能的分开（方差）。
  - PCA选择样本点投影具有最大方差的方向
  - https://www.cnblogs.com/pinard/p/6239403.html
- 算法实现：
  - Python库：sklearn.decomposition.PCA

```
# coding:utf-8
from sklearn.decomposition import PCA

X = [[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]]
pca = PCA(n_components=1)
X2 = pca.fit_transform(X)
print X2
```



### 线性判别分析（LDA）

- 用途：降维+分类

- 算法原理：

  - 目测特性：降维时类别之间的距离更明显
  - LDA选择分类性能最好的投影方向
  - 与PCA的不同：LDA是有监督的降维方法
  - https://www.cnblogs.com/pinard/p/6244265.html

- 算法实现：

  - Python库：sklearn.discriminant_analysis.LinearDiscriminantAnalysis
  - https://www.cnblogs.com/pinard/p/6249328.html

  ```
  import numpy as np
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from sklearn.datasets.samples_generator import make_classification
  
  X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, n_informative=2,n_clusters_per_class=1,class_sep =0.5, random_state =10)
  fig = plt.figure(1)
  ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
  ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
  
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(X)
  print pca.explained_variance_ratio_
  print pca.explained_variance_
  X_new = pca.transform(X)
  plt.figure(2)
  plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
  
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  lda = LinearDiscriminantAnalysis(n_components=2)
  lda.fit(X,y)
  X_new = lda.transform(X)
  plt.figure(3)
  plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
  plt.show()
  ```



### 奇异值分解(SVD)

- 用途：数据压缩和去噪，可用于特征分解、推荐算法、自然语言处理等
- 原理：
  - 核心是特征值和特征向量
  - SVD的性质：降序的奇异值矩阵减少特快，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。
  - https://www.cnblogs.com/pinard/p/6251584.html
- 实现：
  - Python库：numpy.linalg.svd

```
import numpy as np

x = [[1,1,1,0,0],[2,2,2,0,0],[1,1,1,0,0],[5,5,5,0,0],[1,1,0,2,2],[0,0,0,3,3],[0,0,0,1,1]]
X = np.mat(x)	# X = np.array(x)
U, sigma, VT = np.linalg.svd(X)	#numpy计算出来的VT是V的转置

#numpy计算出来的是Sigma矩阵的对角线压缩数组，要用diag函数还原
S = np.diag(sigma)
for i in range(len(VT),len(U)):
	S = np.row_stack((S,np.zeros(len(VT))))
S = np.mat(S)	# S is already np.array type

X_origin = U * S * VT	#还原原矩阵
X_similar = U[:,0:3] * S[0:3,0:3] * VT[0:3,:]	#近似矩阵

# Alternative codes
# X_orgin = np.dot(U,S).dot(VT)
# X_similar = np.dot(U[:,0:3],S[0:3,0:3]).dot(VT[0:3,:])
```

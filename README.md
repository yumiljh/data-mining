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

https://www.jianshu.com/p/c7e642877b0e

https://www.cnblogs.com/pinard/p/5970503.html

```
>>> X
array([[ 1.19304077],
       [ 0.94091142],
       [ 0.02339566],
       [-0.8981732 ],
       [ 0.09943588],
       [ 0.00383871],
       [-0.80779895],
       [ 0.87421378],
       [ 2.3114854 ],
       [ 0.76986146],
       [ 0.05000725],
       [ 1.58109966],
       [ 2.82690925],
       [-0.96564311],
       [-0.16234562],
       [ 0.43630434],
       [ 0.76292887],
       [-2.69115914],
       [-1.2516014 ],
       [-1.65755253]])
>>> Y
array([ 30.16130442,  36.19745451,  17.10101282,  -7.65571261,
        26.76324042,  -5.03382704, -21.33951587,  22.67316442,
        54.01695548,  23.33351057,   2.82356836,  25.49128867,
        91.31885123, -25.95196596,  10.49228655,   4.74371435,
         4.69407196, -90.81921826, -13.93506075, -34.20976885])
>>> coef
array(24.821098110495655)
```







## 关联

### Apriori

- 用途：发觉频繁模式和关联规则
- 算法原理：
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
  - https://www.cnblogs.com/pinard/p/6288716.html
- Python库：sklearn
  - sklearn.cross_decomposition.CCA
  - https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html



## 分类

### 决策树

- 用途：可以用作分类，也可以用作回归
- 算法原理：
  - https://www.cnblogs.com/pinard/p/6050306.html
  - 大杂烩：https://github.com/ljpzzz/machinelearning#8
- 演进：
  - ID3 --> C4.5 --> CART
- Python库：sklearn
  - BUG：sklearn只能进行数值型运算，不能处理字符串样本和结果，必须先进行样本序列化。
  - sklearn.tree.DecisionTreeClassifier
  - sklearn.tree.DecisionTreeRegressor
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
  - SVD的性质：降序的奇异值矩阵减少特快，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。
  - https://www.cnblogs.com/pinard/p/6251584.html
- 实现：
  - Python库：numpy.linalg.svd

```
import numpy as np

x = [[1,1,1,0,0],[2,2,2,0,0],[1,1,1,0,0],[5,5,5,0,0],[1,1,0,2,2],[0,0,0,3,3],[0,0,0,1,1]]
X = np.mat(x)
U, sigma, VT = np.linalg.svd(X)	#numpy计算出来的VT是V的转置

#numpy计算出来的是Sigma矩阵的对角线压缩数组，要用diag函数还原
zero_stack = np.zeros(len(V))
S = np.diag(sigma)
for i in range(len(V),len(U)):
	S = np.row_stack((S,zero_stack))
S = np.mat(S)

X_origin = U * S * VT	#还原原矩阵
X_similar = U[:,0:3] * S[0:3,0:3] * VT[0:3,:]	#近似矩阵
```


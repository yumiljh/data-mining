# Mining Notes
> 一些个人笔记

## 回归

### 梯度下降





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





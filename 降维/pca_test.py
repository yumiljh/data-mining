# coding:utf-8

from sklearn.decomposition import PCA

X = [[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,0.9]]

pca = PCA(n_components=1)
X2 = pca.fit_transform(X)
print X2

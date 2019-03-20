# coding:utf-8

import os
from sklearn import tree
from pydotplus import graph_from_dot_data as dot2graph

X = [[28],[25],[19],[18],[40],[45],[60],[33],[20]]
Y = [1,1,0,0,1,0,0,1,0]

clf = tree.DecisionTreeClassifier()	#default criterion='gini'
#clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X,Y)

test = [[22],[16],[23],[42],[43],[59]]
print test,"\n",clf.predict(test)

dot = tree.export_graphviz(clf, out_file=None, feature_names=['Age'], filled=True, rounded=True, special_characters=True)
graph = dot2graph(dot)
graph.write_pdf("gini_tree_test.pdf")
#graph.write_pdf("entropy_tree_test.pdf")


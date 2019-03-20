# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from pydotplus import graph_from_dot_data as dot2graph


df = pd.DataFrame({
		'Outlook':['Sunny','Sunny','Cloudy','Sunny','Cloudy','Cloudy','Sunny'],
		'Temp':['Cold','Warm','Warm','Warm','Cold','Cold','Cold'],
		'Routine':['InDoor','OutDoor','InDoor','InDoor','InDoor','OutDoor','OutDoor'],
		'WearCoat':['No','No','No','No','Yes','Yes','Yes']
		})

le = dict()
dictX = dict()
for col in ['Outlook','Temp','Routine']:
	le[col] = LabelEncoder()
	dictX[col] = le[col].fit_transform(df[col])
pdX = pd.DataFrame(dictX)
X = list(pdX.values)

le['WearCoat'] = LabelEncoder()
Y = list(le['WearCoat'].fit_transform(df['WearCoat']))

print pdX.columns
test = [[1,1,0]]
clf = DecisionTreeRegressor()
clf = clf.fit(X,Y)
predict = clf.predict(test)
predict = map(int, predict)
print le['WearCoat'].inverse_transform(predict)

'''
clf = DecisionTreeClassifier()	#default criterion='gini'
#clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

dot = export_graphviz(clf, out_file=None, 
		feature_names=pdX.columns, 
		class_names=le['WearCoat'].classes_, 
		filled=True, rounded=True, special_characters=True)
graph = dot2graph(dot)
graph.write_pdf("gini_tree_test2.pdf")
#graph.write_pdf("entropy_tree_test2.pdf")
'''

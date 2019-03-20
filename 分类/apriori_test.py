#-*- coding:utf-8 -*-
from apyori import apriori

transactions = [['Bread','Milk'],
        ['Bread','Diapers','Beer','Eggs'],
        ['Milk','Diapers','Beer','Cola'],
        ['Bread','Milk','Diapers','Beer'],
        ['Bread','Milk','Beer','Cola']]

#result = list(apriori(transactions, min_support=0.6, min_confidence=1.0, max_length=2))
#result = list(apriori(transactions))
result = list(apriori(transactions, min_support=0.4, min_confidence=1.0))

'''
for row in result:
    print list(row.items)
    for rule in row.ordered_statistics:
        print list(rule.items_base), "-->", list(rule.items_add), "support:", row.support, "confidence:", rule.confidence, "lift:", rule.lift
'''

def get_result():
	output = []
	for row in result:
		output.append(str(list(row.items)))
		for rule in row.ordered_statistics:
			s = ""
			s = s + str(list(rule.items_base)) + "-->" + str(list(rule.items_add)) + ", support:" + str(row.support) + ", confidence:" + str(rule.confidence) + ", lift:" + str(rule.lift)
			output.append(s)
	return output


#-*- conding:utf-8 -*-

#from apriori_test import transactions
from fp_growth_test import transactions

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

def get_result():
	return eclat_frequent_itemsets(transactions, 3)


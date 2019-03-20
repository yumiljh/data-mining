#-*- coding:utf-8 -*-

#from apriori_test import transactions
from fp_growth import find_frequent_itemsets


transactions = [['I1','I2','I5'],
			 ['I2','I4'],
			 ['I2','I3'],
			 ['I1','I2','I4'],
			 ['I1','I3'],
			 ['I2','I3'],
			 ['I1','I3'],
			 ['I1','I2','I3','I5'],
			 ['I1','I2','I3']
	]

'''
for minsup in range(1,7):
	itemsets = find_frequent_itemsets(transactions, minsup)
	print "find_frequent_itemsets(transactions,", minsup, "):"
	for item in itemsets:
		print item
'''

def get_result():
	return find_frequent_itemsets(transactions, 2)

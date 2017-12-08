import pandas as pd
import numpy as np
import pickle
import csv
import operator
import sys
import re
csv.field_size_limit(sys.maxsize)

def save_obj(obj, name ):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

words = load_obj("tokenDictionary")

#---------------------testing data reading----------
reader = csv.reader(open(sys.argv[1], "r"))
data = []
for idx, row in enumerate(reader):
	tmp = ''
	for idx, val in enumerate(row):
		if idx != 0:
			tmp = tmp + val
	data.append(tmp)
	
	#print(row[1])

#---------------------testing data writing----------
with open('./EMD_testing_data.csv', 'w') as f:
	#pickle.dump(data, f)
	spamWriter = csv.writer(f)
	for idx, val in enumerate(data):
		if idx > 0:
			#print('val: ', val)
			tmp = val.split(" ")
			#print('idx, tmp : ', idx, tmp)
			#print(tmp[0], tmp[2])
			#print(tmp[1])
			tmp_words = tmp#.split(' ')
			tmp = []
			for word in tmp_words:
				try:
					tmp.append(words[word])
				except:
					tmp.append(0)
					pass
			spamWriter.writerow(tmp)

import pandas as pd
import numpy as np
import pickle
import csv
import operator
import sys
import re
csv.field_size_limit(sys.maxsize)

#---------------------training data reading----------
data = []
reader = csv.reader(open(sys.argv[1], "r"))
for idx, row in enumerate(reader):
	tmp = ''
	#print(len(row))
	for idx, val in enumerate(row):
		#tmp = val.split(" ", 2)
		#print(idx, tmp)
		if idx == 0:
			tmp = tmp + val
		if idx > 0:
			tmp = tmp +','+ val
		#print(idx, tmp)
		#data.append(val[2])

	data.append(tmp)
	#if idx > 500:
	#	exit()
print(len(data))
alldata = data[:]

'''
reader = csv.reader(open("./training_nolabel.txt", "r"))
for idx, row in enumerate(reader):
	tmp = ''
	if len(row) == 0 :
		continue
	for idx, val in enumerate(row):
		#tmp = val.split(" ", 2)
		#print(idx, row)

		if idx == 0:
			tmp = "0 $$$$$ " + val
		if idx > 0:
			tmp = tmp +','+ val
		#print(idx, tmp)
		#data.append(val[2])
	#print(tmp)
	alldata.append(tmp)
'''

'''
reader = csv.reader(open("./testing_data.txt", "r"))
for idx, row in enumerate(reader):
	tmp = ''
	if len(row) == 0 :
		continue
	for idx, val in enumerate(row):
		#tmp = val.split(" ", 2)
		#print(idx, row)

		if idx == 0:
			tmp = "0 $$$$$ " + val
		if idx > 0:
			tmp = tmp +','+ val
		#print(idx, tmp)
		#data.append(val[2])
	#print(tmp)
	alldata.append(tmp)
'''
print(len(alldata))
print(len(data))
#exit()
#--------------------data preprocess------------
regex = re.compile('[^a-zA-Z0-9 ]')

for idx, row in enumerate(alldata):
	row = row.replace(" +++$+++ ", " ")
	row = row.replace(" b c ", " because ")
	row = row.replace(" can ' t", " can not")
	row = row.replace(" won ' t "," will not")
	row = row.replace("n ' t", " not")
	row = row.replace("' m", "am")
	row = row.replace("it ' s", "it is")
	row = row.replace("' ll", "will")
	row = row.replace("' re ", "are ")
	row = row.replace(" ' ve", " have")
	row = row.replace(" im ", " i am ")
	row = row.replace("ain ' t", "am not")
	row = row.replace("ain ' t", "am not")
	#row = regex.sub('', row)
	row = row.replace("  ", " ")
	row = row.replace("  ", " ")
	if idx < 1000:
		pass
	alldata[idx] = row

words = {}
for row in alldata:
	#print(row)
	#print(tmp)
	tmp = row.split(' ', 1)
	#print(tmp)
	tmp = tmp[1].split(' ')

	for val in tmp:
		try: # if it ever appeard
			words[val] = words[val] + 1 # appear times +1
		except: # first appear
			#print(val)
			try: # test if is number
				int(val)
				words[val] = 1
			except ValueError:
				words[val] = 1

sorted_x = sorted(words.items(), key=operator.itemgetter(1))
sorted_x.reverse()
words = {}
for idx, val in enumerate(sorted_x):
	if idx < 10:
		#print(val)
		pass
	words[val[0]] = idx+1

print("i love you CNN = ", end='')
print(words['i'], words['love'], words['you'], words['cnn'])

#---------------------training data writing------------
with open('./EMD_training_label.csv', 'w') as f:
	#pickle.dump(data, f)
	spamWriter = csv.writer(f)
	for idx, val in enumerate(data):
		tmp = val.split(" ", 1)
		#print(idx, tmp)
		#print(tmp[0], tmp[2])
		tmp_words = tmp[1].split(' ')
		tmp = [tmp[0]]
		for word in tmp_words:
			try:
				tmp.append(words[word])
			except:
				tmp.append(0)
				pass
		spamWriter.writerow(tmp)


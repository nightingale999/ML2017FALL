import numpy as np  
import pandas as pd  
from keras.utils import np_utils
import csv
import sys
#import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Dense, Dropout#, Flatten, Input, Activation, Reshape
from keras.layers import Bidirectional
from keras.optimizers import Adam, Adadelta#, SGD
#from keras.initializers import RandomNormal
from keras.layers.pooling import MaxPooling1D#, AveragePooling2D
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Embedding
#from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.models.word2vec import Word2Vec

#from keras.backend import round
csv.field_size_limit(sys.maxsize)

noTestingSwitch = False
trainingMode = False
loadModelPath = './model.h5'

useGenSim = False
expandDataTimes = 0
expandDataThreshold = 0.1

validSplit = 0.08
np.random.seed(1022)
embedding_vecor_length = 64
epoch = 3
batchSize = 512
top_words = 8192
max_review_length = 32

def main():
	#---------------------------Reading Test Data-------------------------------
	print("[status] Reading Test Data")
	testData = []
	if useGenSim == False:
		rawTestData = []
		reader = csv.reader(open("./EMD_testing_data.csv", "r"))
		for row in reader:
			rawTestData.append(row)
		print(len(rawTestData))
		for idx, row in enumerate(rawTestData):
			tmp = []
			for val in row:
				tmp.append(val)
			testData.append(tmp)
		X_test = sequence.pad_sequences(testData, maxlen=max_review_length)
	elif useGenSim == True:
		reader = csv.reader(open("./testing_data.txt", "r"))
		for idx, row in enumerate(reader):
			if idx > 0:
				#print(row)
				#row = row[0].split(",", 1)[1]
				tmp = []
				setence = []
				#tmp.append(row[0])
				for idx, someWords in enumerate(row):
					if idx  > 0:
						for oneWord in someWords:
							setence.append(oneWord)
				#setence = row.split(' ')
				ids = to_ids(setence)

				for value in ids:
					tmp.append(value)
				testData.append(tmp)
		X_test = sequence.pad_sequences(testData, maxlen=max_review_length)

	#--------------------------Predicting--------------------------------
	print("[status] Predicting")
	model = load_model(loadModelPath)
	y_testing_answer = []
	val_proba = model.predict(X_test, verbose=0)
	for val in val_proba:
		y_testing_answer.append(int(round(val[0])))

	#-------------------Output Predict Data--------------------------------
	print("[status] Output Predict Data")
	saveFilePath = sys.argv[1]
	with open(saveFilePath, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'label'])
		for idx, val in enumerate(y_testing_answer):
			spamwriter.writerow([idx, val])

if __name__ == "__main__":
	main()

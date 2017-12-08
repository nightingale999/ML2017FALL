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

noTestingSwitch = True
trainingMode = True
loadModelPath = './models/model_1122221_6738'

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


	#--------------------Reading Training Data-----------------------------
	print('[Status] Reading Training Data')
	rawData = []
	if useGenSim == False:
		reader = csv.reader(open("./EMD_training_label.csv", "r"))
		for idx, row in enumerate(reader):
			rawData.append(row)
		#print(rawData)
		X_train, y_train, X_test, y_test = [], [], [], []
		tmp = len(rawData) * (1-validSplit)
		#tmp = 500
		for idx, row in enumerate(rawData):
			tmpVals = []
			if idx < tmp:
				for index, val in enumerate(row):
					if index == 0:
						y_train.append(val)
					else:
						tmpVals.append(val)
				X_train.append(tmpVals)
			elif idx >= tmp:
				for index, val in enumerate(row):
					if index == 0:
						y_test.append(val)
					else:
						tmpVals.append(val)
				X_test.append(tmpVals)
		X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
		X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	elif useGenSim == True:
		reader = csv.reader(open("./training_label.txt", "r"))
		for idx, row in enumerate(reader):
			#print(row)
			row = row[0].split(" ", 2)
			tmp = []
			tmp.append(row[0])
			setence = row[2].split(' ')
			ids = to_ids(setence)

			for value in ids:
				tmp.append(value)
			rawData.append(tmp)
			'''
			if idx == 1:
				print(setence)
				print(ids)
				exit()
			'''
		print(len(rawData))
		print(rawData[1])
		X_train, y_train, X_test, y_test = [], [], [], []
		tmp = len(rawData) * (1-validSplit)
		#tmp = 500
		for idx, row in enumerate(rawData):
			tmpVals = []
			if idx < tmp:
				for index, val in enumerate(row):
					if index == 0:
						y_train.append(val)
					else:
						tmpVals.append(val)
				X_train.append(tmpVals)
			elif idx >= tmp:
				for index, val in enumerate(row):
					if index == 0:
						y_test.append(val)
					else:
						tmpVals.append(val)
				X_test.append(tmpVals)
		X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
		X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	#print(X_test[0])
	#y_test.append('1')
	#print(y_test)

	#exit()
	#--------------------Setting Model-----------------------------
	print('[Status] Setting Model')
	now = datetime.datetime.now()
	if trainingMode:
		model = Sequential()
		if useGenSim:
			gensimModel(model)
		else:
			#bidirectionalModel(model)
			originModel(model)
		#Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
		#model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
		model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
		print(model.summary())
		try:
			history_callback = model.fit(X_train, y_train, epochs=epoch, verbose=2, batch_size=batchSize, validation_split=validSplit)
		except KeyboardInterrupt :
			pass
	else :
		model = load_model(loadModelPath)
	scores = model.evaluate(X_test, y_test, verbose=2)
	print("")
	print("Origin Accuracy: %.2f%%" % (scores[1]*100))
	print("")
	#---------------------------No label data reading-----------------------------------------
	if expandDataTimes:
		print("[status] Reading No Label Data")
		rawData = []
		testData = []
		reader = csv.reader(open("./EMD_training_nolabel_data.csv", "r"))
		for row in reader:
			rawData.append(row)
		print("\t[Info] No Label Data Count : %d"% len(rawData))
		for idx, row in enumerate(rawData):
			tmp = []
			for val in row:
				tmp.append(val)
			testData.append(tmp)
		X_training_nolabel = sequence.pad_sequences(testData, maxlen=max_review_length)
	#---------------------------No label data expand-----------------------------------------
	for idx in range(expandDataTimes):
		print("[status] Expanding with no label data")
		print("\t[Info] Append Round. %d / %d" % (idx+1, expandDataTimes))
		y_testing_answer = []
		val_proba = model.predict(X_training_nolabel, verbose=1)
		#X_train_append = np.empty()

		#X_training_nolabel_tmp = np.empty([0, max_review_length], dtype=int)
		#y_training_nolabel_tmp = np.empty([0], dtype=int)
		tmp = 0
		for val in val_proba:
			if val[0]>1-expandDataThreshold or val[0] < expandDataThreshold:
				tmp+=1
		X_training_nolabel_tmp = np.zeros([tmp, max_review_length], dtype=int)
		y_training_nolabel_tmp = np.zeros([tmp], dtype=int)
		tmp = 0
		for index, val in enumerate(val_proba):
			if index % 600000 == 0:
				print("\t[Info] Augmenting Process : %d / %d"% (index, len(val_proba)))
			#print(val[0])
			if val[0] > 1-expandDataThreshold:
				X_training_nolabel_tmp[tmp] = X_training_nolabel[index]
				y_training_nolabel_tmp[tmp] = '1'
				tmp +=1
				#y_train.append('1')
			elif val[0] < expandDataThreshold:
				X_training_nolabel_tmp[tmp] = X_training_nolabel[index]
				y_training_nolabel_tmp[tmp] = '0'
				tmp +=1
		X_train_tmp = np.concatenate([X_training_nolabel_tmp, X_train])
		y_train_tmp = np.concatenate([y_training_nolabel_tmp, y_train])

		print("\t[Info] Appeneded Data Count : %d"%len(X_training_nolabel_tmp))
		#np.concatenate((X_train, X_train_append))
		#np.concatenate((y_train, y_train_append))
		model = Sequential()
		bidirectionalModel(model)
		model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
		try:
			history_callback = model.fit(X_train_tmp, y_train_tmp, epochs=epoch, batch_size=batchSize, validation_split=validSplit)
		except:
			pass
		scores = model.evaluate(X_test, y_test, verbose=2)
		print("")
		print("Final Accuracy: %.2f%%" % (scores[1]*100))
		print("")

	#-----------------Saving Model and Writing Log-------------------------------
	now = datetime.datetime.now()
	if trainingMode:
		print("[status] Saving Model & Writing Log")
		model.save('./model_%02d%02d%02d%02d_%d.h5'%(now.month, now.day, now.hour, now.minute, int(scores[1]*10000)))
	if noTestingSwitch == True:
		exit()	
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
	y_testing_answer = []
	val_proba = model.predict(X_test, verbose=0)
	for val in val_proba:
		y_testing_answer.append(int(round(val[0])))
	if trainingMode:
		file = open("./Log.txt", "a")
		file.write("\nTop twenty answers:\n")
		for val in range(20):
			file.write(str(y_testing_answer[val]) + " ")
		file.write("\n")
		file.close()

	#-------------------Output Predict Data--------------------------------
	print("[status] Output Predict Data")
	saveFilePath = './outputs/output_%02d%02d%02d%02d_%d.csv'%(now.month, now.day, now.hour, now.minute, int(scores[1]*10000))
	with open(saveFilePath, 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id', 'label'])
		for idx, val in enumerate(y_testing_answer):
			spamwriter.writerow([idx, val])

#--------------------Models--------------------------
def originModel(model):
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(Conv1D(filters=256, kernel_size=2, padding='same', activation='relu'))
	
	#model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	#model.add(Conv1D(filters=1024, kernel_size=3, padding='same', activation='relu'))
	#model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	
	model.add(Dropout(0.2))
	#model.add(LSTM(8, return_sequences=True))#, dropout=0.2, recurrent_dropout=0.2))
	#model.add(LSTM(8, return_sequences=True))
	#model.add(LSTM(8, return_sequences=True))
	#model.add(GRU(48))
	model.add(LSTM(64))#, dropout=0.4, recurrent_dropout=0.4))
	model.add(Dropout(0.2))
	
	model.add(Dense(128))
	#model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(Dense(128))
	#model.add(Dense(128))
	#model.add(Dense(128))
	#model.add(Dropout(0.2))
	#model.add(Dense(256))
	#model.add(Dense(128))
	#model.add(Dense(8))
	#model.add(Dense(128))
	#model.add(Dense(128))
	#model.add(Dense(128))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))

def bidirectionalModel(model):
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	#model.add(Conv1D(filters=128, kernel_size=2, input_shape=(184000, max_review_length), padding='same', activation='relu'))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(48)))#, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(Dense(128))
	model.add(Dense(128))
	model.add(Dense(1, activation='sigmoid'))
def gensimModel(model):
	embedding_layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights],trainable=False)
	model.add(embedding_layer)
	#model.add(Conv1D(filters=128, kernel_size=2, input_shape=(184000, max_review_length), padding='same', activation='relu'))
	model.add(Conv1D(filters=1024, kernel_size=3, padding='same', activation='relu'))
	#model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
	#model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
	model.add(Dropout(0.2))
	model.add(LSTM(48))#, dropout=0.2, return_sequences=False, recurrent_dropout=0.2))
	#model.add(GRU(48))
	model.add(Dropout(0.2))
	#model.add(Dense(128))
	#model.add(Dense(128))
	#model.add(Dense(128))
	model.add(Dense(1, activation='sigmoid'))

def to_ids(words):  
    def word_to_id(word):
        id = vocab.get(word)
        if id is None:
            id = 0
        return id

    words = list(map(word_to_id, words))
    #return np.array(x)
    return words

if __name__ == "__main__":
	main()

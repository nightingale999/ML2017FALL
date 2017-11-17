import numpy as np  
import pandas as pd  
from keras.utils import np_utils
import csv
import sys
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Reshape
from keras.optimizers import SGD, Adam, Adadelta
from keras.initializers import RandomNormal
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

np.random.seed(10)
validate_portion = 0.1
normalization = True
augmen = True
epochs = 20
#trainingSwitch = True
batch_size = 256

def main():
	'''
	#--------------------------------Reading--------------------------------
	print("[status] Reading Data")
	RawTrainingData = pd.read_csv("./train.csv")
	#print(RawTrainingData.shape)
	RawTrainingData = RawTrainingData.values

	#print(RawTrainingData[0, 1])
	dataCount = RawTrainingData.shape[0]
	training_portion = 1-validate_portion
	trainingCount = int(training_portion * RawTrainingData.shape[0])
	validatingCount = dataCount - trainingCount

	X_train_image = np.zeros((trainingCount, 48, 48))
	y_train_label = np.zeros((trainingCount,))
	X_test_image = np.zeros((validatingCount, 48, 48))
	y_test_label = np.zeros((validatingCount,))

	for idx, row in enumerate(RawTrainingData):
		x = row[1].split()
		x = np.asarray(x)
		x = np.reshape(x, (48, 48))
		if idx < trainingCount:
			X_train_image[idx] = x
			y_train_label[idx] = row[0]
		else:
			X_test_image[idx-trainingCount] = x
			y_test_label[idx-trainingCount] = row[0]

		#print(len(x))
		#print(x[0], type(x[0]))
		#exit()

	print("\t[Info] train data={:7,}".format(len(X_train_image)))  
	print("\t[Info] test  data={:7,}".format(len(X_test_image))) 

	x_Train = X_train_image.reshape(trainingCount, 48, 48, 1).astype('float32')
	x_Test = X_test_image.reshape(validatingCount, 48, 48, 1).astype('float32')
	print("\t[Info] xTrain: %s" % (str(x_Train.shape)))
	print("\t[Info] xTest: %s" % (str(x_Test.shape)))
	#exit()
	#---------------------Normalization-----------------------
	print("[status] Normalization")

	if normalization:
		x_Train_norm = x_Train/255  
		x_Test_norm = x_Test/255
	else:
		x_Train_norm = x_Train
		x_Test_norm = x_Test

	y_TrainOneHot = np_utils.to_categorical(y_train_label)
	y_TestOneHot = np_utils.to_categorical(y_test_label)
	#print(y_test_label)
	#print(y_TestOneHot)

	print("\t[Info] ", end = '')
	print(x_Train_norm.shape, x_Test_norm.shape, y_TrainOneHot.shape, y_TestOneHot.shape)

	#print(y_TrainOneHot[:1])
	#----------------Training Key Code ----------------------########################
	print("[status] Training Setting")

	model = Sequential()  # Build Linear Model

	#---vgg16
	#VGG16paras(model)
	#MYparas(model)
	MYparas0217(model)
	#DNN(model)
	#TAparas(model)
	print("\t[Info] Batch size : %d" % batch_size)
	print("\t[Info] Epochs : %d" % epochs)


	print("\t[Info] Model summary:") 
	model.summary()
	

	#--------------------------Training-------------------------------------
	if trainingSwitch:
		print("[status] Training")
		#sgd = SGD(lr=0.0000001, decay=1e-6, momentum=0.1)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

		if augmen:
			print('\t[info] Augmentation is on')
			datagen = ImageDataGenerator(
			featurewise_center=False,
			featurewise_std_normalization=False,
			rotation_range=5,
			width_shift_range=0.1,
			height_shift_range=0.1,
			zoom_range=0.1,
			horizontal_flip=True)
			datagen.fit(x_Train_norm)
			try:
				train_history = model.fit_generator(datagen.flow(x_Train_norm, y_TrainOneHot, batch_size=batch_size),
				                steps_per_epoch=len(x_Train_norm) / 64, epochs=epochs,
				                validation_data = (x_Test_norm, y_TestOneHot), verbose=2)
			except KeyboardInterrupt:
				pass

		else :
			try:
				train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot,
				   validation_split=validate_portion, epochs=epochs, batch_size=batch_size, verbose=2) 
			except KeyboardInterrupt:
				pass


		scores = model.evaluate(x_Test_norm, y_TestOneHot)  
		print()  
		print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 

		#print("\t[Info] Making prediction to x_Test_norm")
		print("[status] Saving Model and Parameters")
		model.save('./model.h5')
		json_string = model.to_json()

		#print(train_history)
		#print("")
		#print(json_string)

		file = open("trainModel.json", 'w')
		file.write(json_string)
		file.close()
	'''
	#-----------------Load Trained Model------------------------------
	print("[status] Load Trained Model")
	file = open("./trainModel1632_638.json", 'r')
	json_string = file.read()
	file.close()

	print("[status] Load Trained Weight")
	model = model_from_json(json_string)
	model = load_model('./model1632_638.h5')

	#-----------------Reading Test Data-------------------------------
	print("[status] Reading Test Data")

	RawTestingData = pd.read_csv(sys.argv[1])
	#print(RawTestingData.shape)
	RawTestingData = RawTestingData.values

	#print(RawTestingData[0, 1])
	dataCount = RawTestingData.shape[0]
	X_testing_image = np.zeros((dataCount, 48, 48))
	#y_test_answer = np.zeros((dataCount,))

	for idx, row in enumerate(RawTestingData):
		x = row[1].split()
		x = np.asarray(x)
		x = np.reshape(x, (48, 48))
		X_testing_image[idx] = x

	x_Testing = X_testing_image.reshape(dataCount, 48, 48, 1).astype('float32')

	if normalization:
		x_Testing_norm = x_Testing/255
	else:
		x_Testing_norm = x_Testing

	#--------------------------Predicting--------------------------------
	print("[status] Predicting")

	#y_testing_answer = model.predict_classes(x_Testing_norm)
	val_proba = model.predict(x_Testing_norm)
	y_testing_answer = val_proba.argmax(axis=-1)

	#-------------------Output Predict Data--------------------------------
	print("[status] Output Predict Data")

	with open(sys.argv[2], 'w') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow(['id', 'label'])
	    for idx, val in enumerate(y_testing_answer):
	    	spamwriter.writerow([idx, val])
	#-------------------Plot training process-----------------------------
	
	'''
	plt.plot(train_history.history['acc'])
	plt.plot(train_history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	print("[status] Program Ending")
	'''
if __name__ == "__main__":
	main()
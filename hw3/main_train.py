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
epochs = 40
trainingSwitch = True
batch_size = 256
trainPath = sys.argv[1]

def main():
	#--------------------------------Reading--------------------------------
	print("[status] Reading Data")
	RawTrainingData = pd.read_csv(trainPath)
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

	#-----------------Load Trained Model------------------------------
	else:
		print("[status] Load Trained Model")
		file = open("trainModel.json", 'r')
		json_string = file.read()
		file.close()

		print("[status] Load Trained Weight")
		model = model_from_json(json_string)
		model = load_model('./model.h5')
'''
	#-----------------Reading Test Data-------------------------------
	print("[status] Reading Test Data")

	RawTestingData = pd.read_csv("./test.csv")
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

	with open('./output.csv', 'w') as csvfile:
	    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	    spamwriter.writerow(['id', 'label'])
	    for idx, val in enumerate(y_testing_answer):
	    	spamwriter.writerow([idx, val])
	#-------------------Plot training process-----------------------------
	'''

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

def DNN(model):
	model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(48, 48, 1), padding='same'))
	model.add(Dense(units=64, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dense(units=64, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dense(units=64, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dense(units=64, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dense(units=64, activation='relu'))

	model.add(Flatten())

	model.add(Dense(units=8, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.3))
	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.4))

	model.add(Dense(units=7))
	model.add(Activation('softmax'))


def MYparas0217(model):
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	#model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	#model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
	#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))	

	model.add(Flatten())

	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=2048, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=7))
	model.add(Activation('softmax'))


def MYparas0148(model): #26 epoch reach 60
	model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(48, 48, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Flatten())

	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(units=7))
	model.add(Activation('softmax'))

def MYparas(model):#paul
	model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(48, 48, 1), activation='relu'))
	model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
	model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	#model.add(Dropout(0.1))
	#model.add(Dense(units=256, input_dim=2304, kernel_initializer='normal', activation='relu'))
	#model.add(Dropout(0.3))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	#model.add(Dropout(0.1))
	#model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	#model.add(Dense(units=256, input_dim=2304, kernel_initializer='normal', activation='relu'))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	#model.add(Dropout(0.1))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	#model.add(Dropout(0.1))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
	###model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

	model.add(Flatten())

	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.3))

	model.add(Dense(units=512, activation='relu'))
	#model.add(Dropout(0.4))

	model.add(Dense(units=1024, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(units=1024, activation='relu'))
	#model.add(Dense(units=1024, activation='relu'))
	#model.add(Dense(units=1024, activation='relu'))
	#model.add(Dense(units=1024, activation='relu'))
	model.add(Dropout(0.5))

	#model.add(Dense(units=7, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  
	model.add(Dense(units=7))
	model.add(Activation('softmax'))


def TAparas(model):
	input_img = Input(shape=(48, 48, 1))
	block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
	block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

	block2 = Conv2D(64, (3, 3), activation='relu')(block1)
	block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

	block3 = Conv2D(64, (3, 3), activation='relu')(block2)
	block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
	block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

	block4 = Conv2D(128, (3, 3), activation='relu')(block3)
	block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

	block5 = Conv2D(128, (3, 3), activation='relu')(block4)
	block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
	block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
	block5 = Flatten()(block5)

	fc1 = Dense(1024, activation='relu')(block5)
	fc1 = Dropout(0.5)(fc1)

	fc2 = Dense(1024, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)

	model = Model(inputs=input_img, outputs=predict)

	# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


if __name__ == "__main__":
	main()
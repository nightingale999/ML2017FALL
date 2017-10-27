# Logistic Regression on Diabetes Dataset
from copy import deepcopy
from random import seed
from random import randrange
from csv import reader
import csv
from math import exp
import sys
import numpy as np
import pandas as pd

#mainLogic.py#
n_folds = 2#
l_rate = 0.1#
n_epoch = 10#
seed(1022)#
noramlization = 1#
rowsToDelete = []
lamb = 0#
#rowsToDelete = [1, 14, 52, 53, 54, 55, 56, 57, 58, 105]#


try:
	if sys.argv[1] == 'test':
		trainType = False
	else:
		trainType = True
	filenameX = sys.argv[2]
	filenameY = sys.argv[3]
	filenameXtest = sys.argv[4]
	fileSave = sys.argv[5]
	argumentsRoute = sys.argv[6]
	#if not trainType:
	#	final_coef
	print("Arguments success")

except:
	print("Arguments failed, use default")
	trainType = True
	filenameX = './X_train'
	filenameY = './Y_train'
	filenameXtest = './X_test'
	fileSave = './ans.csv'
	argumentsRoute = 'arguments.csv'

#---------------preprocess---------------------
print("preprocess is on")
data = pd.read_csv(filenameX)
data = data.values
data = np.insert(data, 0, np.zeros(106), 0)
#np.insert(data, 0, 87*np.zeros(data.shape[1]), 0)
data = np.delete(data, rowsToDelete, 1)
#print(data)
np.savetxt("./X_train_processed", data,fmt = '%u', delimiter = ',')

data = pd.read_csv(filenameXtest)
data = data.values
data = np.insert(data, 0, np.zeros(106), 0)
data = np.delete(data, rowsToDelete, 1)
np.savetxt("./X_test_processed", data, fmt = '%u', delimiter = ',')

filenameX = 'X_train_processed'
filenameXtest = 'X_test_processed'

final_coef = []

def nested_change(item, func):
    if isinstance(item, list):
        return [nested_change(x, func) for x in item]
    return func(item)

# Load a CSV file
def load_csv(filenameX, filenameY):
	dataset = list()
	with open(filenameX, 'r') as file:
		csv_reader = reader(file)
		for idx, row in enumerate(csv_reader):
			if idx == 0:
				continue
			if not row:
				continue
			dataset.append(row)
			#print(dataset)
			#exit()
	with open(filenameY, 'r') as file:
		csv_reader = reader(file)
		for idx, value in enumerate(csv_reader):
			if idx == 0:
				continue
			if not value:
				continue
			dataset[idx-1].append(value[0])
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
	#print(dataset)

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	#print(dataset, minmax)
	for row in dataset:
		for i in range(len(row)):
			#print(row[i], minmax[i][0], minmax[i][1], minmax[i][0])
			try:
				row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
			except ZeroDivisionError:
				row[i] = 1

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	print("evaluate is on")
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for idx, fold in enumerate(folds, 1):
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
		print("Fold no. %d Score: %.3f" %(idx, accuracy))
		print()
	return scores

# Make a prediction with coefficients
def predict(row, coefficients):
	#print(row, coefficients)
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	#print(yhat)
	#print(yhat, end = '')
	#print(1.0 / (1.0 + exp(-yhat)))
	try:
		return 1.0 / (1.0 + exp(-yhat))
	except OverflowError:
		return 0

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = (row[-1] - yhat) + lamb * sum(coef)  #------------------------------------------
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat) * row[0]
			#print(coef[0], l_rate, error, yhat, (1.0-yhat))
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print("epoch = %d, lrate = %.3f, error = %.3f" % (epoch, l_rate, sum_error))
	global final_coef 
	final_coef= list(coef)
	#print(final_coef)
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

#----------------------key code (training)--------------------------------

if trainType:
	print("train is on")
	dataset = load_csv(filenameX, filenameY)
	print(len(dataset[0]))
	for i in range(len(dataset[0])):
		str_column_to_float(dataset, i)
	# normalize
	minmax = dataset_minmax(dataset)
	if noramlization == True:
		normalize_dataset(dataset, minmax)
	scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
	with open("arguments.csv", 'w') as file:
		w = csv.writer(file)
		w.writerow(final_coef)
		#print(type(final_coef))
		#print(final_coef)

#----------------------testing-------------------------------------
if not trainType:
	print("test is on")
	with open(argumentsRoute, 'r') as file:
		w = reader(file)
		for row in w:
			for idx, val in enumerate(row):
				final_coef.append(float(val))
	#print(w)
	#print(final_coef)
testing_set = list()
with open(filenameXtest, 'r') as file:
	csv_reader = reader(file)
	for idx, row in enumerate(csv_reader):
		if idx == 0:
			continue
		if not row:
			continue
		#print(row)
		testing_set.append(row)
answer = list()
testing_set = nested_change(testing_set, float)
minmax = dataset_minmax(testing_set)
normalize_dataset(testing_set, minmax)
for row in testing_set:
	answer.append(round(predict(row, final_coef)))
ans_write = []
for idx, value in enumerate(answer, 1):
	ans_write.append(str(idx) + "," + str(answer))
with open(fileSave, 'w', newline='', encoding = "Big5") as csvfile:
	sw = csv.writer(csvfile, quoting=csv.QUOTE_NONE,escapechar='\\')
	sw.writerow(['id', 'label'])
	for idx, value in enumerate(answer, 1):
		sw.writerow([idx, value])

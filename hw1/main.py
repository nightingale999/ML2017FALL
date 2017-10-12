import sys
import csv
import math
import random
import numpy as np
#-----------user change section
hours = 9
dataCount = 1000
numIterations= 40000
alpha = 0.00005
ListOfFeatureNotUsed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
lamb = 0
#-----------

wordListOfFeatureNotUsed = []
feature = 18-len(ListOfFeatureNotUsed)
if 0 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("AMB_TEMP")
if 1 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("CH4")
if 2 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("CO")
if 3 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("NMHC")
if 4 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("NO")
if 5 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("NO2")
if 6 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("NOx")
if 7 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("O3")
if 8 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("PM10")
if 9 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("PM2.5")
if 10 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("RAINFALL")
if 11 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("RH")
if 12 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("SO2")
if 13 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("THC")
if 14 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("WD_HR")
if 15 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("WIND_DIREC")
if 16 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("WIND_SPEED")
if 17 in ListOfFeatureNotUsed:wordListOfFeatureNotUsed.append("WS_HR")

def gradientDescent(x, y, theta, alpha, m, numIterations):
	xTrans = x.transpose()
	for i in range(0, numIterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y + lamb * np.sum(theta ** 2)
		cost = np.sum(loss ** 2) / (2 * m)
		if (i % 1000) == 0:
			print("Iteration %d, Cost: %f" % (i, cost))
		gradient = np.dot(xTrans, loss) / m
		theta = theta - alpha * gradient
	print("about Theta : ")
	for idx, val in enumerate(theta):
		if idx % hours == 0:
			print("\n", int(idx/hours), " : ")
		print(val)
	print()
	return theta

def test(ipRoute, opRoute, argsRoute): # default : ./test.csv ./res.csv ./arguments.csv
	ans = []
	with open(argsRoute, 'r') as g:
		readerG = csv.reader(g)
		for row in readerG:
			theta = row
		'''
		for idx in range(hours * feature):
			if (idx % hours) == 0 :
				print("\nNum", end = '')
				print(int(idx/hours), end = ':\n')
			print(theta[idx])
		'''
		np.asarray(theta)


	with open(ipRoute, 'r') as f:
		print("%s is on" % ipRoute)
		reader = csv.reader(f)
		answer = 0

		featureCount = 0
		tempRow = np.zeros(shape = (feature*hours))
		#print("\nStart from here")
		for idx, row in enumerate(reader):
			
			if (row[1] not in wordListOfFeatureNotUsed):#####################################################################################
				for index, value in enumerate(row):
					if index > 10-hours:
						try:
							#print(featureCount, hours, index)
							#print(row[index])
							tempRow[featureCount * hours + index - 11 + hours] = row[index]
							#print(featureCount * hours + index - 2)
						except ValueError:
							tempRow[featureCount * hours + index - 11 + hours] = -1

				featureCount = featureCount + 1

			if row[1] == 'WS_HR':
				featureCount = 0
				#print(len(tempRow), len(theta))
				#print(tempRow)
				for index, value in enumerate(tempRow):
					answer = answer + float(tempRow[index]) * float(theta[index])
				ans.append(answer)
				answer = 0
			#print(tempRow)
			#print(featureCount)
		print(np.shape(ans))

	with open(opRoute, 'w', newline='', encoding = "Big5") as csvfile:
		sw = csv.writer(csvfile, quoting=csv.QUOTE_NONE,escapechar='\\')
		#sw.writerow(['Spam'] * 5 + ['Baked Beans'])
		#sw.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		sw.writerow(['id', 'value'])
		for x in range(240):
			temp = "id_" + str(x)
			sw.writerow([temp, ans[x]])



def train(ipRoute, opRoute): # default : ./train.csv ./arguments.csv
	with open(ipRoute, 'r', encoding = "Big5") as f:
		data = np.zeros(shape=(12, 18, 480), dtype=np.float)
		reader = csv.reader(f)

		#-------------read data and put Data in DATA---------------------
		for line, row in enumerate(reader, 1):
			month = int((line-2)/360)
			date = (line-2) % 360 // 18
			featureType = (line-2) % 18
			#print(line, month, date, featureType)
			for idx, x in enumerate(row):
				if idx > 2 and featureType not in ListOfFeatureNotUsed:####################################################################################################
					try:
						data[month][featureType][24*date+idx-3] = row[idx]
						#print(month, featureType, 24*date+idx-3, row[idx], data[month][featureType][24*date+idx-3])
					except ValueError:
						data[month][featureType][24*date+idx-3] = 0
		'''
		try:
			for i in range(12):
				for j in range(18):
					for k in range(480):
						#print(i, j, k, data[i][j][k],",")
						#print("", end = '')
					#print()
		except KeyboardInterrupt or Exception:
			pass
		'''
		#------------------from DATA randomly pick data to train---------------
		x = np.zeros(shape=(dataCount , hours*feature))
		y = np.zeros(shape=dataCount)

		for line in range(dataCount):
			randMonth = random.randint(0, 11)
			randDate = random.randint(0, 19)
			randHour = random.randint(0,470)
			featureCount = 0
			#print(randMonth, randDate, randHour)
			for fea in range(18):
				if fea in ListOfFeatureNotUsed:#############################################################################################
					#print("i jump")
					continue
				for hoursIter in range(hours):
					#print(line, index)
					x[line][featureCount * hours + hoursIter] = data[randMonth][fea][randHour+hoursIter]
					#print("x DATA", end = '')
					#print(x[line][featureCount * hours + hoursIter], data[randMonth][fea][randHour+hoursIter])

				#print(random.randint(0, 11))
				featureCount = featureCount + 1
				#print("featureCount : ", featureCount)
			y[line] = data[randMonth][9][randHour + hours]
	'''
	for r in range(dataCount):
		for c in range(hours*feature):
			if c % hours == 0:
				print()
			print(x[r][c],",", end ='')
		print("\n\n")

	print(y)
	exit()
	'''
	m, n = np.shape(x)
	theta = np.ones(n)
	theta = gradientDescent(x, y, theta, alpha, m, numIterations)

	print('m = %d'% m)
	print('n = %d'% n)
	print('theta = ', end='')
	#print(theta)

	with open(opRoute, 'w', newline='', encoding = "Big5") as csvfile:
		sw = csv.writer(csvfile, quoting=csv.QUOTE_NONE,escapechar='\\')
		#sw.writerow(['Spam'] * 5 + ['Baked Beans'])
		#sw.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		sw.writerow(theta)

	#psudoTest(ipRoute, opRoute)
	test("./test.csv", "./res.csv", "./arguments.csv")



def main():
	try:
		
		ipRoute = str(sys.argv[2])
		opRoute = str(sys.argv[3])
		print(ipRoute)
		print(opRoute)
		print("Argument Certified")
	except:
		print("wrong arg detected, use default")
		if sys.argv[1] == 'test' :
			ipRoute = "./test.csv"
			opRoute = "./res.csv"
			argsRoute  = "./arguments.csv"
		elif sys.argv[1] == 'train' :
			ipRoute = "./train.csv"
			opRoute = "./arguments.csv"
		
	print("Input Route:", end = '')
	print(ipRoute)
	print("Output Route:", end = '')
	print(opRoute)
		

	if sys.argv[1] == 'test' :
		print("test is on")
		test(ipRoute, opRoute, 'arguments.csv')

	elif sys.argv[1] == 'train' :
		print("train is on")
		train(ipRoute, opRoute)

	print("Hello World!!")

if __name__ == "__main__":
	main()
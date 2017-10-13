import sys
import csv
import numpy as np

def gradientDescent(x, y, theta, alpha, m, numIterations):
	xTrans = x.transpose()
	for i in range(0, numIterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		# avg cost per example (the 2 in 2*m doesn't really matter here.
		# But to be consistent with the gradient, I include it)
		cost = np.sum(loss ** 2) / (2 * m)
		if (i % 1000) == 0:
			print("Iteration %d | Cost: %f" % (i, cost))
		# avg gradient per example
		gradient = np.dot(xTrans, loss) / m
		# update
		theta = theta - alpha * gradient
	print(theta)
	return theta

def test(ipRoute, opRoute, argsRoute): # default : ./test.csv ./res.csv ./arguments_best.csv
	ans = []
	with open(argsRoute, 'r') as g:
		readerG = csv.reader(g)
		for row in readerG:
			theta = row
		print(theta)

	with open(ipRoute, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[1] == 'PM2.5':
				answer = 0
				for idx, val in enumerate(row):
					if idx > 1 :
						answer = float(val) * float(theta[idx-2]) + answer

				ans.append(int(answer))
				#print(row[0]," pm2.5 is", row[10])

	with open(opRoute, 'w', newline='') as csvfile:
		sw = csv.writer(csvfile, quoting=csv.QUOTE_NONE,escapechar='\\')
		#sw.writerow(['Spam'] * 5 + ['Baked Beans'])
		#sw.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		sw.writerow(['id', 'value'])
		for x in range(240):
			temp = "id_" + str(x)
			sw.writerow([temp, ans[x]])



def train(ipRoute, opRoute): # default : ./train.csv ./arguments.csv
	with open(ipRoute, 'r', encoding = "Big5") as f:
		ans = []
		data = []
		reader = csv.reader(f)
		for row in reader:
			data.append(row)
			if row[2] == 'PM2.5':
				ans.append(row)
				#print(row)
		#print("length of data = ")
		#print(len(data))

		x = np.zeros(shape=(len(ans) , 9))
		y = np.zeros(shape=len(ans))

		localIter = 0
		for val in data:
			print(val)
			if val[2] == 'PM2.5':
				for idx, val in enumerate(val):
					if idx > 2 and idx < 12:
						print(localIter, idx-3, val)
						print(val)
						x[localIter][idx-3] = val
					if idx == 12:
						y[localIter] = val
				localIter = localIter + 1
		'''
		for x in x:
			print(x)
		for x in y:
			print(x)
		'''

	m, n = np.shape(x)
	numIterations= 40000
	alpha = 0.00005
	theta = np.ones(n)

	print(theta)
	theta = gradientDescent(x, y, theta, alpha, m, numIterations)

	print('m = ', end='')
	print(m)
	print('n = ', end='')
	print(n)
	print('theta = ', end='')
	print(theta)

	with open(opRoute, 'w', newline='', encoding = "Big5") as csvfile:
		sw = csv.writer(csvfile, quoting=csv.QUOTE_NONE,escapechar='\\')
		#sw.writerow(['Spam'] * 5 + ['Baked Beans'])
		#sw.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		sw.writerow(theta)

	test("./test.csv", "./res.csv", "./arguments.csv")



def main():
	try:
		print("Py Arg 0:")
		print(sys.argv[0])
		print("Py Arg 1:")
		print(sys.argv[2])
		print("Py Arg 2:")
		print(sys.argv[3])
		print("Argument Certified")
		ipRoute = str(sys.argv[2])
		opRoute = str(sys.argv[3])
	except:
		print("wrong arg detected, use default")
		if sys.argv[1] == 'test' :
			ipRoute = "./test.csv"
			opRoute = "./res.csv"
			argsRoute  = "./arguments_best.csv"
		elif sys.argv[1] == 'train' :
			ipRoute = "./train.csv"
			opRoute = "./arguments.csv"
		
		

	if sys.argv[1] == 'test' :
		print("test is on")
		test(ipRoute, opRoute, "./arguments.csv")

	elif sys.argv[1] == 'train' :
		print("train is on")
		train(ipRoute, opRoute)

	print("Hello World!!")

if __name__ == "__main__":
	main()
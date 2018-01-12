import numpy as np
import pandas as pd
import skimage
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn import cluster, metrics
from sklearn import svm
from keras.models import Sequential, load_model, Model, model_from_json
import csv
import sys
import datetime
from sklearn.manifold import TSNE
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adadelta


nComponents = 32
loadModelPath = './model_01102204_1821.h5'
imagePath = sys.argv[1]
testDataPath = sys.argv[2]
saveFilePath = sys.argv[3]


#-------------------------Reading Training Data-----------------------
print("[Status] Reading Training Data")
rawData = np.load(imagePath)
#print(rawData)
'''
df = pd.read_csv(
	filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
	header=None,
	sep=',')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end
df.tail()
#print(df)
'''
#X = df.ix[:,0:4].values
#y = df.ix[:,4].values

X = rawData
X = X.astype(float)
X = X / 256

#X = np.reshape(X, [len(X), 28, 28])

'''
#-----------------------------------SVD---------------------------------------------
print("[Status] SVD")
Y_2sklearn = np.zeros((len(X), 28, 28))
for idx, val in enumerate(X):
	U, s, V = np.linalg.svd(X[idx], full_matrices=True)
	S = np.diag(s)
	Y_2sklearn[idx] = np.dot(U, S)

X_lessDim = np.zeros((len(X), 28*28))
for idx, val in enumerate(Y_2sklearn):
	X_lessDim[idx] = np.reshape(Y_2sklearn[idx], [28*28])
'''
'''
#-----------------------------------truncatedSVD---------------------------------------------
print("[Status] truncatedSVD")

svd = TruncatedSVD(n_components=nComponents, n_iter=20, random_state=42)

#X_lessDim = np.zeros((len(X), nComponents))

X_lessDim = svd.fit_transform(X)
'''

'''
X_lessDim = np.zeros((len(X), 28*nComponents))
for idx, val in enumerate(Y_2sklearn):
	X_lessDim[idx] = np.reshape(Y_2sklearn[idx], [28*nComponents])
'''
#-----------------------------------DNN---------------------------------------------
print("[Status] DNN")
'''
model = Sequential()
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
'''
'''
encoding_dim = nComponents
input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
#encoded = Dense(128, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
#decoded = Dense(64, activation='relu')(encoded)
#decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)




# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


history_callback = autoencoder.fit(X, X, epochs=100, batch_size=256, shuffle=True)#, validation_data = (X, X))
loss_history = history_callback.history["loss"]

fitLoss = loss_history[-1]*10000

encoder = Model(input = input_img, output = encoded)
encoded_input = Input(shape = (784, ))
'''
encoder = load_model(loadModelPath)

X_lessDim = encoder.predict(X)
X_lessDim[0:3]



'''
#-----------------------------------PCA---------------------------------------------
print("[Status] PCA")

X_lessDim = np.zeros((len(X), nComponents))
sklearn_pca = sklearnPCA(n_components=nComponents)

X_lessDim = sklearn_pca.fit_transform(X)

PCAscore = np.sum(sklearn_pca.explained_variance_ratio_)#sthash.txY3TTeg.dpuf
print("\t[Info] PCA Score: %06f"%PCAscore)

#X_lessDim = sklearn_pca.fit_transform(X)
#print(X)
#print(X)
#print(X_lessDim)
#print(type(X_lessDim))
#print(X_lessDim.shape)
'''

'''
#-----------------------------------RandomizedPCA---------------------------------------------
print("[Status] RandomizedPCA")
X_lessDim = RandomizedPCA(64).fit(X)
'''


'''
#-----------------------------------SVM---------------------------------------------
print("[Status] SVM")
#nu = outliers.shape[0] / target.shape[0]
#print('nu', nu)

model = svm.OneClassSVM(nu=0.5, kernel='rbf',gamma=0.00005, verbose = True, cache_size=65536)

model.fit(X_lessDim)
#exit()
ans = model.predict(X_lessDim)
print(ans)
'''
'''
#-----------------------------------t-SNE---------------------------------------------
print("[Status] t-SNE")
tsne = TSNE(n_components=2, verbose=2, perplexity=50)#, n_iter=250)
X_lessDim_tsne = tsne.fit_transform(X_lessDim)
'''


#----------------------------------K-Means---------------------------------------------
print("[Status] K-Means")

#X_lessDim = X_lessDim / 256

#clf = cluster.KMeans(init='k-means++', n_clusters=2, random_state=1)
#clf.fit(X_lessDim)
#clusters = clf.fit_predict(X_lessDim)

clusters = cluster.KMeans(n_clusters=2, random_state=1).fit(X_lessDim)
clusters = clusters.labels_

score = metrics.silhouette_score(X_lessDim, clusters, sample_size=1024)
print("\t[Info] Clustering Score: ", score)
summary = [0, 0, 0]
for val in clusters:
	if val == 0:
		summary[0] += 1
	elif val == 1:
		summary[1] += 1
	else:
		summary[2] += 1

print("\t[Info] numbers of each dataset: ", summary)


'''
#-----------------------------------Writing Log---------------------------------------------
print("[Status] Writing Log")

now = datetime.datetime.now()
file = open("Log.txt", "a")
file.write("\n\n################################################################")
file.write("\n%s\nnComp:%s\tA%d\tB%d\tABdif:%d\t\nfitLoss:%d\tClusterScore:%.4f"%(str(now),nComponents, summary[0], summary[1], abs(summary[1]-summary[0]),fitLoss, score))

file.close()
encoder.save('./models/model_%02d%02d%02d%02d_%d.h5'%(now.month, now.day, now.hour, now.minute, fitLoss))
'''

#-----------------------------------Reading Testing Data---------------------------------------------
print("[Status] Reading Testing Data")
testData = []
reader = csv.reader(open(testDataPath, "r"))
for idx, row in enumerate(reader):
	if idx > 0:
		testData.append(row)

#-----------------------------------Predicting---------------------------------------------
print("[Status] Predicting")

print("\t[Info] By Pre-Clustered Information")
ans = []
for idx, row in enumerate(testData):
	if (clusters[int(row[1])] == clusters[int(row[2])]):
		ans.append([idx,1])
	else:
		ans.append([idx,0])



'''
print("\t[Info] By similarity(MSE)")
summ = 0
ans = []
imgDiff = []
for idx, row in enumerate(testData):
	diffArr = X_lessDim[int(row[1])]-X_lessDim[int(row[2])]
	diff = np.sum(np.square(diffArr))
	summ += diff
	imgDiff.append(diff)
avg = summ/len(testData)
for idx, row in enumerate(testData):
	if imgDiff[idx] < avg:
		ans.append([idx,1])
	else:
		ans.append([idx,0])
'''

#-----------------------------------Writing Answer---------------------------------------------
print("[Status] Writing Answer")
with open(saveFilePath, 'w') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	nothing = spamwriter.writerow(['ID', 'Ans'])
	for idx, val in enumerate(ans):
		nothing = spamwriter.writerow(val)


import pandas as pd
import numpy as np
import csv
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Reshape
from keras.layers import Embedding, Dot, Add
from keras.models import Sequential, load_model, Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys

normalizeFlag = True
noBiasFlag = False

epoch = 100
latent_dim = 50
batchSize = 1024
test = pd.read_csv(sys.argv[1])
saveAnsPath = sys.argv[2]
loadModelPath = sys.argv[3]


'''
train = pd.read_csv('./train.csv')
train = train.sample(frac=1).reset_index(drop=True)
'''
'''
matrix = pd.concat([train,test]).pivot('UserID','MovieID','Rating')

if normalizeFlag:
	print(train['Rating'].std())
	print(train['Rating'].mean())
	print(train['Rating'].describe())
	std = train['Rating'].std()
	mean = train['Rating'].mean()
	train['Rating'] = (train['Rating'] - mean) / std
	#test['Rating'] = (test['Rating'] - mean) / std

#print(train)

#print(matrix)
print(type(matrix))
#print(matrix.index)
#print(matrix.columns)
print(type(matrix.index))
print(type(matrix.columns))
'''
'''
n_users = matrix['UserID'].values.tolist()
print(n_users)
exit()

n_users = len(matrix.index)
n_items = len(matrix.columns)
'''
#model = Sequential()
'''
user_input = Input(shape=[1])
item_input = Input(shape=[1])
user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
user_vec = Flatten()(user_vec)
item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
item_vec = Flatten()(item_vec)
user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
user_bias = Flatten()(user_bias)
item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
item_bias = Flatten()(item_bias)
r_hat = Dot(axes=1)([user_vec, item_vec])
'''
#print(r_hat)
#print(user_bias)
#print(item_bias)
#print([r_hat, user_bias, item_bias])
'''
if not noBiasFlag:
	r_hat = Add()([r_hat, user_bias, item_bias])


model = Model([user_input, item_input], r_hat)
model.compile(loss='mse', optimizer='Adam')
model.summary()
print(latent_dim)


callbacks = [EarlyStopping(monitor='val_loss', patience=1)]
history_callback = model.fit(x=[train['UserID'], train['MovieID']],
 y=train['Rating'], 
 epochs=epoch, 
 batch_size=batchSize,
 validation_split = 0.1,
 verbose=2,
 callbacks=callbacks
 )
'''
model = load_model(loadModelPath)
model.summary()

val_proba = model.predict(x=[test['UserID'], test['MovieID']])

std =  1.116898
mean = 3.581712

y_testing_answer = []
if normalizeFlag:
	for val in val_proba:
		y_testing_answer.append(round(min(max(val[0]*std+mean, 1), 5), 1))
	try:
		print("\n\nEstimated Loss: %.4f\n"%(loss*std))
	except:
		pass
else:
	for val in val_proba:
		y_testing_answer.append(round(min(max(val[0], 1), 5), 1))



with open(saveAnsPath, 'w') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	spamwriter.writerow(['TestDataID', 'Rating'])
	for idx, val in enumerate(y_testing_answer):
		spamwriter.writerow([idx+1, val])

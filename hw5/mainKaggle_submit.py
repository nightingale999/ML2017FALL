import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import random
import pickle
import sys

trainingFlag = False
test = pd.read_csv(sys.argv[1])

#print(test['TrainDataID'])
if trainingFlag == True:
    train = pd.read_csv('./train.csv')
    matrix = pd.concat([train,test]).pivot('UserID','MovieID','Rating')
    movie_means = matrix.mean()
    user_means = matrix.mean(axis=1)
    mzm = matrix-movie_means
    mz = mzm.fillna(0)
    mask = -mzm.isnull()
    iteration = 0
    mse_last = 999

    while iteration<40:
        iteration += 1
        svd = TruncatedSVD(n_components=15,random_state=16)
        svd.fit(mz)
        mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

        mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
        print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
        mzsvd[mask] = mzm[mask]

        mz = mzsvd
        if mse_last-mse<0.00001: break
        mse_last = mse

    m = mz+movie_means
    m = m.clip(lower=1,upper=5)
    #print(m)

    with open('best.pickle', 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('best.pickle', 'rb') as handle:
        m = pickle.load(handle)
test['Rating'] = round(test.apply(lambda x:m[m.index==x.UserID][x.MovieID].values[0],axis=1), 1)
# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(test.Rating.isnull())[0]
if trainingFlag:
    test.ix[missing,'Rating'] = user_means[test.loc[missing].UserID].values
else:
    test.ix[missing,'Rating'] = 3.8
test.to_csv(sys.argv[2],index=False,columns=['TestDataID','Rating'])
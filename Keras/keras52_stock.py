import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, LSTM,  Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# 1
path = './_data/sm/'

samsung = pd.read_csv(path + '삼성전자 주가.csv' , header=0, index_col=None, encoding='cp949')
# print(samsung)
# print(samsung.shape)  #(1980, 17)

amore = pd.read_csv(path + '아모레퍼시픽 주가.csv', header=0, index_col=None, encoding='cp949')
# print(amore)
# print(amore.shape)  (2220, 17)

#삼성전자 x, y
samsung_x = samsung[['고가', '저가', '금액(백만)', '종가', '등락률']]
samsung_y = samsung[['시가']].values #numpy 배열로 전환 시킴 split 거쳐 변환
# print(samsung_x)
# # print(samsung_y)
# print(samsung_x.shape) #(1980, 5)
# print(samsung_y.shape)  #(1980, 1)

#아모레 x, y        삼성전자에 맞춰서 아모레를 행을 맞춰줌
amore_x = amore.loc[0:1979,['고가', '저가', '금액(백만)', '종가', '등락률']]
# print(amore_x)
# print(amore_x.shape) #(1980, 5)


samsung_x = MinMaxScaler().fit_transform(samsung_x)
amore_x = MinMaxScaler().fit_transform(amore_x)

def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)

samsung_x = split_data(samsung_x, 5)
amore_x = split_data(amore_x, 5)
# print(samsung_x.shape) #(1976, 5, 5)
# print(amore_x.shape) #(1976, 5, 5)

samsung_y = samsung_y[4:, :]

amsung_x_predict = samsung_x[-1].reshape(-1, 5, 5)
amore_x_predict = amore_x[-1].reshape(-1, 5, 5)


samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test  = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.7, random_state=123)

print(samsung_x_train.shape, samsung_x_test.shape)  # (1383, 5, 5) (593, 5, 5)
print(samsung_y_train.shape, samsung_y_test.shape) # (1383, 1) (593, 1)
print(amore_x_train.shape, amore_x_test.shape)


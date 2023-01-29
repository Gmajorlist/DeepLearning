

from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

path = './_data/sm/'

samsung = pd.read_csv(path + '삼성전자 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(samsung)
# print(samsung.shape) #(1980, 17)

amore = pd.read_csv(path + '아모레퍼시픽 주가.csv', header=0, index_col=None, sep=',', encoding='cp949', thousands=',').loc[::-1]
# print(amore)
# print(amore.shape)   #(2220, 17)

# 삼성전자 x ,y 
samsung_x = samsung[['고가', '저가','종가', '외인(수량)', '기관']]
samsung_y = samsung[['시가']].values
# print(samsung_y)
# print(samsung_x.shape) # (1980, 5)
# print(samsung_y.shape) # (1980, 1)

# 아모레 x, y 
amore_x = amore.loc[1979:0,['고가', '저가', '종가', '외인(수량)', '시가']]
# print(amore_x)
# print(amore_x.shape) #(1980, 5)

samsung_x = MinMaxScaler().fit_transform(samsung_x)
amore_x = MinMaxScaler().fit_transform(amore_x)

def split_data(dataset, timesteps):
    tp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tp.append(subset)
    return np.array(tp)

samsung_x = split_data(samsung_x, 5)
amore_x = split_data(amore_x, 5)
# print(samsung_x.shape) #(1976, 5, 5)
# print(amore_x.shape) #(1976, 5, 5)

samsung_y = samsung_y[4:, :] 
# print(samsung_y.shape) #(1976, 1)

# 예측에 사용할 마지막 값을 데이터 추출
samsung_x_predict = samsung_x[-1].reshape(-1, 5, 5)
amore_x_predict = amore_x[-1].reshape(-1, 5, 5)
# print(samsung_x_predict.shape) # (5, 5, 1)
# print(amore_x_predict.shape) # (5, 5, 1)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test  = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.7, random_state=123)

print(samsung_x_train.shape, samsung_x_test.shape)  # (1383, 5, 5) (593, 5, 5)
print(samsung_y_train.shape, samsung_y_test.shape) # (1383, 1) (593, 1)
print(amore_x_train.shape, amore_x_test.shape) #(1383, 5, 5) (593, 5, 5)


#samsung
samsung_input = Input(shape=(5,5))
dense11 = LSTM(128, return_sequences=True, activation='relu')(samsung_input)
dense11 = Dropout(0.2)(dense11)
dense22 = LSTM(256, activation='relu')(dense11)
dense33 = Dense(524, activation='relu')(dense22)
dense44 = Dense(128, activation='relu')(dense33)
dense55 = Dropout(0.2)(dense44)
dense66 = Dense(82, activation='relu')(dense55)
dense77 = Dense(44, activation='relu')(dense66)
dense88 = Dense(22, activation='relu')(dense77)
samsung_output = Dense(1,  activation='relu')(dense88)

#amore
amore_input = Input(shape=(5,5))
dense1 = LSTM(128, return_sequences=True, activation='relu')(samsung_input)
dense1 = Dropout(0.2)(dense1)
dense2 = LSTM(256, activation='relu')(dense1)
dense3 = Dense(524, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dropout(0.2)(dense4)
dense6 = Dense(82, activation='relu')(dense5)
dense7 = Dense(44, activation='relu')(dense6)
dense8 = Dense(22, activation='relu')(dense7)
amore_output = Dense(1, activation='relu')(dense8)

#merge
merge1 = concatenate([samsung_output, amore_output])
merge2 = Dense(256, activation='relu')(merge1)
merge3 = Dense(128, activation='relu')(merge2)
merge4 = Dense(64, activation='relu')(merge3)
merge_output = Dense(1, activation='relu')(merge4)


#3
filepath = './keras_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
model = Model(inputs=[samsung_input, amore_input], outputs=[merge_output])
model.summary
model.compile(loss='mse', optimizer='adam')
ES = EarlyStopping(monitor='val_loss', mode='auto', patience=20, 
                   restore_best_weights=True)
MCP = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                    filepath = filepath  + '_'    )
model.save_weights(path + 'samsungstock_weight.h5')

model.fit([samsung_x_train, amore_x_train], samsung_y_train , epochs=102, 
          batch_size=555, validation_batch_size=0.2, callbacks=[ES, MCP])

#4
loss=model.evaluate([samsung_x_test, amore_x_test], samsung_y_test, batch_size=100)
samsung_y_predict=model.predict([samsung_x_predict, amore_x_predict])

print("loss : ", loss)
print("삼성전자 시가 :" , samsung_y_predict)

# loss :  4798170112.0
# 삼성전자 시가 : [[72358.34]]


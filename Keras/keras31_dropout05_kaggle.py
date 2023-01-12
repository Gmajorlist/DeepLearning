from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,Dropout
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SS
from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint 
import numpy as np
import pandas as pd
 

#데이터
path = './_data/bike/'
train_data = pd.read_csv(path + 'train.csv', index_col=0)
test_data= pd.read_csv(path + 'test.csv', index_col=0)
samplesubmission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

train_data = train_data.drop(['casual', 'registered'], axis = 1)
x = train_data.drop(['count'], axis=1)                              # y 값(count 열) 분리, axis = 1 → 열에 대해 동작
y = train_data['count']

scaler = SS()
x_train = scaler.fit_transform(train_data)
x_test = scaler.transform(test_data)


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=13))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

# #2. 모델구성 
# input = Input(shape=(13,))
# dense1 = Dense(32)(input)
# dense2 = Dense(64, activation= 'relu')(dense1)
# drop1 = Dropout(0.5)(dense2)
# dense3 = Dense(256, activation= 'relu')(drop1)
# drop2 = Dropout(0.3)(dense3)
# dense4 = Dense(128, activation= 'relu')(drop2)
# drop3 = Dropout(0.2)(dense4)
# output = Dense(1)(drop3)
# model = Model(inputs=input, outputs=output)


model.compile(loss = 'mse', optimizer='adam')

ES = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                   verbose = 1, restore_best_weights=False) 
# restore_best_weights=False: default → EarlyStopping된 지점에서부터 patience만큼(가중치가 가장 좋은 지점 X)

import datetime
date = datetime.datetime.now() #현재 시간이 나옴
print(date)
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") 
print(date)  #0112_1503
print(type(date)) 

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
# 04d 정수로 네 자리 받아드리겠다 #4f 소수 넷째 자리까지

MCP = ModelCheckpoint(monitor='val_loss', mode = 'auto',
                      save_best_only=True, verbose = 1, 
                    #   filepath = path + 'keras30_ModelCheckPoint1.hdf5')
                    filepath = filepath + 'k31_01 ' + date+ '_' + filename) 
# ModelCheckpoint: 모델과 가중치 저장, save_best_only=True: 가장 좋은 가중치 저장
model.fit(x_train, y_train, epochs=1024, batch_size=16, validation_split=0.2, 
          callbacks=[ES, MCP]) 



#4. 평가 및 예측
print('========================= 1. 기본 출력 ===========================')
loss = model.evaluate(x_test, y_test, verbose=3)
print('loss: ', loss)

y_predict = model.predict(x_test, verbose=3)
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)


y_submit = model.predict(scaler.transform(test_data))
submission['count'] = y_submit
samplemission.to_csv(path + 'submission_1111.csv')

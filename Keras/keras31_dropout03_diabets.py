from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SS
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

#데이터
path = 'C:/study/keras_save/MCP/'
dataset = load_diabetes()        
x = dataset.data                
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=123
)
# scaler = MMS()
scaler = SS()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

#3. 컴파일 및 훈련
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


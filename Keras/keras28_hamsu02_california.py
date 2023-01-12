import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y
    , test_size=0.2, random_state=29 )
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

# #모델구성
# model = Sequential()
# model.add(Dense(1, activation='relu' input_dim=8)) 
# model.add(Dense(2, activation='relu' input_shape=(8, ))) 
# model.add(Dense(7, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='relu'))

input1 = Input(shape=(8, ))
dense1 = Dense(1, activation='relu')(input1)
dense2 = Dense(2, activation='relu')(dense1)
dense3 = Dense(7, activation='relu' )(dense2)
dense4 = Dense(4, activation='relu')(dense3)
dense5 = Dense(4, activation='relu')(dense4)
output1 = Dense(1, activation='relu')(dense5)
model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #대문자 - 파이썬에 클래스으로 구성
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5, # 인내심 - 몇번참는다 
                              restore_best_weights=True, #최소의 loss일 때 데이터를 가져옴
                              verbose=1,
                               )
hist = model.fit(x_train, y_train, epochs=50, batch_size=50,
          validation_split=0.2, verbose=1
          )
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict(x_test)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red'
#          , marker = '.', label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue'
#          , marker = '.', label = 'val_loss')
# plt.grid() 
# plt.xlabel('epoch')   
# plt.ylabel('loss')
# plt.title('boston loss')
# plt.legend()
# plt.show()





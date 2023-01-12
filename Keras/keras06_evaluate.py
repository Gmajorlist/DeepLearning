import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(31, input_dim=1)) #Dense는 output
model.add(Dense(52))
model.add(Dense(41))
model.add(Dense(27))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=3) 

#4.평가, 예측
loss = model.evaluate(x, y) #loss값은 기준이다 # 훈련데이터x
print('loss : ', loss)
result = model.predict([6])
print('6의 결과:', result)


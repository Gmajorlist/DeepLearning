import tensorflow as tf
import numpy as np

print(tf.__version__)

#1. 데이터
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델구성
from tensorflow.keras.models import Sequential # .은 그 안에 뭐 . 그 안에 뭐
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))   #input =x 1= y

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=100) #epoch=훈련을 여러번 시키는것 /  fit= 훈련 시킴 / epoch 너무 많이하면 성능이 안좋아짐 / epoch수치를 찾아내야함

#4. 평가, 예측
result = model.predict([4])
print('결과 : ', result)
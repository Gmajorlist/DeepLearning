import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10)]) #(10, ) (10, 1) input_dim  에서 동일하게 적용
print(x.shape) 
y = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])   # x에 9 를 훈련 시켜서 y 끝에 숫자가 가깝게 나와야함
print(y.shape)

x = x.T #(10, 1)
y = y.T #(10, 3)
print(x.shape)
print(y.shape)

#2. layer
model = Sequential()
model.add(Dense(200, input_dim=1)) # x열 수
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(29))
model.add(Dense(3)) # y 열 수 

#3. compile
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=220, batch_size=2)

#4. predict
loss = model.evaluate(x, y)
print('loss:', loss)

result = model.predict([[9]])#x 의 끝 수
print('[9]의 예측값:', result)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([range(10), range(21, 31), range(201, 211)]) # 0부터 10 - 1 (9)까지
print(x.shape)  # (3, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
print(y.shape) # (2, 10)

x = x.T
print(x.shape) #(10, 3)
y = y.T
print(y.shape) #(10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=3))
model.add(Dense(50))
model.add(Dense(600))
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(2)) # y가 열이 2개

# 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=700, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 :', result)
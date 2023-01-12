import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))
x_test = np.array([18,19,20])
y_test = np.array([18,19,20])
x_val = np.array([21,22,23])
y_val = np.array([21,22,23])

#train_test_split로 자르고 10:3:3 나누기

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train,
    y_train, test_size=0.15, random_state=12)
x_train, x_val, y_train, y_val = train_test_split(x_train, 
    y_train, test_size=0.2, random_state=11)

print(x_train)#[14 13 16  8 12  2 11 15  7 10  3]
print(y_train)#[14 13 16  8 12  2 11 15  7 10  3]
print(x_test) #[ 6  9 10]
print(y_test) #[ 6  9 10]
print(x_val)  #[11 13  3]
print(y_val)  #[11 13  3]


# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_vaildation = np.array([14,15,16])
# y_vaildation = np.array([14,15,16])

#2.모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([17])
print("17의 예측값 :", result)


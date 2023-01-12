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

#train_test_split를 이용하여 7:3으로 잘라 모델구현 소스 완성
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=12
)
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)
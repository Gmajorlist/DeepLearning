import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
#데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
arr = np.array([1,2,3,4,5,6,7])
# 노드의 개수 7 
#(array([1, 2, 3, 4, 5, 6, 7]), 
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510],  dtype=int64))
# print(y)


######################1.케라스 투카테고리컬 #####################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape) #(581012, 8)
# print(type(y)) #<class 'numpy.ndarray'> #확인하고
# print(np.unique(y[: ,0], return_counts= True)) # :모든행의 0번째 표현
# y = np.delete(y, 0, axis=1)

######################2.판다스 겟더미스 #####################
# import pandas as pd   #판다스는 헤더와 인덱스가 자동으로 생성                             
# y = pd.get_dummies(y)
# # y = y.values         # y 가 numpy로 바뀐다
# y = y.to_numpy() # 위에랑 두가지 방법이 있다.
# print(type(y)) # 바뀐지확인

######################3.사이킷런 원핫인코더 #####################
from sklearn.preprocessing import OneHotEncoder        
ohe = OneHotEncoder()
print(y.shape)
y = y.reshape(581012, 1)
print(y.shape)

# ohe.fit(y.reshape(-1, 1))
#쉐이프를 맞추는 작업
# y = ohe.transform(y.reshape(-1, 1)).toarray()

# ohe.fit(y) 
# y = ohe.transform(y)
y = ohe.fit_transform(y)  # 위에 두줄을 한줄로 줄임
y = y.toarray()




x_train, x_test, y_train, y_test = train_test_split(x,y,
    shuffle=True,
    random_state=123,
    test_size=0.9,
    stratify=y)

#모델
model = Sequential()
model.add(Dense(86, activation='relu', input_shape=(54, )))
model.add(Dense(76, activation='sigmoid'))
model.add(Dense(56, activation='relu'))
model.add(Dense(66, activation='relu'))
model.add(Dense(76, activation='relu'))
model.add(Dense(86, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='softmax'))

#컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earltStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=50,
          validation_split=0.2,
          verbose=1,
          callbacks=[earltStopping])
#평가예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
print(y_test[:2])
y_predict = model.predict(x_test[:2])
print(y_predict)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)  #2-tf.argmax
print("y_pred:", y_predict)
y_test = np.argmax(y_test, axis=1)    #2-tf.argmax
print("y_test:",y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)

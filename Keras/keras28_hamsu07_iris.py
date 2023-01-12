from sklearn.datasets import load_iris #꽃분류
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#데이터
datasets = load_iris()
print(datasets.DESCR) # 정보 설명 / 판다스.describe()/ ,info()
print(datasets.feature_names)

x = datasets.data
y= datasets['target']

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  # 원핫인코딩
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, 
    random_state=333, 
    test_size=0.9,
    stratify=y)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# #모델
# model = Sequential()
# model.add(Dense(50, activation='relu', input_shape= (4,)))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(3, activation='softmax'))

input1 = Input(shape=(4, ))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu' )(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(3, activation='softmax')(dense4)

model = Model(inputs=input1, outputs=output1)
model.summary()





#컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

#평가예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)


# 가장 큰 위치값을 찾아낸다
from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값):", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값):", y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)
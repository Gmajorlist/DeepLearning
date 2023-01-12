import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
arr = np.array([1,2,3,4,5,6,7])

from sklearn.preprocessing import OneHotEncoder        
ohe = OneHotEncoder()
print(y.shape)
y = y.reshape(581012, 1)
print(y.shape)
y = ohe.fit_transform(y)  # 위에 두줄을 한줄로 줄임
y = y.toarray()

x_train, x_test, y_train, y_test = train_test_split(x,y,
    shuffle=True,
    random_state=123,
    test_size=0.9,
    stratify=y)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#모델
model = Sequential()
model.add(Dense(86, activation='relu', input_shape=(54, )))
model.add(Dense(76, activation='sigmoid'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='softmax'))

#모델 (함수형)
input1 = Input(shape=(54, ))
dense1 = Dense(86, activation='relu')(input1)
dense2 = Dense(76, activation='sigmoid')(dense1)
dense3 = Dense(8, activation='linear')(dense2)
output1 = Dense(7, activation='softmax')(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()

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


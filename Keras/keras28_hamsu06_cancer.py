from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, random_state=333, test_size=0.2)
print(x.shape, y.shape) #(569, 30) (569,)

# scaler = MinMaxScaler() 
scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#모델구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30, )))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) ##이거랑 외워

input1 = Input(shape=(30, ))
dense1 = Dense(50, activation='linear')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu' )(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)

model = Model(inputs=input1, outputs=output1)
model.summary()





#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])#외워이것도
from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5, 
                              restore_best_weights=True,
                              verbose=1,
                               )
model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2 , callbacks=[earlyStopping],
          verbose=1
          )

#평가예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)

import numpy as np
y_predict = model.predict(x_test)
print(y_predict[:10])               # >정수형으로 바꿔줘야함
print(y_test[:10])

y_predict = np.round(y_predict)
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuracy_score:", acc)
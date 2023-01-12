import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler

datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)
print(y)
print(np.unique(y)) #[0 1 2]   y-> 0,1,2 만 있다
print(np.unique(y,return_counts=True))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  # 원핫인코딩
print(y)
print(y.shape)  #(178, 3)


x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, 
    random_state=333, 
    test_size=0.9,
    stratify=y )
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#모델
model = Sequential()
model.add(Dense(100, activation='relu', input_shape= (13,)))
model.add(Dense(80, activation='sigmoid'))
model.add(Dense(70, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(55, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(55, activation='linear'))
model.add(Dense(3, activation='softmax'))

#컴파일훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=5,  
                              restore_best_weights=True, 
                              verbose=1,
                               )
model.fit(x_train, y_train, epochs=99999999, batch_size=1,
          validation_split=0.2,
          verbose=1,
          callbacks=[earlyStopping])

#평가예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred:", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test:", y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)


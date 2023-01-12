import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#데이터 
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x,y,
    shuffle=True,
    random_state=123,
    test_size=0.9,
    stratify=y)

#모델
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(64, )))
model.add(Dense(90, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='linear'))
model.add(Dense(10, activation='softmax'))

#컴파일 훈련
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
print(y_test[:])
y_predict = model.predict(x_test[:])
print(y_predict)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred:", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test:", y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)






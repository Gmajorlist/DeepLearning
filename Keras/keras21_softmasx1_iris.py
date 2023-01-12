from sklearn.datasets import load_iris #꽃분류
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#데이터
datasets = load_iris()
# print(datasets.DESCR) # 정보 설명 / 판다스.describe()/ ,info()
# print(datasets.feature_names) #판다스.columns
x = datasets.data
y= datasets['target']
#print(x)
#print(y)
#print(x.shape, y.shape) # (150, 4 ) , (150, )

# 원핫인코딩 One-Hot-Encoding   (150,) -> (150, 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1] ->
# [[1, 0, 0]
# [0, 1, 0]
# [0, 0, 1]
# [0, 1, 0]]   (4,) -> (4, 3)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)  # 원핫인코딩
# print(y)
# print(y.shape) #(150, 3)

# from sklearn.preprocessing import OneHotEncoder
# one_hot_encoder = OneHotEncoder()


x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, 
    random_state=333, 
    test_size=0.9,
    stratify=y
    ) #false의 문제점은 훈련이 제대로 되지않음
# print(y_train)
# print(y_test)

#모델
model = Sequential()
model.add(Dense(50, activation='relu', input_shape= (4,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))


#컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

#평가예측
loss,accuracy = model.evaluate(x_test, y_test)
print('loss:', loss, 'accuracy:', accuracy)

   # 원 핫 인코딩 y=(150, ) -> (150,3)이 된다
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)



# 가장 큰 위치값을 찾아낸다
from sklearn.metrics import accuracy_score
import numpy as np

y_predict = np.argmax(model.predict(x_test), axis = 1)
print('y_predict: ', y_predict)

# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# print("y_pred(예측값):", y_predict)

y_test = np.argmax(y_test, axis = 1)
print('y_test: ', y_test)


# y_test = np.argmax(y_test, axis=1)
# print("y_test(원래값):", y_test)
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)



import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y ,
    shuffle= True, random_state=333, test_size=0.2)

#모델구성
model = Sequential()
model.add(Dense(1, input_dim=10)) # 행과 열
model.add(Dense(2, input_shape=(10, ))) #(13, )
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
#컴파일훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)


print("================================")
print(hist)#<keras.callbacks.History object at 0x0000021C42043DF0>
print("================================")
print(hist.history)
print("================================")
print(hist.history['loss'])
print("================================")
print(hist.history['val_loss'])
 
 
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red'
         , marker = '.', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue'
         , marker = '.', label = 'val_loss')
plt.grid() 
plt.xlabel('epoch')   
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()
plt.show()


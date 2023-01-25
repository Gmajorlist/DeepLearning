from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(np.unique(y_train, return_counts = True)) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)   

figure = plt.figure(figsize=(20,5))
for i in range(36):
    img = figure.add_subplot(4, 9, i+1, xticks=[], yticks=[])
    img.imshow(x_train[i], 'gray')       
    
    
model = Sequential()
model.add(Dense(units=64, input_shape=(28, 28 ), activation='relu')) 
model.add(Dense(units=64, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(units=32, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(units=16, activation='relu')) 
model.add(Flatten())
model.add(Dense(units=1, activation='softmax')) 
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc') # one-hot encoding 하지 않아도 되는 데이터이므로 loss= sparse_categorical_crossentropy

path = 'C:/study/keras/keras_save/MCP/'
MCP = ModelCheckpoint(monitor='acc', mode='auto', save_best_only=True,
                      filepath=path+'keras36_1_mnist.hdf5') 
ES = EarlyStopping(monitor='acc', mode='auto', 
                   patience=4, restore_best_weights=True) 
model.fit(x_train, y_train, epochs=32, batch_size=512,
          validation_split=0.1, callbacks=[ES, MCP])

metric = model.evaluate(x_test, y_test) # compile에서 metrics = acc를 지정했으므로 evaluate는 값을 배열 형태로 2개 반환함
print('loss: ', metric[0], 'acc: ', metric[1])
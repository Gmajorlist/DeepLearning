from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 


train_Gen = ImageDataGenerator(
    rescale= 1./255
)

test_Gen = ImageDataGenerator(
    rescale= 1./255
)

xy_train = train_Gen.flow_from_directory(
    'C:/_data/dogs-vs-cats/train/',
    target_size=(200, 200), 
    batch_size=1000,  
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True 
   
    ) 



xy_test = test_Gen.flow_from_directory(
     'C:/_data/dogs-vs-cats/test1/', 
    target_size=(200, 200), 
    batch_size=1000,  
    class_mode='binary', 
    color_mode='grayscale',
    shuffle=True )


x_train = np.load('C:/study/keras/keras_data/brain/brain_x_train.npy')
y_train = np.load('C:/study/keras/keras_data/brain/brain_y_train.npy')

x_test = np.load('C:/study/keras/keras_data/brain/brain_x_test.npy')
y_test = np.load('C:/study/keras/keras_data/brain/brain_y_test.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=128, validation_data=xy_test, validation_steps=4)  # steps_per_epoch: 1 에포크 당 배치 사이즈에 따른 훈련 횟수
hist = model.fit(x_train, y_train, epochs=128, batch_size=10, validation_data=([x_test, y_test]))  # steps_per_epoch: 1 에포크 당 배치 사이즈에 따른 훈련 횟수

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
print('acc: ', acc[-1])
print('loss: ', loss[-1])
print('val_acc: ', val_acc[-1])
print('val_loss: ', val_loss[-1])






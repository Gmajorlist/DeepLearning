from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2) ,  #  kerner_size 는 이미지를 조각내는 사이즈  
                 input_shape=(5, 5, 1)))  # 1은 흑백 / 3은 컬러  
model.add(Conv2D(filters=5, kernel_size=(2,2)))
model.add(Flatten())  # 한거 다 펴짐
model.add(Dense(10))
model.add(Dense(1))

model.summary()

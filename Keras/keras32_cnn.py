from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
 # 인풋은 ( 60000,5,5,1)
model.add(Conv2D(filters=10, kernel_size=(2,2) ,  #  kerner_size 는 이미지를 조각내는 사이즈  
                 input_shape=(5, 5, 1))) # 1은 흑백 / 3은 컬러   #(N,4,4,10)
 # (batch_size 1번이면 10번훈련, rows, columns, channels )
model.add(Conv2D(5, (2,2)))                                      #(N,3,3,5)
model.add(Flatten())  # 한거 다 펴짐                             #(N,45)
model.add(Dense(units=10))                                       #(N.10)
#인풋은 (batch_size, input_dim)
model.add(Dense(4, activation='relu'))                           #(N,1)

model.summary()

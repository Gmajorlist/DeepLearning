import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , SimpleRNN


#1.데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])   #(10, )
# y = ???

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
# 3일치 데이터로 잘랐다.
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)   #(7, 3) (7, )

x = x.reshape(7, 3, 1)    # [[1], [2], [3]]
                          # [[2], [3], [4]].....
print(x.shape)  #(7, 3, 1)         //  3,1  input shape가 된다


#모델구성

model = Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1)))
                            #(N, 3 , 1) -> ([batch(데이터개수) , timesteps(y데이터가 없음, 만들어줘야함),>> 하나씩만든게>> feature])              
                            #                                         자르고                                        만큼 일을 시켜라
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 64* (64 + 1 + 1 ) = 4224
# units * ( feature + bias + units ) = prams 
# 연산량이 많으면 성능이 좋고 속도가 느리다 

# cnn 4차원  dnn 2차원 rnn 3차원
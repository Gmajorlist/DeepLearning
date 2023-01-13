import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 확인
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) 흑백데이터  # 처음에는 이렇게 다운받는다 
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)

# 몇번째 그림이 무엇인지 확인
print(x_train[10])       #빛의 삼원색
print(y_train[10])
# 그림 
import matplotlib.pyplot as plt
plt.imshow(x_train[10], 'gray')
plt.show()





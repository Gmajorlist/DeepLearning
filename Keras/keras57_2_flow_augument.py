import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
augument_size= 40000
randidx = np.random.randint(x_train.shape[0], size= augument_size)
                               # 60000임 = 이유는 엑스트레인에 0번째는 6만임 
print(randidx)                               
print(len(randidx)) # 40000개를 추출 해서 변화시켜

x_augument = x_train[randidx].copy() # 메모리에서 원본건들지않고 복사본을 그대로 가져옴 엑스트레인을 건들지않고 
y_augument = y_train[randidx].copy()  # copy는 안전빵으로 하는게 좋다 

print(x_augument.shape, y_augument.shape)

x_augument = x_augument.reshape(40000, 28, 28, 1)


# 사만개를 변화 시켰다 원래잇던 육만개와 합쳐서 십만개가 됨 
# 왜지?

                                 
# 사진을 증폭도 가능 >훈련데이터에 쓰일 것 
train_datagen = ImageDataGenerator(
    rescale=1./255, #원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높음 (통상적인 learning rate를 사용할 경우). 
    #    그래서 이를 1/255로 스케일링하여 0-1 범위로 변환. 이는 다른 전처리 과정에 앞서 가장 먼저 적
    horizontal_flip=True,  #True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집음. 
        #원본 이미지에 수평 비대칭성이 없을 때 효과적. 즉, 뒤집어도 자연스러울 때 사용하면 좋음.
    vertical_flip=True,
    width_shift_range=0.1, # #그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
    height_shift_range=0.1, 
    rotation_range=0.5, # 이미지 회전 범위
    zoom_range=1.2, # #임의 확대/축소 범위
    shear_range=0.7, # 임의 전단 변환 (shearing transformation) 범위
    fill_mode='nearest' #  #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)
test_datagen = ImageDataGenerator(
    rescale=1./255  # test데이터는 rescale만 한다
                    # 이유 _ 평가데이터
)
# 증폭
x_augumented = train_datagen.flow(
    x_augument,
    y_augument,
    batch_size=augument_size,
    shuffle=True,
 )

print(x_augumented[0][0].shape)   
print(x_augumented[0][1].shape)     

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape,  y_train.shape)








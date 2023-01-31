import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 사진을 증폭도 가능 >훈련데이터에 쓰일 것 
train_datagen = ImageDataGenerator(
    rescale=1./255, #원본 영상은 0-255의 RGB 계수로 구성되는데, 
    # #    이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높음 (통상적인 learning rate를 사용할 경우). 
    # #    그래서 이를 1/255로 스케일링하여 0-1 범위로 변환. 이는 다른 전처리 과정에 앞서 가장 먼저 적
    # horizontal_flip=True,  #True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집음. 
    #     #원본 이미지에 수평 비대칭성이 없을 때 효과적. 즉, 뒤집어도 자연스러울 때 사용하면 좋음.
    # vertical_flip=True,
    # width_shift_range=0.1, # #그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
    # height_shift_range=0.1, 
    # rotation_range=0.5, # 이미지 회전 범위
    # zoom_range=1.2, # #임의 확대/축소 범위
    # shear_range=0.7, # 임의 전단 변환 (shearing transformation) 범위
    # fill_mode='nearest' #  #이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)
test_datagen = ImageDataGenerator(
    rescale=1./255  # test데이터는 rescale만 한다
                    # 이유 _ 평가데이터
)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/', # ad = 0 / nomal = 1 
    target_size=(200, 200), # 200에 200 하면 증폭이 됨 100에100이면 축소 지금 사진은 150 임
    batch_size=10000,  # 10개씩 자르겠다\
    class_mode='binary',
    # class_mode='categorical', # 수치  #원핫 되었어요   # 데이터 커지는 것도 고려를 해야한다. 카테고리컬을 할 경우 
    color_mode='grayscale',
    shuffle=True # 섞는다  
    #Found 160 images belonging to 2 classes.
    ) # directory < 폴더 (파일이모여있는곳)
# x = (160, 150, 150 , 1 ) 흑백  y = (160, 1:80 )


xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/', # ad = 0 / nomal = 1 
    target_size=(200, 200), # 200에 200 하면 증폭이 됨 100에100이면 축소 지금 사진은 150 임
    batch_size=10000,  # 10개씩 자르겠다
    class_mode='binary', # 수치
    color_mode='grayscale',
    shuffle=True )

print(xy_train)

# from sklean.datasets import load_iris
# datasets = load_iris()
# print(datasets)
# print(xy_train[0])
# print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape) #(10, 200, 200, 1) batch_size
print(xy_train[0][1].shape)

np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# np.save('./_date/brain/brain_1_xy_train_x.npy', arr=xy_train[0])
np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_tset.npy', arr=xy_test[0][1])


# print(type(xy_train))  # 타입을 찍으면 데이터 형태를 봄 #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))   #<class 'tuple'> _ 리스트랑 똑같다 
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
#numpy는 연산이 빠르다. numpy로 저장하고 땡겨씁시다.



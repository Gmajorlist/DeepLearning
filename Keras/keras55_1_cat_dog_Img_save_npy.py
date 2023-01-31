# https://www.kaggle.com/competitions/dogs-vs-cats/overview

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


import os

# 원본 이미지 파일 경로
img_path = 'C:/_data/dogs-vs-cats/train/'

# 새로운 폴더 경로
cat_folder = './_data/dogs-vs-cats/train/train_cat/'
dog_folder = './_data/dogs-vs-cats/train/train_dog/'

# 새로운 폴더 생성
if not os.path.exists(cat_folder):
    os.makedirs(cat_folder)
if not os.path.exists(dog_folder):
    os.makedirs(dog_folder)

# 원본 이미지 파일 리스트
img_list = os.listdir(img_path)

# 원본 이미지 파일 리스트에서 cat 파일과 dog 파일 분류
for img in img_list:
    src = img_path + img
    if 'cat' in img:
        dst = cat_folder + img
        os.rename(src, dst)
    elif 'dog' in img:
        dst = dog_folder + img
        os.rename(src, dst)
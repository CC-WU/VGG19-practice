# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:04:04 2020

@author: User
"""

from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.models import  load_model
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import tensorflow as tf
 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 

# 初始計算各類別的準確率
class_name = {'again':0,'back_to_top':0,'backward':0,'continue':0,'end_play':0,'end_program':0,'end':0,'enter':0,'faster':0,'forward':0,'loudly':0,'next_page':0,'pause':0,'previous_page':0,'repeat':0,'say_again':0,'search_again':0,'search':0,'slower':0,'speed':0,'start_playing':0,'start':0,'whisper':0}
time_sum = 0

def count_acc(image_path, name): # 計算命中個數
    image_class = str(image_path).split('\\')[-1]
    image_class = str(image_class).split('.')[0]
    name = str(name).split('[\'')[-1].split('\']')[0]
    if name in image_class:
        class_name[name] = int(class_name[name])+1

def count_avg(class_nums = 23): # 算平均值
    ccou = 0
    for key, value in class_name.items() :
        ccou += int(value)
    print('命中個數: ', ccou)
    ccou = ccou / (5*class_nums)
    print('準確率: ', ccou)
    print('平均花費時間: ', time_sum/(5*class_nums))

#label = np.array(['cat','dog'])
label = np.array(["again", "back_to_top", "backward", "continue", "end", "end_play", "end_program", "enter", "faster", "forward", "loudly", "next_page", "pause", "previous_page", "repeat", "say_again", "search", "search_again", "slower", "speed", "start", "start_playing", "whisper"])

#载入模型
start_time = time.time()
#model=load_model('model_cnn.h5')
model=load_model('./model/checkpoint-490e-val_acc_0.94.h5')
end_time = time.time()
print('model load time: ', end_time - start_time)



dict_genres = {0: 'again', 1: 'back_to_top', 2: 'backward', 3: 'continue', 4: 'end', 5: 'end_play', 6: 'end_program', 7: 'enter', 8: 'faster', 9: 'forward', 10: 'loudly', 11: 'next_page', 12: 'pause', 13: 'previous_page', 14: 'repeat', 15: 'say_again', 16: 'search', 17: 'search_again', 18: 'slower', 19: 'speed', 20: 'start', 21: 'start_playing', 22: 'whisper' }
path = os.getcwd()
train_data_dir = path + '\\Public_test\\'
level = os.listdir(train_data_dir)
for i in range(len(level)):
    img_path = train_data_dir + level[i] # 取檔名
    start_time = time.time()
    # 本地图片进行预测
    # 导入图片
    image = load_img(img_path)
    #plt.imshow(image)
    #plt.show()
    image=image.resize((150,150))
    image=img_to_array(image)
    image=image/255
    image=np.expand_dims(image,0)
    print(image.shape)
    print(model.predict_classes(image))
    print(img_path, ' Predicted:', label[model.predict_classes(image)])
    #plot_preds(img, preds,labels)
    end_time = time.time()
    readfile_time = end_time - start_time # 計算測試時間
    time_sum += readfile_time # 計算總花費時間
    print('image test time: ', readfile_time)
    count_acc(img_path, dict_genres[int(model.predict_classes(image))]) # 累加命中個數

count_avg() # 計算準確率

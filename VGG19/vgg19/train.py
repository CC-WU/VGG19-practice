# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:47:33 2020

@author: User
"""

from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import  load_model
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import tensorflow as tf
 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

train_dir = 'images/train'  # 训练集数据路径
val_dir = 'images/val' # 验证集数据
classes = 23 #類別數
batch = 64 #每次训练传入32张照片
epochs = 500 #訓練代數

nb_train_samples = get_nb_files(train_dir)      # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)       #验证集样本个数
nb_epoch = int(epochs)                # epoch数量
batch_size = int(batch)  

#VGG16模型是使用imagenet数据集训练出来的
#include_top=False代表只需VGG16模型中的卷积和池化层
#input_shape=(150,150,3)：特征提取
vgg19_model=VGG19(weights='imagenet',include_top=False,input_shape=(150,150,3))
# 1. include_top：是否包含頂部(Top) 3層『完全連階層』(fully-connected layers)。
#       include_top = False：只利用VGG16萃取特徵，後面的分類處理，都要自己設計。反之，就是全盤接受VGG16，只是要改變輸入而已。
#       注意!! 頂部(Top)指的是位於結構圖最後面，因為它是一個『後進先出法』的概念，一層一層堆疊上去，最上面一層就是頂部(Top)。
# 2. weights：使用的權重，分兩種
#       imagenet：即使用ImageNet的預先訓練的資料，約100萬張圖片，判斷1000類別的日常事物，例如動物、交通工具...等，我們通常選這一項。
#       None：隨機起始值，我沒試過，請有興趣的讀者自行測試。


#搭建全连接层
top_model=Sequential()
top_model.add(Flatten(input_shape=vgg19_model.output_shape[1:]))#图片输出四维，1代表数量
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(classes,activation='softmax'))

#把VGG16和全连接层整合
model=Sequential()
model.add(vgg19_model)
model.add(top_model)

# 改變ouput的分類法的另一種方式
# # 從頂部移出一層
# model.layers.pop()
# model.outputs = [model.layers[-1].output]
# model.layers[-1].outbound_nodes = []
# # 加一層，只辨識10類
# from keras.layers import Dense
# num_classes=10
# x=Dense(num_classes, activation='softmax')(model.output)
# # 重新建立模型結構
# model=Model(model.input,x)

#数据增强
train_datagen=ImageDataGenerator(
    rotation_range=40,#随机旋转度数
    width_shift_range=0.2,#随机水平平移
    height_shift_range=0.2,#随机竖直平移
    rescale=1/255,#数据归一化
    shear_range=0.2,#随机裁剪
    zoom_range=0.2,#随机放大
    horizontal_flip=True,#水平翻转
    fill_mode='nearest',#填充方式
)
test_datagen=ImageDataGenerator(
    rescale=1/255,#数据归一化
)
 
#生成训练数据
train_generator=train_datagen.flow_from_directory(
    'images/train',#从训练集这个目录生成数据
    target_size=(150,150),#把生成数据大小定位150*150
    batch_size=batch,
)
#测试数据
test_generator=test_datagen.flow_from_directory(
    'images/val',#从训练集这个目录生成数据
    target_size=(150,150),#把生成数据大小定位150*150
    batch_size=batch,
)

#更好地保存模型 Save the model after every epoch.
output_model_file = './model/checkpoint-{epoch:02d}e-val_acc_{val_accuracy:.2f}.h5'
#keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
checkpoint = ModelCheckpoint(output_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)

#查看定义类别分类
print(train_generator.class_indices)
#定义优化器、代价函数、训练过程中计算准确率
model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
#传入生成的训练数据、每张图片训练1次，验证数据为生成的测试数据
history_ft = model.fit_generator(train_generator,epochs=epochs,validation_data=test_generator,callbacks=[checkpoint])

#保存最終模型
#model.save('model_vgg19.h5')

def plot_training(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r-')
  plt.plot(epochs, val_acc, 'b')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r-')
  plt.plot(epochs, val_loss, 'b-')
  plt.title('Training and validation loss')
  plt.show()
 
# 训练的acc_loss图
plot_training(history_ft)

# 自行定義 VGG19 結構
# class Vgg19(object):
#     """
#     A trainable version VGG19.
#     """

#     def __init__(self, bgr_image, num_class, vgg19_npy_path=None, trainable=True, dropout=0.5):
#         if vgg19_npy_path is not None:
#             self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
#         else:
#             self.data_dict = None
#         self.BGR_IMAGE = bgr_image
#         self.NUM_CLASS = num_class
#         self.var_dict = {}
#         self.trainable = trainable
#         self.dropout = dropout

#         self.build()

#     def build(self, train_mode=None):

#         self.conv1_1 = self.conv_layer(self.BGR_IMAGE, 3, 64, "conv1_1")
#         self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
#         self.pool1 = self.max_pool(self.conv1_2, 'pool1')

#         self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
#         self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
#         self.pool2 = self.max_pool(self.conv2_2, 'pool2')

#         self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
#         self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
#         self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
#         self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
#         self.pool3 = self.max_pool(self.conv3_4, 'pool3')

#         self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
#         self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
#         self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
#         self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
#         self.pool4 = self.max_pool(self.conv4_4, 'pool4')

#         self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
#         self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
#         self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
#         self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
#         self.pool5 = self.max_pool(self.conv5_4, 'pool5')

#         self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
#         self.relu6 = tf.nn.relu(self.fc6)
#         if train_mode is not None:
#             self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
#         elif train_mode:
#             self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

#         self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
#         self.relu7 = tf.nn.relu(self.fc7)
#         if train_mode is not None:
#             self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
#         elif train_mode:
#             self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

#         self.fc8 = self.fc_layer(self.relu7, 4096, self.NUM_CLASS, "fc8")

#         self.prob = tf.nn.softmax(self.fc8, name="prob")

#         self.data_dict = None

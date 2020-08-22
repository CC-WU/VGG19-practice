# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:05:25 2020

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'again_1.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
block4_pool_features = model.predict(x)

plt.imshow(block4_pool_features[0][3])
plt.show()
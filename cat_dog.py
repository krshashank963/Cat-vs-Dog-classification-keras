# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:49:01 2020

@author: SHASHANK RAJPUT
"""




from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size=64

training_set = train_datagen.flow_from_directory('dataset/training_set',
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary',
shuffle=True)

test_set = test_datagen.flow_from_directory('dataset/test_set',
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary')

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import adam
import numpy as np

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))
classifier.add(Conv2D(64, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(64))
classifier.add(Activation("relu")) 
classifier.add(Dense(128))
classifier.add(Activation("relu")) 
classifier.add(Dense(activation = 'sigmoid', units=1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(training_set,
                         steps_per_epoch=np.ceil(training_set.samples / batch_size),
                         epochs=6, 
                         validation_steps=np.ceil(test_set.samples / batch_size),
                         validation_data=test_set)


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
test_image = image.load_img("E:\study material\Deep Learning A-Z™ Hands-On Artificial Neural Networks\dataset\single_prediction\dog.jpg", target_size = (100, 100)) 
plt.imshow(test_image)
plt.grid(None) 
plt.show()
res_list= [" a cat","a dog !"]
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(res_list[int(classifier.predict(test_image))])
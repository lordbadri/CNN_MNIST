# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:26:48 2017

@author: abhinath
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop, Adadelta
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("train.csv")
    
test_images = (pd.read_csv("test.csv").values).astype('float32')
    
train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')
    
print(train_images.shape)
train_images = train_images.reshape((42000, 28 * 28))
    
print(test_images.shape)
    
train_images = train_images / 255
test_images = test_images / 255
    
    
train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]
print(num_classes)

seed = 43
np.random.seed(seed)

model = Sequential()
model.add(Dense(128, activation='relu',input_dim=(28 * 28)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer=Adadelta(),metrics=['accuracy'])

history=model.fit(train_images, train_labels, nb_epoch=20, batch_size=128, validation_split=.05)

predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("keras_mnist_pred.csv", index=False, header=True)

history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(acc_values, 'bo')
plt.plot(val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()
            
    

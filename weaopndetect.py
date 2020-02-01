from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


m,n = 50,50
model = tf.keras.models.load_model("model_latest.h5")
#print(model.summary())

#files=os.listdir(path);

'''x=[]
for i in files:
    im = Image.open(path + i);
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)
x=np.array(x);
predictions = model.predict(x)
print (predictions)
print (model.summary())'''
def biggun(img)
    x=[]
    im = Image.open(img);
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)
    x=np.array(x);
    predictions = model.predict(x)
    for i in predictions:
        if any((j>=0.7for j in i)):
            print("hooray")

#print (model.summary())

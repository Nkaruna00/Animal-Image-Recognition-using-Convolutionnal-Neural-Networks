#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 22:25:35 2018

@author: nithushan
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing import image
import os



# Softmax function used in the callfunction for training
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference




# Function to print the model prediction result

def printer(resultat,filenames):
    for x,y in np.ndenumerate(resultat):
        if(resultat[x] > 0.50):
            print("%s représente un chien (%f)" % (str(filenames[x[0]]),resultat[x]))
        else:
            print("%s représente un chat (%f)" % (str(filenames[x[0]]),1 - resultat[x]))

        
           
#Load the model


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("model.h5")


from keras.preprocessing.image import ImageDataGenerator

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('donnees',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            shuffle = False)



resultat = classifier.predict_generator(test_set,len(test_set))
tmp = os.listdir("donnees/stockage")
tmp.sort()
filenames = np.asarray(tmp)


printer(resultat,filenames)






1





































































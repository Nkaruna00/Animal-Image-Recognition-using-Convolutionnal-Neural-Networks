# Animal Image Recognition using Convolutionnal Neural Networks
Classifies images according to whether they are an image containing a cat or a dog using a convolutional neural network.
Made with Python3 and Keras with a Tensorflow backend.


## Description

Classifies images according to whether they are an image containing a cat or a dog using a convolutional neural network.
Made with Python3 and Keras with a Tensorflow backend  

The Data is augmented with ImageGenerator() to have more images.

The model consists of :

* A first convolution layer containing 32 neurons with a Relu activation function and a MaxPooling to spatially reduce the data

* The second layer contains 32 neurons with a Relu activation function and a MaxPooling to spatially reduce the data

* The third layer flattens the multi-dimensional input tensors into a single dimension

* The fourth layer contains 128 neurons with a Relu activation function.

* The fifth layer contains 1 neuron with a sigmoid activation function to determine the presence of a cat or a dog (close to 0 for a cat, close to 1 for a dog)  

The trained model is saved in the model.h5 and model.json files.

## Getting Started

### Dependencies

* Python3
* Cat / Dog images dataset
* Keras with Tensorflow Backend

### Executing program

* Run the load.py to train the model Or Run model.py to load the trained model and test
```
python3 cnn.py or python3 model.py
```

## Authors

KARUNAKARAN Nithushan


This project is licensed under the MIT License - see the LICENSE.md file for details


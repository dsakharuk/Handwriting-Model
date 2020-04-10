import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from PIL import Image
import h5py


updated_size =(90,90)
prediction_dict = np.load('mapping_dictionary.npy',allow_pickle='TRUE').item()


def preprocess_img(image):
    '''
    This method processes the image into the correct expected shape in the model (90, 90). 
    ''' 
    if (image.mode == 'RGB'): 
        # Convert RGB to grayscale. 
        image = image.convert("L")
    image = image.resize(updated_size)
    return image


def image_loader(image):
    ''' 
    This method loads the image into a numpy array. 
    '''
    image = np.array(image)
    image = np.expand_dims(image,axis=2)
    image = np.expand_dims(image,axis=0)
    image = image.astype('float16',copy=False)
    image = image/255
    return image

class math_symbol_predictor: 
    def __init__(self):
        self.model = tf.keras.models.load_model('67_acc_keras_model')
 

    def predict(self, request):
        '''
        This method reads the file uploaded from the Flask application POST request, 
        and performs a prediction using the model. 
        '''
        f = request.files['image']
        image = Image.open(f)
        image = preprocess_img(image)
        image = image_loader(image)
        model_output = self.model.predict_classes(image,batch_size = 10,verbose = 1)
        prediction = prediction_dict[model_output[0]]
        return prediction
    
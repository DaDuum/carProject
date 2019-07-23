# -*- coding: utf-8 -*-
"""
Created on May 10

@author: Admin
"""

import socketio 
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


import tensorflow as tf  
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  
                                    
sess = tf.Session(config=config)  
set_session(sess)  # set this TensorFlow session as the default session for Keras  

#####

model_name = 'UltraModel.hdf5'

sio = socketio.Server()

app = Flask(__name__) #'__main__' 
speed_limit = 30
def img_preprocess(img):
     img = img[60:135, : , :]
     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
     img = cv2.GaussianBlur(img, (3, 3), 0)
     img = cv2.resize(img, (200, 66))
     img = img/255
     return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))  
    throttle = 1.0 - speed/speed_limit
    #throttle = 1.0
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
     

@sio.on('connect') 
def connect(sid, environ):
    print('Connected') 
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
            'steering_angle' : steering_angle.__str__(),
            'throttle' : throttle.__str__()
            })

if __name__ == '__main__':
    model = load_model(model_name)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
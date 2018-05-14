import os
import numpy as np
import tensorflow as tf
from create_images import create_images
from model import CNN
from lib.model_io import get_model_id

#create 64X17 images from the binary files
cr=create_images()
train_params, train_params_value=cr.create_train_params()
valid_params, valid_params_value=cr.create_dev_params()

model_id = get_model_id()

# Create the network
network = CNN(train_params, train_params_value, valid_params, valid_params_value)

# Define the train computation graph
network.define_train_operations()

# Train the network
sess = tf.Session()
try:
    network.train(sess)
except KeyboardInterrupt:
    print()
finally:
    sess.close()

import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants


def save_tf_model(path_to_model, path_to_saving):
    print("Loading the .h5 model from {} path ...".format(path_to_model))
    model = tf.keras.models.load_model(path_to_model, compile=False)
    print("Saving .h5 model as the .pb file at {} path ...".format(path_to_saving))
    tf.saved_model.save(model, path_to_saving)



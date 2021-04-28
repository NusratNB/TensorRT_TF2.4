import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants


def save_tf_model(path_to_model, path_to_saving):
    print("Loading the .h5 model from {} path ...".format(path_to_model))
    model = tf.keras.models.load_model(path_to_model, compile=False)
    print("Saving .h5 model as the .pb file at {} path ...".format(path_to_saving))
    tf.saved_model.save(model, path_to_saving)


def load_saved_model(saved_model_dir):
    print("Loading saved model from {} path ...".format(saved_model_dir))
    model = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
    return model


def tf_to_trt_graph(precision_mode="float32", saved_model_dir=None, max_workspace_size_bytes=4000000000):

    if precision_mode == "float32":
        precision_mode = trt.TrtPrecisionMode.FP32
        output_prefix = "_FP32"
    elif precision_mode == "float16":
        precision_mode = trt.TrtPrecisionMode.FP16
        output_prefix = "_FP16"

    out_save_dir = os.path.join(saved_model_dir, output_prefix)
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=max_workspace_size_bytes
    )


import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
import argparse
import yaml


def save_tf_model(path_to_model, path_to_saving):
    print("Loading the .h5 model from {} path ...".format(path_to_model))
    model = tf.keras.models.load_model(path_to_model)
    print("Saving .h5 model as the .pb frozen graph at {} path ...".format(path_to_saving))
    tf.saved_model.save(model, path_to_saving)


def load_saved_model(saved_model_dir):
    print("Loading saved model from {} path ...".format(saved_model_dir))
    model = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
    return model


def tf_to_trt_graph(precision_mode=None, saved_model_dir=None, max_workspace_size_bytes=None):
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
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=conversion_params
    )
    print('Converting {} to TF-TRT graph precision mode {}... '.format(saved_model_dir, precision_mode ))
    converter.convert()
    print('Saving converted model to {} '.format(saved_model_dir))
    converter.save(out_save_dir)
    print("Done!")


if __name__ == '__main__':

    with open("config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile)
        print("Read successful")
    config = data[0]["Config"]

    save_tf_model(config['dir_to_model'], config['dir_fr_graph'])
    tf_to_trt_graph(config['pr_mode'], config['dir_fr_graph'], int(config['pr_mode']))



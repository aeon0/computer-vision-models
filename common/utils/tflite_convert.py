import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import subprocess
import os
from typing import List, Union


def tflite_convert(model: Union[str, tf.keras.Model], save_dir: str, quantize: bool, compile_edge_tpu_flag: bool, dataset: List, model_is_quantized: bool = False):  
  converter = None
  if isinstance(model, str):
    model_path = model
    model = tf.keras.models.load_model(model, compile=False)
  
    if model_path.lower().endswith(".h5"):
      save_dir = os.path.dirname(save_dir)
      converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
      converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
  else:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

  if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
      # Get sample input data as a numpy array in a method of your choosing.
      for data in dataset:
        # not sure why, for single input data use [data] for multiple inputs just data...
        yield [data]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = False

    tflite_model = converter.convert()
    tflite_model_path = save_dir + "/model_quant.tflite"
    open(tflite_model_path, "wb").write(tflite_model)
    print("Saved Quantized TFLite Model to: " + tflite_model_path)

    # Compile for EdgeTpu
    if compile_edge_tpu_flag:
      print("Compile for EdgeTpu")
      print(tf.__version__)
      subprocess.run("edgetpu_compiler -a -s -o %s %s" % (save_dir, tflite_model_path), shell=True)
      print("Saved Quantized EdgeTpu Model to: " + save_dir)
  
  else:
    tflite_model = converter.convert()
    tflite_model_path = save_dir + "/model.tflite"
    open(tflite_model_path, "wb").write(tflite_model)
    print("Saved TFLite Model to: " + tflite_model_path)

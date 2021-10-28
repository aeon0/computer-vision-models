## Some utility functions

### save_to_storage.py
Callback implementation to save a model to the local harddrive as a tensorflow model as well as save metrics as
json file.

### plot_metrics.py
Metrics that are saved by save_to_storage.py can be visualized in a graph.

### tflite_converter.py
Converting a tensorflow model to tflite model and optionally quantize the weights. For running the model on edgetpu
there is one more compile step with the edgetpu command line tool needed (https://coral.ai/docs/edgetpu/compiler):
```bash
# install edgetpu command line tool
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
# compile tflite model to tflite model for edgetpu
edgetpu-compiler path/to/model.tflite
```

# Computer Vision Models
A set of tensorflow models for different computer vision tasks.

## Getting started

### Dependencies
- [Docker](https://docs.docker.com/engine/install/ubuntu/) to create a image and containers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) To access GPUs from the containers
- [MongoDB](https://docs.mongodb.com/manual/installation/) to store and read training data

### Docker
Check the `Dockerfile` to see what kind of dependencies are installed for the docker image. The `.vscode/task.json` shows all the flags for a run command:
```bash
# add current user to the docker group
sudo usermod -aG docker $USER
# allow access to screen from docker container (best added to .bashrc)
xhost local:root

docker run -it --rm \
  --gpus all \
  --network="host" \
  --privileged -v /dev/bus/usb:/dev/bus/usb \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  --mount "type=bind,source=$(pwd),target=/home/computer-vision-models" \
  computer-vision-models bash

# access gpus
# allow access to localhost (for mongodb)
# allow access to usb for Coral USB Accelerator
# to display images on screen
# bind local workspace with the one on docker to avoid rebuilding for every code change
```

### EdgeTpu support
Dont forget to plugin the Coral USB Accelerator :)

### Spotty support
Train on AWS with https://github.com/spotty-cloud/spotty. To have a working mongodb setup these steps should be done:
- Install spotty cli with `pip install spotty`
- Create a on-demand instance with a volume size that fits all the training data
- Install mongodb for the operating system you are using
- In /etc/mongod.conf make sure that bindIp: 0.0.0.0 to allow access outside of localhost
- On the outbound rules in the security group allow traffic (specifally port 27017) from inside the VPC
- Add the local IP to the [aws_mongodb] config in config.ini and use the aws_config in train.py

## Folder structure
Overview of the folder structure and it's contents. A README.md in each folder provides more detailed documentation.
#### common
Includes all the common code shared between all the different model implementations. E.g. data reading, storing/plotting results, logging, saving models or converting models to different platforms.
#### data
In an attempt to be able to combine same kinds of data (e.g. semseg, 2D od, etc.) from different sources, custom label specs are used. Since data can also come in many different forms, all data is combined into MongoDB for easy storage and access. For each datasource an "upload script" exists that converts the source data to the internal label spec and uploads it do MongoDB.
#### models
Different model implementations for different computer vision tasks. Includes all necesarry pre- and post-processing, training, model description, inference, etc.
#### eval
Does not exist yet, work in progress. But the idea is that just like different data sources are combined to an internal label spec, different models implementing the same type of computer vision algo (e.g. semseg or 2D detections) should also output a common output spec to be evaluated against each other.

## Tests
Tests should be in the same location as the file that is tested with the naming convention $(FILE_NAME)_test.py. To run tests call `$ pytest` in the root directory or use your favorite test runner (e.g. pycharm or vs code).

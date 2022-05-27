# Instructions

## Requirements

* NVIDIA GPU with the latest driver installed
* docker / nvidia-docker
This build has been tested with Nvidia Drivers 510.47.03 and CUDA 11.6 on a Ubutun 20.04 machine. Please update the base image if you plan on using older versions of CUDA.

This build has been tested with Nvidia Drivers 460.91.03 and CUDA 11.2 on a Ubutun 20.04 machine.
Please update the base image if you plan on using older versions of CUDA.

## Build
Build the image with:
```
docker build -t project-dev -f Dockerfile .
```
Note: dockerfile will try to build TensorFlow 2.5.0, but the tested setup is in 2.7.0, so in requirements.txt, tensorflow 2.7.0 is installed to replace older version

Create a container with:
```
docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash
```
and any other flag you find useful to your system (eg, `--shm-size`).

Start a container with 
```
docker start -i <container-name>
```
start a new process of a container 
```
docker exec -it <container-name> bash
```
## Set up

Once in container, you will need to install gsutil, which you can easily do by running:
```
curl https://sdk.cloud.google.com | bash
```

Once gsutil is installed and added to your path, you can auth using:
```
gcloud auth login
```

## Debug
* Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) if you run into any issue with the installation of the
tf object detection api

## Updating the instructions
Feel free to submit PRs or issues should you see a scope for improvement.

# DetectionTutorialSpaceNet
This is a tutorial on training a network to detect buildings with the SpaceNet data.

## Introduction
Recently DigitalGlobe released aerial image dataset of Rio De Janeiro. This unprecedented data set includes fully developed regions with residential and commercial buildings as well as rural areas with none at all. The perimeter and shape of each building is provided with latitude and longitude coordinates. acting as the labels for performing building detection. A blog post with preliminary detection results is posted [here](https://devblogs.nvidia.com/parallelforall/exploring-spacenet-dataset-using-digits). This tutorial discusses how the detection results were achieved in greater detail so that readers can duplicate presented results presented and act as a starting point for improving mean average precision (mAP), recall and precision.  
m
## Dataset Preparation
The latitude and longitude coordinates act as the labels for the data. These coordinates need to be converted to pixel space before being used for training.
Over XXX images are provided with
The BEE-Sharp script xxx.py was used to convert the

Use the coordinates to create bounding boxes

Omit images that have less than 50% pixel information.

Fill in the black areas with Gaussian noise

## Create a Train, Val, and Test Set
## Network Configuration
The default DetectNet network configuration with a few modifications to the network input image parameters and python layers is used to generate the results provided in this document.


## Training with DIGITS
Creating the Dataset and Training with DIGITS

## Results

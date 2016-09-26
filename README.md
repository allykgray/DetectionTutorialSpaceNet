# Detection Tutorial Using The SpaceNet Data
This is a tutorial on training a network to detect buildings with the SpaceNet data.

## Introduction
Recently DigitalGlobe released aerial image dataset of Rio De Janeiro. This unprecedented data set includes fully developed regions with residential and commercial buildings as well as rural areas with none at all. The perimeter and shape of each building is provided with latitude and longitude coordinates. acting as the labels for performing building detection. A blog post with preliminary detection results is posted [here](https://devblogs.nvidia.com/parallelforall/exploring-spacenet-dataset-using-digits). This tutorial discusses how the detection results were achieved in greater detail so that readers can duplicate the presented results presented and act as a starting point for improving mean average precision (mAP), recall, and precision.

## Hardware and Software
NVIDIA GPUs and open source software, Deep Learning GPU Training System, [DIGITS 4](https://github.com/nvidia/digits) is used to create these results. For my testing, I used Openstack VM with a Tesla M60. The M60 is a dual GPU card that enables multi-GPU training or two trainings in parallel. Although DIGITS supports to use of both Torch and Caffe, I used [NVIDIA's Caffe branch version 15](https://github.com/nvidia/caffe) to generate the results shown here. Prebuilt packages of DIGITS 4 with Caffe v15 and torch can be downloaded through [NVIDIA developer portal](http://developer.nvidia.com/digits). Otherwise it can be installed from source at github.

## Dataset Preparation
The latitude and longitude coordinates act as the labels for the data. These coordinates need to be converted to pixel space before being used for training.
Over 7000 images are provided with building information.
The BEE-Sharp script convert_Kaggle_ag.py was used to convert the latitude and longitude coordinates to pixel space. The coordinates for each building are published to a csv file. The boundaries of the building is used to create a bounding box that encompasses the building.

![Data Image With Label](imgs/*.png)

Some regions are entirely omitted from the data and appear as black or blank space. No terrain information is present and pixels are zero in each color band. All of the images with blank regions occupying more than 50% of the image are omitted from the data set. Gaussian noise is inserted into the blank regions of the image if it is above the 50% threshold. After filtering the dataset by this metric is it reduced from over 7000 to approximately 4000. Three example images with blank regions less than 50% is in the example image below.

![Example_Images](imgs/ExampleDataWithNoise.png)
Figure #. Three example images used in the data set. The same images are in each row. The second row shows the images after Gaussian noise has been applied.


The final step before creating the data base is formatting images and labels properly. Each image name is a numerical string and its label has the exact same name minus the extension. For example one image titled "000001.png" has a corresponding label file "000001.txt." All of my images and labels are comprised of numerical strings. After the labels and images have been created they are placed in their own directories and used to create a dataset in DIGITS. Select the RGB, PNG lossless encoding, and image dimensions, and desired resizing dimensions. Give the dataset a name. An image of the setup is displayed in the figure below.

![Dataset Creation in Digitis](imgs/DataSetCreationPage.png)

## Network Configuration
The default DetectNet network configuration with a some minor modifications is used to train the network. This network is comprised of a data augmentation layers for preprocessing data, modified version of the GoogleNet CNN, and post-processing layers to predict object locations. The DetectNet data transformation layers are at the beginning of the network are defined near the beginning of the network. This is defined for both the train and validation data and requires information about the training data. Below is a snippet from my network for the training set. Two parameters from the default DetectNet network are changed in the code below, image_size_x, image_size_y, and [crop_bboxes, do we know what this does?]. Although the image size is 1280x1280, smaller dimensions 512x512 are entered as the image_size to define dimensions for random cropping. The main reason for this is to reduce memory usage during training. The crop bounding boxes parameter is set to false in this layer. During testing, we were unable to achieve low losses and mAP greater than one when this is set to true. Each M60 GPU has 8 GB of memory and performing random cropping allowed me to train with a larger batch size.

<pre></code>
layer {
  name: "train_transform"
  type: "DetectNetTransformation"
  bottom: "data"
  bottom: "label"
  top: "transformed_data"
  top: "transformed_label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_value: 127.0
  }
  detectnet_groundtruth_param {
    stride: 16
    scale_cvg: 0.4
    gridbox_type: GRIDBOX_MIN
    min_cvg_len: 20
    coverage_type: RECTANGULAR
    image_size_x: 512
    image_size_y: 512
    obj_norm: true
    crop_bboxes: false
  }
</code></pre>

The validation transform layer of network file shown below. The image_size parameters are the dimensions of the input data for the validation transform layer because no cropping on input data is performed during validation testing. This could be done if desired but it was not experimented by the author.

<pre><code>
layer {
  name: "val_transform"
  type: "DetectNetTransformation"
  bottom: "data"
  bottom: "label"
  top: "transformed_data"
  top: "transformed_label"
  include {
    phase: TEST
  }
  transform_param {
    mean_value: 127.0
  }
  detectnet_groundtruth_param {
    stride: 16
    scale_cvg: 0.4
    gridbox_type: GRIDBOX_MIN
    min_cvg_len: 20
    coverage_type: RECTANGULAR
    image_size_x: 1280
    image_size_y: 1280
    obj_norm: true
    crop_bboxes: false
  }
}
</pre></code>

As mentioned previously, the CNN is a modified version of GoogleNet, such that the output is a feature map rather than a vector that predicts a category. The layers after the CNN are python layers and are used to predict bounding boxes based on output feature map from the CNN. These layer names and types are unchanged relative to the DetectNet's default configuration, but the parameters are different. The image dimensions match this data set for each layer, 1280 x 1280.


The cluster layers requires five parameters to predict building locations. This uses the function groupRectangles function from OpenCV to generate a list of bounding boxes. To ascertain the results presented here, 0.06, 3, and 0.02 were used for the thresholds and equivalence (eps) parameters. This clusters the input rectangles using the rectangle equivalence criteria (eps) combining rectangles with similar sizes and locations. In cases where the cluster of boxes is less than the group threshold small clusters containing less than or equal to gridbox_rect_threshold are omitted. Predicted rectangles are the average of box clusters greater than this parameter. The minimum height is the last parameter provided to the cluster Python layer and is 10 in the layer section presented below.

<pre><code>
layer {
  name: "cluster"
  type: "Python"
  bottom: "coverage"
  bottom: "bboxes"
  top: "bbox-list"
  include {
    phase: TEST
  }
  python_param {
    module: "caffe.layers.detectnet.clustering"
    layer: "ClusterDetections"
    parameters - img_size_x, img_size_y, stride,
      gridbox_cvg_threshold, gridbox_rect_threshold, gridbox_rect_eps, min_height
    param_str: "1280, 1280, 16, 0.06, 3, 0.02, 10"
  }
}
layer {
  name: "cluster_gt"
  type: "Python"
  bottom: "coverage-label"
  bottom: "bbox-label"
  top: "bbox-list-label"
  include {
    phase: TEST
  }
  python_param {
    module: "caffe.layers.detectnet.clustering"
    layer: "ClusterGroundtruth"
    param_str: "1280, 1280, 16"
  }
}
</pre></code>

A copy of my network file can be downloaded [here](models/train_val.prototxt).

## Training with DIGITS
For training the Adam solver is used with a exponential decay learning policy and initial learning rate of 1e-05. Shortly after training begins the mAP, recall, precision begin to increase. Although the these metrics reach values greater than zero the loss values are still high. There is still room for improvement with the network configuration and training settings.


![Training_Results](imgs/trainingResults.png)

## Results

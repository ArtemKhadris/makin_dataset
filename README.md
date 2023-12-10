# Code for creating a dataset for Yolov8.
dataset_builder for object detection and dataset_for_class for object classification.
Both codes distort the image, change perspective and rotate, change colors, brightness, etc.
For a dataset, you need to have a set of pictures with names corresponding to classes (object names). 
For detection it is necessary to have a background against which the image will be “searched”.

It was originally made to classify playing cards by rank and suit; naturally, it can be changed.

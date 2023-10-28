## CNN_LSTM_LRCN_for_rear_signal

This project is credit to the website: https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/ and is powered by google colab

The tutorial is about constructing and training a CNN-LSTM model and a LRCN model to recognize rear signal turning left and right

## Step 0 Import the Libraries
``` shell
# Install the required libraries.
!pip install pafy youtube-dl moviepy
```
``` shell
# Import the required libraries.
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from moviepy.editor import *
%matplotlib inline

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
```
And will set Numpy, Python, and Tensorflow seeds to get consistent results on every execution. -> this is for the naming of the trained model, you can skip this part but the naming rule have to be changed.
``` shell
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)
```

## Step 1: Create Dataset and its Labels
if you want to try with UCF50 – Action Recognition Dataset, then you can click the link and follow the origin way. Here I'm going to show how to prepare your own dataset with a youtube video

1.1 Here is a youtube video link https://www.youtube.com/watch?v=_q3kA8OwZoU dowload it with whatever way you want (I recommend this website: https://yt1s.ltd/zh-tw164h/youtube-to-mp4 )

1.2 (optional) Trim the video with whatever you want (I recommend this website: https://online-video-cutter.com/tw/ ). Now I get a 6sec video showing a car turning left

1.3 Turned the mp4 into several consecutive frames with ffmpeg. You can download here: https://www.ffmpeg.org/
![my first git tutorial](https://github.com/HunterWang123456/CNN_LSTM_LRCN_for_rear_signal/assets/74261517/d70a2c34-bdde-458a-bcc3-fe8ed83c697f)
install it and operate it with your terminal
of course, if you think that preparing your own data is very time consuming and effort costing, you can also choose to download a dataset containing  consecutive frames. As far as I'm concerned, I chooes the Vehicle Rear Signal Dataset (http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal#) which contained thousands of rear signal frames. 

## Step 1.5: Create Dataset -- subtraction of the consecutive frames
according the this paper

## Step 2: Preprocess the Dataset
``` shell
# Specify the height and width to which each video frame will be resized in our dataset.-> the value is flexible 
IMAGE_HEIGHT , IMAGE_WIDTH = 256, 256

# Specify the number of frames of a video that will be fed to the model as one sequence.-> the value is flexible, larger sequence means longer video for training
SEQUENCE_LENGTH = 20

# Specify the directory containing the prepared dataset.
DATASET_DIR = "tell_left_right"

# Specify the list containing the names of the classes used for training. It depends on the classes and purpose of your project.
CLASSES_LIST = ["turn_left", "turn_right"]
```
Note: The IMAGE_HEIGHT, IMAGE_WIDTH and SEQUENCE_LENGTH constants can be increased for better results, although increasing the sequence length is only effective to a certain point, and increasing the values will result in the process being more computationally expensive.

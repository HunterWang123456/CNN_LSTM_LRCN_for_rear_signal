## CNN_LSTM_LRCN_for_rear_signal

Project credit to [this website](https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/) and is powered by google colab

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
And we will set Numpy, Python, and Tensorflow seeds to get consistent results on every execution. -> this is for the naming of the trained model, you can skip this part but the naming rule have to be changed.
``` shell
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)
```

## Step 1: Create Dataset and its Labels
If you want to try with UCF50 – Action Recognition Dataset, then you can click the link above and follow the auther's way. But here I'm going to show how to prepare your own dataset with a youtube video

1.1 Here is a [youtube video link](https://www.youtube.com/watch?v=_q3kA8OwZoU)  dowload it with whatever way you want (I recommend [this website](https://yt1s.ltd/zh-tw164h/youtube-to-mp4) )

1.2 (optional) Trim the video with whatever you want (I recommend [this website](https://online-video-cutter.com/tw/) )

1.3 Turned the mp4 into several consecutive frames with ffmpeg. You can download [here](https://www.ffmpeg.org/): 
![my first git tutorial](https://github.com/HunterWang123456/CNN_LSTM_LRCN_for_rear_signal/assets/74261517/d70a2c34-bdde-458a-bcc3-fe8ed83c697f)
install it and operate it with your terminal
of course, if you think that preparing your own data is very time consuming and effort costing, you can also choose to download a dataset containing  consecutive frames. As far as I'm concerned, I chooes the [Vehicle Rear Signal Dataset](http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal#) which contained thousands of rear signal frames. 

## Step 2: Create Dataset -- subtraction of the consecutive frames using sift flow
according the [this paper](https://drive.google.com/file/d/13bCTSnB-29U83QgmLWihMXlqErMmJy-E/view), when a car is turing left/right, the only thing I want the model to focus on is the flashing rear signal. However, the model seemed to be confused by the other features in the picture instead of only focus on the turing light. Thus, it's recommended that we matched the cars with sift in openCV and subtract the vehicle before feeding into the model. That means, after processing, the only thing in the picture will be the flashing rear signal. The following is a demonstration.

2.1 Introduction of SIFT

The SIFT (Scale-Invariant Feature Transform) algorithm is a computer vision technique used for feature detection and description. It detects distinctive key points or features in an image that are robust to changes in scale, rotation, and affine transformations. SIFT works by identifying keypoints based on their local intensity extrema and computing descriptors that capture the local image information around those keypoints. These descriptors can then be used for tasks like image matching, object recognition, and image retrieval. Other detail can be [reviewed here](https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/)

Upload two images and run the following code 
``` shell
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

# read images
img1 = cv2.imread('Eiffel1.png')  
img2 = cv2.imread('Eiffel2.png')

# gray scale to avoid SIFT interfered by the original color and only focus on outline
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:30], img2, flags=2)
plt.imshow(img3),plt.show()
```
![SIFT](https://github.com/HunterWang123456/Turning-Light-Detection/assets/74261517/0e213c86-3a2c-43e5-9807-655df3be5889)
Next, we are going to use this technique to match our vehicle and warp the image up to create training data

2.2 vehicle image subtraction
``` shell
import os
cur_dir = os.curdir                                                         # The root directory

data_OLO = os.path.join(cur_dir, 'turn_left')                               # OLO represent turn left in the original article provided above
data_result = os.path.join(cur_dir, 'result')
obj_classes = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10']
# A function to create the directory for pre- and post-processing data restoration
def create_dirs():
  os.mkdir(data_OLO)                                                        # Create 'turn_left' folder
  for folder_name in obj_classes:                                           # Create the folders from the list inside the 'turn_left' folder
    os.mkdir(os.path.join(data_OLO, folder_name))
  os.mkdir(data_result)                                                     # Create 'result' folder
  for folder_name in obj_classes:                                           # Create the folders from the list inside the 'result' folder
    os.mkdir(os.path.join(data_result, folder_name))
```
``` shell
create_dirs()
```
![dataset](https://github.com/HunterWang123456/CNN_LSTM_LRCN_for_rear_signal/assets/74261517/0a514e2c-154d-4781-8760-0659a72d21ad)
you can either upload the data from your own laptop or upload the dataset to google drive and transfer them to colab virtual machine via the code !cp [googel drive path]  [colab data folder path]

2.3 Implementation of sift flow
``` shell
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def linear_x_y_diff_average(img1, img2, matches, kp1, kp2):
  x_diff=[]
  y_diff=[]

  for m in matches:
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    x_diff.append(p2[0] - p1[0])
    y_diff.append(p2[1] - p1[1])

  # (TODO) optional: remove outlier

  # shift the queryImage
  x_shift = np.average(x_diff[0:5])
  y_shift = np.average(y_diff[0:5])
  print(x_shift, y_shift)
  T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
  return cv.warpAffine(img1, T, (img1.shape[1], img1.shape[0]))

def perspective_transform(img1, img2, matches, kp1, kp2):
  input_pts = []
  output_pts = []
  for m in matches:
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    input_pts.append([p1[0], p1[1]])
    output_pts.append([p2[0], p2[1]])

  M = cv.getPerspectiveTransform(np.float32(input_pts[:4]),np.float32(output_pts[:4]))
  return cv.warpPerspective(img1,M,(img1.shape[1], img1.shape[0]))

#retrive data from data set
for folder_name in obj_classes:
  data_OLO_F=os.path.join('turn_left',folder_name)
  result_OLO_F=os.path.join('result',folder_name)
  data_OLO_F_list=os.listdir(data_OLO_F)
  result_OLO_F_list=os.listdir(result_OLO_F)
  data_OLO_F_list.sort()
  for i in range(len(data_OLO_F_list)-1):
    img_file_1 = os.path.join(data_OLO_F,data_OLO_F_list[i])
    img_file_2 = os.path.join(data_OLO_F,data_OLO_F_list[i+1])
    img1 = cv.imread(img_file_1,cv.COLOR_BGR2GRAY) # first image
    print(img_file_1)
    img1 = cv.resize(img1, (256, 256))             # You have to resized all the image into the same size before feeding them into CNN-LSTM model, the value depends on the size of the original picture and you can customized the value
    img2 = cv.imread(img_file_2,cv.COLOR_BGR2GRAY) # Image for the subtraction
    print(img_file_2)
    img2 = cv.resize(img2, (256, 256))
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img1T = linear_x_y_diff_average(img1, img2, matches, kp1, kp2)
    output = cv.subtract(img2, img1T)
    out_file = os.path.join(result_OLO_F, "Img-" + str(i) + ".jpg")
    cv.imwrite(out_file,output)

# (TODO): compare the effect of perspective transform and linear shift. which is better?
#img1T = linear_x_y_diff_average(img1, img2, matches, kp1, kp2)
#img1T = perspective_transform(img1, img2, matches, kp1, kp2)

# subtract
#output = cv.subtract(img2, img1T)

#cv.imwrite('p1Tp2diff.png',output)
#plt.imshow(output),plt.show()
```
![rear_light](https://github.com/HunterWang123456/CNN_LSTM_LRCN_for_rear_signal/assets/74261517/8f6a7744-5368-4982-a0ca-5098c573c889)

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

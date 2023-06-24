# -*- coding: utf-8 -*-
"""OpenCV Practice_1

# **Chapter 2**

**Importing Libraries**
"""

import numpy
import cv2
import os # for randomly generating raw bytes
from google.colab.patches import cv2_imshow

"""**Creating a 3x3 square black image from scratch by simply creating a 2D NumPy array:**"""

img = numpy.zeros((3,3), dtype=numpy.uint8)
print("Shape of img:",img.shape)
print(img)

"""**Converting this image into blue-green-red (BGR) format using the cv2.cvtColor function**"""

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img)

"""**Converting from PNG to JPG**"""

image = cv2.imread('earth.png') # Normal
# image = cv2.imread('earth.png', cv2.IMREAD_GRAYSCALE) #Gray Scaled
cv2.imwrite('earth.jpg',image)

"""**Converting between an image and raw bytes**"""

# (row,column) ----> (y axis,x axis) ----> (height, width)
# Make sure to import os library for randomly generating raw bytes (not Efficient)


# Make an array of 120,000 random bytes.

# randomByteArray = bytearray(os.urandom(120000))
# flatNumpyArray = numpy.array(randomByteArray)

flatNumpyArray = numpy.random.randint(0, 256,120000).reshape(300, 400) #Efficient Way

# Convert the array to make a 600x200 grayscale image.
# 600 x 200 = 120000. It should match

grayImage = flatNumpyArray.reshape(200, 600)
cv2.imwrite('RandomGray.png', grayImage)

# Convert the array to make a 300x100 color image.
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)

"""**Accessing image data with numpy.array**"""

img = cv2.imread('RandomColor.png')
img[0,0] = [255,255,255]
cv2_imshow(img)

# In the following example, we change the value of the blue channel at (150, 120) from its
# current value to an arbitrary 255:

img = cv2.imread('RandomGray.png')
img.itemset((150,120,0),255)
print(img.item(150,120,0))
cv2_imshow(img)

# Let's consider an example of using array
# slicing to manipulate color channels. Setting all G (green) values of an image to 0 is as
# simple as the following code:

img = cv2.imread('RandomColor.png')
img[:, :, 1] = 0
cv2_imshow(img)

# Defining Regions of Interests (ROI)

# we can bind this region to a variable, define a second region, and assign the value of the first region to the second
# (hence, copying a portion of the image over to another position in the image)

img = cv2.imread('earth.png')
print("Shape:\t",img.shape)
print("Size:\t",img.size)
print("DType:\t",img.dtype)

my_roi = img[0:100, 0:100]
img[300:400, 300:400] = my_roi
cv2_imshow(img)

"""**Reading/writing a video file**"""

videoCapture = cv2.VideoCapture('MyInputVid.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
'MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'),
fps, size)
success, frame = videoCapture.read()
while success: # Loop until there are no more frames.
  videoWriter.write(frame)
  success, frame = videoCapture.read()

"""**Capturing Camera Frames**"""

cameraCapture = cv2.VideoCapture(0)
fps = 30 # An assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'),fps, size)
success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1 # 10 seconds of frames
while success and numFramesRemaining > 0:
  videoWriter.write(frame)
  success, frame = cameraCapture.read()
  numFramesRemaining -= 1

"""**Displaying an image in a window**"""

# It won't work in Google Collab

# img = cv2.imread('RandomGray.png')
# cv2.imshow('RandomGray',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""**Displaying camera frames in a window**"""

# Does not work in Google Colab

import cv2

clicked = False
def onMouse(event, x, y, flags, param):
  global clicked
  if event == cv2.EVENT_LBUTTONUP:
    clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
  cv2.imshow('MyWindow', frame)
  success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()

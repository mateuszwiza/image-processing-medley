from PIL import Image

import numpy as np
from matplotlib import pyplot
import math

from skimage.transform import warp

#Load the image as an array
img = np.asarray(Image.open("images/face.jpeg"))

#Get number of rows, columns and channels (RGB)
rows, cols, channel = img.shape

##############################################################################
## WAVES

# 1. HORIZONTAL WAVE
    
#Create blank output image
horizontal_wave = np.zeros(img.shape, dtype=img.dtype) 

#Scan the image and apply transformation to x coordinates of each pixel
for d in range(channel):
    for i in range(rows): 
        for j in range(cols): 
            transform_x = int(25.0 * math.sin(2 * 3.14 * i / 180)) 
            transform_y = 0 
            if j+transform_x < rows: 
                horizontal_wave[i,j,d] = img[i,(j+transform_x)%cols,d] 
            else: 
                horizontal_wave[i,j,d] = 0

# 2. VERTICAL WAVE
    
#Create blank output image             
vertical_wave = np.zeros(img.shape, dtype=img.dtype) 

#Scan the image and apply transformation to y coordinates of each pixel 
for d in range(channel):
    for i in range(rows): 
        for j in range(cols): 
            transform_x = 0 
            transform_y = int(16.0 * math.sin(2 * 3.14 * j / 150)) 
            if i+transform_y < rows: 
                vertical_wave[i,j,d] = img[(i+transform_y)%rows,j,d] 
            else: 
                vertical_wave[i,j,d] = 0 

#Plot the horizontal and vertical wave
pyplot.subplot(131),pyplot.imshow(img)
pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(132),pyplot.imshow(horizontal_wave)
pyplot.title('Horizontal wave'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(133),pyplot.imshow(vertical_wave)
pyplot.title('Vertical wave'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()

# 2. DOUBLE WAVE
    
#Create blank output image   
double_wave = np.zeros(img.shape, dtype=img.dtype) 

#Scan the image and apply transformation to x and y coordinates of each pixel 
for d in range(channel):
    for i in range(rows): 
        for j in range(cols): 
            transform_x = int(20.0 * math.sin(2 * 3.14 * i / 150)) 
            transform_y = int(20.0 * math.cos(2 * 3.14 * j / 150)) 
            if i+transform_y < rows and j+transform_x < cols: 
                double_wave[i,j,d] = img[(i+transform_y)%rows,(j+transform_x)%cols,d] 
            else: 
                double_wave[i,j,d] = 0 

#Plot the double wave
pyplot.subplot(121),pyplot.imshow(img)
pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(double_wave)
pyplot.title('Double wave'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()

##############################################################################
## SWIRL

#Apply swirl warping function f(u,v) given the image, middle coordinates and parameter R
def swirl(im, x0, y0, R):
    r = np.sqrt((im[:,1]-x0)**2 + (im[:,0]-y0)**2)
    a = np.pi * r / R
    im[:, 1] = (im[:, 1]-x0)*np.cos(a) + (im[:, 0]-y0)*np.sin(a) + x0
    im[:, 0] = -(im[:, 1]-x0)*np.sin(a) + (im[:, 0]-y0)*np.cos(a) + y0
    return im

#Warp the image using swirl function and Skimage function warp()
#witch changing parameter R
    
img_output = warp(img, swirl, map_args={'x0':360, 'y0':360, 'R':360})
img_output1 = warp(img, swirl, map_args={'x0':360, 'y0':360, 'R':180})
img_output2 = warp(img, swirl, map_args={'x0':360, 'y0':360, 'R':20})

#Plot the results
pyplot.subplot(121),pyplot.imshow(img)
pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(img_output)
pyplot.title('Swirl with R=360'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()

pyplot.subplot(121),pyplot.imshow(img_output1)
pyplot.title('Swirl with R=180'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(img_output2)
pyplot.title('Swirl with R=20'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()

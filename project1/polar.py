import cv2 

import numpy as np
from matplotlib import pyplot
import math

#Function for calculating magnitude rho and angle phi 
#from cartesian coordinates x and y
def get_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = math.degrees(np.arctan2(y, x))
    return(rho, phi)
    
#Read the image as an array    
img = cv2.imread('images/sun.jpg',0)

#Get number of rows and columns
rows, cols = img.shape

#Create a blank output image
img_output = np.zeros(img.shape, dtype=img.dtype)

#Get center point in the image
center_x = img.shape[1]/2
center_y = img.shape[0]/2

#Calculate maxRadius parameter
maxRadius = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))

#Get parameters Kx and Ky
Kx = img.shape[1]/maxRadius
Ky = img.shape[0]/360

#Scan all pixels in the image
for i in range(rows):
    for j in range(cols):
        #Get polar coordinates relative to the center of the image
        rho, phi = get_polar(j-center_x,i-center_y)
        #Calculate new positions of pixels
        rho = int(rho * Kx)
        phi = int(phi * Ky)
        #Get the pixel values from the input image
        if phi < rows and rho < cols:
            img_output[phi][rho] = img[i][j]
        else:
            img_output[phi][j] = 0
            
#Plot the results
pyplot.subplot(121),pyplot.imshow(img,cmap = 'gray')
pyplot.title('Cartesian coordinates'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(img_output,cmap = 'Greys')
pyplot.title('Polar coordinates'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()


##Alternative: CV2 Function

#Calculate maxRadius parameter
maxRadius = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))

polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), maxRadius, cv2.WARP_FILL_OUTLIERS)

pyplot.subplot(121),pyplot.imshow(img,cmap = 'gray')
pyplot.title('Cartesian coordinates'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(polar_image,cmap = 'Greys')
pyplot.title('Polar coordinates'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()



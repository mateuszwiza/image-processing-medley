import cv2 
from matplotlib import pyplot

from skimage.util import random_noise

from prewitt import prewitt, gaussian_kernel
from scipy.ndimage.filters import convolve

#Load the image
img = cv2.imread('images/pyramids.jpg',0)

#Add random noise using Skimage random_noise() function
noisy = random_noise(img, mode='s&p',amount=0.05)

#Plot the difference between original image and noisy image
pyplot.subplot(121),pyplot.imshow(img,cmap = 'gray')
pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(noisy,cmap = 'gray')
pyplot.title('With noise'), pyplot.xticks([]), pyplot.yticks([])

#Use function from prewitt.py for Prewitt edge detection
prew = prewitt(img)

#Use cv2 function for Laplacian edge detection
laplacian = cv2.Laplacian(img,cv2.CV_8U)

#Plot the results
pyplot.subplot(131),pyplot.imshow(noisy)
pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(132),pyplot.imshow(prew,cmap = 'Greys')
pyplot.title('Prewitt'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(133),pyplot.imshow(laplacian,cmap = 'gray')
pyplot.title('Laplacian'), pyplot.xticks([]), pyplot.yticks([])

##############################################################################
##Solution to poor performance: Gaussian smoothing

#Use the function in prewitt.py to generate Gaussian Kernel 
#and Scipy function convolve() to apply to the image
noisy_smoothed = convolve(noisy, gaussian_kernel(5))

#Use function from prewitt.py for Prewitt edge detection with smoothed image
prew_smoothed = prewitt(noisy_smoothed)

#Plot the difference for Prewitt
pyplot.subplot(121),pyplot.imshow(prew,cmap = 'Greys')
pyplot.title('Prewitt'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(prew_smoothed,cmap = 'Greys')
pyplot.title('Prewitt smoothed'), pyplot.xticks([]), pyplot.yticks([])

#Use cv2 function for Laplacian edge detection
laplacian_smoothed = cv2.Laplacian(noisy_smoothed,cv2.CV_8U)

#Plot the difference for Laplacian
pyplot.subplot(121),pyplot.imshow(laplacian,cmap = 'gray')
pyplot.title('Laplacian'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(laplacian_smoothed,cmap = 'gray')
pyplot.title('Laplacian smoothed'), pyplot.xticks([]), pyplot.yticks([])


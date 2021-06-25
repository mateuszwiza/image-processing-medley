import cv2 
from matplotlib import pyplot
from scipy.ndimage.filters import convolve

from prewitt import gaussian_kernel

##Canny Edge Detection with CV2 function

#Load the image as an array 
img = cv2.imread('images/sun.jpg',0)

#Detect edges with varrying threshold values
edges = cv2.Canny(img,0,250)
edges2 = cv2.Canny(img,100,250)

#Plot the results
pyplot.subplot(131),pyplot.imshow(img,cmap = 'gray')
pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(132),pyplot.imshow(edges,cmap = 'Greys')
pyplot.title('THs: 0, 250'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(133),pyplot.imshow(edges2,cmap = 'Greys')
pyplot.title('THs: 100, 250'), pyplot.xticks([]), pyplot.yticks([])
 
pyplot.show()

#Use additional Gaussian smoothing to improve result

    #Use the function in prewitt.py to generate Gaussian Kernel 
    #and Scipy function convolve() to apply to the image
img_smoothed = convolve(img, gaussian_kernel(5))

#Detect edges using smoothed image
edges_smoothed = cv2.Canny(img_smoothed,100,250)

pyplot.subplot(121),pyplot.imshow(img,cmap = 'gray')
pyplot.title('Canny smoothing'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122),pyplot.imshow(img_smoothed,cmap = 'Greys')
pyplot.title('Additional smoothing'), pyplot.xticks([]), pyplot.yticks([])




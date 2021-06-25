import cv2 
import numpy as np
from matplotlib import pyplot
from scipy.ndimage.filters import convolve

#Function for Prewitt edge detection and plotting the results
def prewitt(img):

    #Get number of rows and columns and channels
    rows, cols, channels = img.shape
    
    #Define Prewitt masks
    
    #horizontal = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])  
    #vertical = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])  
    
    diagonal1 = np.array([[0,1,1], [-1,0,1], [-1,-1,0]])
    diagonal2 = np.array([[-1,-1,0], [-1,0,1], [0,1,1]])
    
    #Create empty output images
    diagonal1_output = np.zeros((rows, cols))
    diagonal2_output = np.zeros((rows, cols))
    output_image = np.zeros((rows, cols, channels))

    #Convolate the image and apply the Prewitt kernels to get gradients
    #and set proper pixel intensities based on gradients
    for d in range(channels):
        for i in range(rows):
            for j in range(cols):
                #Consider a pixel and its 8-neighbours
                gradient1 = (diagonal1[0, 0] * img[i - 1, j - 1,d]) + \
                                 (diagonal1[0, 1] * img[i - 1, j,d]) + \
                                 (diagonal1[0, 2] * img[i - 1, j + 1,d]) + \
                                 (diagonal1[1, 0] * img[i, j - 1,d]) + \
                                 (diagonal1[1, 1] * img[i, j,d]) + \
                                 (diagonal1[1, 2] * img[i, j + 1,d]) + \
                                 (diagonal1[2, 0] * img[i + 1, j - 1,d]) + \
                                 (diagonal1[2, 1] * img[i + 1, j,d]) + \
                                 (diagonal1[2, 2] * img[i + 1, j + 1,d])
        
                diagonal1_output[i, j] = abs(gradient1)
        
                gradient2 = (diagonal2[0, 0] * img[i - 1, j - 1,d]) + \
                               (diagonal2[0, 1] * img[i - 1, j,d]) + \
                               (diagonal2[0, 2] * img[i - 1, j + 1,d]) + \
                               (diagonal2[1, 0] * img[i, j - 1,d]) + \
                               (diagonal2[1, 1] * img[i, j,d]) + \
                               (diagonal2[1, 2] * img[i, j + 1,d]) + \
                               (diagonal2[2, 0] * img[i + 1, j - 1,d]) + \
                               (diagonal2[2, 1] * img[i + 1, j,d]) + \
                               (diagonal2[2, 2] * img[i + 1, j + 1,d])
        
                diagonal2_output[i, j] = abs(gradient2)
        
                #Based on the gradients set the magnitude of pixels in the output               
                mag = np.sqrt(pow(gradient1, 2.0) + pow(gradient2, 2.0))
                output_image[i - 1, j - 1, d] = mag
    
    #Combine outputs from different channels into one output image             
    output_rgb = output_image[:,:,0] + output_image[:,:,1] + output_image[:,:,2]
    
    #Plot the results        
    pyplot.subplot(131),pyplot.imshow(diagonal1_output,cmap = 'Greys')
    pyplot.title('Diagonal1'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(132),pyplot.imshow(diagonal2_output,cmap = 'Greys')
    pyplot.title('Diagonal2'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(133),pyplot.imshow(output_rgb)
    pyplot.title('Prewitt edge detection'), pyplot.xticks([]), pyplot.yticks([])
            
    pyplot.show()
    
#Function for generating Gaussian kernel of given size
def gaussian_kernel(size):
    size = int(size) // 2
    sigma=1
    x, y = np.mgrid[-size:size+1, -size:size+1]
    gaus = 1 / (2.0 * np.pi * sigma**2)
    G =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * gaus
    return G

##############################################################################
    
#Load the image as an array
img = cv2.imread('images/sun.jpg',0)

#Detect edges using Prewitt kernels
prewitt(img)

#Use the function in prewitt.py to generate Gaussian Kernel 
#and Scipy function convolve() to apply to the image
smoothed = convolve(img, gaussian_kernel(5))

#Detect edges using Prewitt kernels of smoothed image
prewitt(smoothed)



#ALTERNATIVE: Use cv2 function for Prewitt edge detection
#define masks
kernelx = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
#apply detection
img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
#combine results into output image
img_prewitt = img_prewittx + img_prewitty

#Plot the results
pyplot.subplot(131),pyplot.imshow(img_prewittx,cmap = 'Greys')
pyplot.title('X'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(132),pyplot.imshow(img_prewitty,cmap = 'Greys')
pyplot.title('Y'), pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(133),pyplot.imshow(img_prewitt,cmap = 'Greys')
pyplot.title('Prewitt'), pyplot.xticks([]), pyplot.yticks([])
pyplot.show()
    

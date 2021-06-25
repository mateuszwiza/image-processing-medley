# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:51:39 2020

@author: Mateusz.Wiza
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt


## BINARIZATION

def binarize(img):
    rows, cols = img.shape
    pixels = rows*cols
    
    #calculate and plot image histogram
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.show()
    
    #choose possible thresholds
    t = []
    for idx, element in enumerate(hist):
        if element > (pixels/256) and idx > 20 and idx < 245:
            t.append(idx)
    
    #calculate Within-Class variance according to formulas here:
    #http://www.ijcttjournal.org/Volume17/number-5/IJCTT-V17P150.pdf
    wc_variance = []
    
    for idx,element in enumerate(t):
        
        #background
        Wb = Mb = Mb1 = Mb2 = Vb = Vb1 = Vb2 = 0
        for i in range(element):
            Wb += (hist[i][0])/pixels   #background Weight
            Mb1 += i*(hist[i][0])
            Mb2 += (hist[i][0])
        if Mb2 > 0:
            Mb = Mb1/Mb2     #background mean
        else:
            Mb = 0
        for i in range(element):
            Vb1 = (i-Mb)**2 * (hist[i][0])
            Vb2 += (hist[i][0])
        if Vb2 > 0:
            Vb = Vb1/Vb2     #background variance
        else:
            Vb = 0
        
        #foreground
        Wf = Mf = Mf1 = Mf2 = Vf = Vf1 = Vf2 = 0
        L = 255-element
        for i in range(L):
            n_i = i+element
            Wf += (hist[n_i][0])/pixels     #foreground Weight
            Mf1 += n_i*(hist[n_i][0])
            Mf2 += (hist[n_i][0])
        if Mf2 > 0:
            Mf = Mf1/Mf2     #foreground mean
        else:
            Mf = 0
        
        for i in range(L):
            Vf1 = (n_i-Mf)**2 * (hist[n_i][0])
            Vf2 += (hist[n_i][0])
        if Vf2 > 0:
            Vf = Vf1/Vf2     #foreground variance
        else:
            Vf = 0
        
        variance = Wb*Vb + Wf*Vf     #Within-Class Variance
        wc_variance.append(variance)    #List of WC-Variances
    
    #Choose global threshold by getting possible threshold t with smallest 
    #Within-Class variance
    T = t[wc_variance.index(min(wc_variance))]    
    
    #Binarize image using global threshold
    bin_img = np.zeros(img.shape)
    for i in range(rows):
        for j in range(cols):
            if img[i][j]<T:
                bin_img[i][j] = 1
                
    return bin_img

            
## EROSION
                
def erosion(img,kernel):
    rows, cols = img.shape
    img_erosion = np.zeros(img.shape)
    
    #The kernel is moved along the original image and the center pixels are 
    #set to ‘1’ if all pixels under the kernel are ‘1s’
    for i in range(rows-1):
        for j in range(cols-1):
            if img[i-1, j-1] == kernel[0][0] and img[i-1, j] == kernel[0][1] and img[i-1, j+1] == kernel[0][2] and img[i, j-1] == kernel[1][0] and img[i, j] == kernel[1][1] and img[i, j+1] == kernel[1][2] and img[i+1, j-1] == kernel[2][0] and img[i+1, j] == kernel[2][1] and img[i+1, j+1] == kernel[2][2]:
                img_erosion[i][j] = 1
    
    return img_erosion


## DILATION

def dilation(img,kernel):
    rows, cols = img.shape
    img_dilation = np.zeros(img.shape)
    
    #The kernel is moved along the original image and the center pixels are 
    #set to ‘1’ if at least one pixel under the kernel is ‘1’
    for i in range(rows-1):
        for j in range(cols-1):
            if img[i-1, j-1] == kernel[0][0] or img[i-1, j] == kernel[0][1] or img[i-1, j+1] == kernel[0][2] or img[i, j-1] == kernel[1][0] or img[i, j] == kernel[1][1] or img[i, j+1] == kernel[1][2] or img[i+1, j-1] == kernel[2][0] or img[i+1, j] == kernel[2][1] or img[i+1, j+1] == kernel[2][2]:
                img_dilation[i][j] = 1
    
    return img_dilation

## MAIN
 
#Load the image    
img = cv2.imread('images/satelite.png', 0) 

#Binarize
bin_img = binarize(img)

#Show the results
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bin_img,cmap = 'Greys')
plt.title('Binarizied'), plt.xticks([]), plt.yticks([])
plt.show()

#Define kernel for morphological transformations
kernel = np.ones((3,3), np.uint8)

#Opening
img_erosion = erosion(img,kernel)
img_opening = dilation(img_erosion,kernel)

#Closing of opening
img_dilation = dilation(img_opening,kernel)
img_closing = erosion(img_dilation,kernel)

#Plot the results
plt.subplot(131),plt.imshow(bin_img,cmap = 'Greys')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_opening,cmap = 'Greys')
plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_closing,cmap = 'Greys')
plt.title('Closing of opening'), plt.xticks([]), plt.yticks([])
  
plt.show()

#Opening and closing with CV2
kernel = np.ones((5,5), np.uint8)
img_opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)

#Plot the results
plt.subplot(131),plt.imshow(bin_img,cmap = 'Greys')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_opening,cmap = 'Greys')
plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_closing,cmap = 'Greys')
plt.title('Closing of opening'), plt.xticks([]), plt.yticks([])
  
plt.show()

#Plot the final comparison between input and output
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_closing,cmap = 'Greys')
plt.title('Transformed'), plt.xticks([]), plt.yticks([])
plt.show()

    
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:38:21 2020

@author: Mateusz.Wiza
"""
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from math import copysign, log10, sqrt

# LOAD THE VIDEO as a list of frames

video = cv2.VideoCapture('videos/person22_handwaving_d1_uncomp.avi')
#video = cv2.VideoCapture('videos/person11_boxing_d4_uncomp.avi')
#video = cv2.VideoCapture('videos/person01_handclapping_d4_uncomp.avi')
frames = []
while(video.isOpened()):
    ret, frame = video.read()
    if ret == False:
        break
    frames.append(frame)
video.release()
cv2.destroyAllWindows()

#%%
# CREATE MEIs

n,rows,cols,ch = np.shape(frames)
MEI = np.zeros([rows,cols])

first = 25  #index of first frame
w = 2       #temporal window
last = 95   #index of the last frame

for idx in range(last-first):
    prvs = cv2.cvtColor(frames[first+idx-w],cv2.COLOR_BGR2GRAY)
    nextf = cv2.cvtColor(frames[first+idx],cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,nextf, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    for i in range(rows):
        for j in range(cols):
            if abs(flow[i][j][0] - flow[i][j][1])>2.5:
                MEI[i][j] = 1
    
    #Display multiple steps (frame + MEI)
    #plt.imshow(nextf,cmap = 'gray'), plt.title('Frame'+str(first+idx)), plt.xticks([]), plt.yticks([])
    #plt.show()
    #plt.imshow(MEI,cmap = 'Greys'), plt.title('MEI'+str(first+idx)), plt.xticks([]), plt.yticks([])
    #plt.show()

#Display final MEI 
plt.imshow(MEI,cmap = 'Greys'), plt.title('MEI clapping'), plt.xticks([]), plt.yticks([])
plt.show()

MEI_wave = MEI
#MEI_clap = MEI
#MEI_box = MEI

#%%
# APPLY MORPHOLOGICAL TRANSFORMATIONS

#Define kernel
kernel = np.ones((5,5), np.uint8)


wave_closing = cv2.morphologyEx(MEI_wave, cv2.MORPH_CLOSE, kernel)
wave_opening = cv2.morphologyEx(wave_closing, cv2.MORPH_OPEN, kernel)

clap_closing = cv2.morphologyEx(MEI_clap, cv2.MORPH_CLOSE, kernel)
clap_opening = cv2.morphologyEx(clap_closing, cv2.MORPH_OPEN, kernel)

box_closing = cv2.morphologyEx(MEI_box, cv2.MORPH_CLOSE, kernel)
box_opening = cv2.morphologyEx(box_closing, cv2.MORPH_OPEN, kernel)

#Display the results
plt.subplot(131),plt.imshow(MEI_wave,cmap = 'Greys')
plt.title('Waving'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(MEI_clap,cmap = 'Greys')
plt.title('Clapping'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(MEI_box,cmap = 'Greys')
plt.title('Boxing'), plt.xticks([]), plt.yticks([])
plt.show()

#%%
# EXTRACT OUTLINE

#Define kernel
kernel = np.ones((3,3), np.uint8)
gradient_wave = cv2.morphologyEx(wave_opening, cv2.MORPH_GRADIENT, kernel)
gradient_clap = cv2.morphologyEx(clap_opening, cv2.MORPH_GRADIENT, kernel)
gradient_box = cv2.morphologyEx(box_opening, cv2.MORPH_GRADIENT, kernel)

#Display the results
plt.subplot(131),plt.imshow(gradient_wave,cmap = 'Greys')
plt.title('Waving'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(gradient_clap,cmap = 'Greys')
plt.title('Clapping'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(gradient_box,cmap = 'Greys')
plt.title('Boxing'), plt.xticks([]), plt.yticks([])
plt.show()

#%%
# HU MOMENTS

# Inspired by: https://github.com/spmallick/learnopencv/tree/master/HuMoments

showLogTransformedHuMoments = True

# Threshold image
#ths,im = cv2.threshold(MEI, 128, 255, cv2.THRESH_BINARY)

# Calculate Moments
moment_wave = cv2.moments(gradient_wave)
moment_clap = cv2.moments(gradient_clap)
moment_box = cv2.moments(gradient_box)

# Calculate Hu Moments
huMoments_wave = cv2.HuMoments(moment_wave)
huMoments_clap = cv2.HuMoments(moment_clap)
huMoments_box = cv2.HuMoments(moment_box)

hu_wave = []
hu_clap = []
hu_box = []
#Log-transform the Hu Moments
for i in range(0,7):
    hu1 = -1*copysign(1.0,huMoments_wave[i])*log10(abs(huMoments_wave[i]))
    hu2 = -1*copysign(1.0,huMoments_clap[i])*log10(abs(huMoments_clap[i]))
    hu3 = -1*copysign(1.0,huMoments_box[i])*log10(abs(huMoments_box[i]))
    hu_wave.append(hu1)
    hu_clap.append(hu2)
    hu_box.append(hu3)

# Print Hu Moments
print("{}: ".format('Waving'),end='')

for i in range(0,7):
    if showLogTransformedHuMoments:
        print("{:.5f}".format(hu_wave[i]),\
                end=' ')
    else:
        # Hu Moments without log transform
        print("{:.5f}".format(huMoments_wave[i]),end=' ')
print()
print("{}: ".format('Clapping'),end='')
for i in range(0,7):
    if showLogTransformedHuMoments:
        print("{:.5f}".format(hu_clap[i]),\
                end=' ')
    else:
        # Hu Moments without log transform
        print("{:.5f}".format(huMoments_clap[i]),end=' ')
print()
print("{}: ".format('Boxing'),end='')
for i in range(0,7):
    if showLogTransformedHuMoments:
        print("{:.5f}".format(hu_box[i]),\
                end=' ')
    else:
        # Hu Moments without log transform
        print("{:.5f}".format(huMoments_box[i]),end=' ')
print()

#%%
# EUCLIDEAN DISTANCE

dist_wave_clap = dist_wave_box = dist_clap_box = 0

for i in range(0,7):
    dist_wave_clap += (huMoments_wave[i]-huMoments_clap[i])**2
    dist_wave_box += (huMoments_wave[i]-huMoments_box[i])**2
    dist_clap_box += (huMoments_clap[i]-huMoments_box[i])**2

dist_wave_clap = sqrt(dist_wave_clap)
dist_wave_box = sqrt(dist_wave_box)
dist_clap_box = sqrt(dist_clap_box)

print('Distance WAVING-CLAPPING:', dist_wave_clap)
print('Distance WAVING-BOXING:', dist_wave_box)
print('Distance BOXING-CLAPPING:', dist_clap_box)

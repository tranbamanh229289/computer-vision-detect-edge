import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE = 'images/17.jpg'

img = cv2.imread(IMAGE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[:,:,::-1]

plt.subplot(2,2,1)
gx = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)  # Gradient y => detect edge Ox
plt.imshow(gx)

plt.subplot(2,2,2)
gy = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3) # Gradient x => detect edge Oy
plt.imshow(gy)

plt.subplot(2,2,3)
g, theta = cv2.cartToPolar(gx, gy, angleInDegrees=True)  # g = Magnitute gradient Ox, Oy, theta = direction gradient
plt.imshow(g)

plt.subplot(2,2,4)
plt.imshow(theta)

plt.show()
import cv2
from skimage import exposure
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np

path = 'images/16.jpg'

N_BINS = 9
CELL_SIZE = np.array([8,8]) #pixel
BLOCK_SIZE = np.array([2,2])  #cell
root_image = cv2.imread(path)
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
blockSize = CELL_SIZE*BLOCK_SIZE  #pixel
winSize = np.array([image.shape[0]//CELL_SIZE[0] * CELL_SIZE[0], image.shape[1]//CELL_SIZE[1] * CELL_SIZE[1]]) # num cell
blockStride = CELL_SIZE  #pixel
dx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])  #Bộ lọc Sobel theo phương x
dy = np.transpose(dx)  # Sobel theo phương y
gx = cv2.filter2D(image, cv2.CV_32F, dx)
gy = cv2.filter2D(image, cv2.CV_32F, dy)

num_cell = np.array([image.shape[0] // CELL_SIZE[0], image.shape[1] // CELL_SIZE[1]])
(h, hogImage) = feature.hog(image, orientations=N_BINS, pixels_per_cell= CELL_SIZE, cells_per_block= BLOCK_SIZE, transform_sqrt=True, block_norm="L2", visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0,255))
hogImage = hogImage.astype("uint8")

fig_hog = plt.subplot(2,2,2)
plt.imshow(hogImage)
fig_hog.set_title("hog image")

fig_root = plt.subplot(2,2,1)
plt.imshow(root_image[:,:,::-1])
fig_root.set_title("root image")

fig_sobel_x = plt.subplot(2,2,3)
plt.imshow(gx)
fig_sobel_x.set_title("gradient sobel x ")

fig_sobel_y = plt.subplot(2,2,4)
plt.imshow(gy)
fig_sobel_y.set_title("gradient sobel y")

plt.show()

print("Vector hog : ", h)
print("Kích thước vector hog là: ", h.shape, "=", num_cell[0]-1, "x", num_cell[1]-1, "x",BLOCK_SIZE[0]*BLOCK_SIZE[1], "x", N_BINS )


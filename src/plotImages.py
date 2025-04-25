# --------------------------------------------------------------
# --------------------------------------------------------------

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --------------------------------------------------------------
# function to plot thermal imagens from raw file
# --------------------------------------------------------------

def plot_csv_image(csv_file):
    data = pd.read_csv(csv_file, header=None)
    image_array = data.to_numpy()
    plt.imshow(image_array, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.show()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# # example of healthy image
# plot_csv_image(csv_file=x_train_files[0])
# # plot_csv_image(csv_file=x_test_files[0])
# print(x_test_files[0])
# print(y_test[0])

# --------------------------------------------------------------
# normalize image withing [min, max] range
# --------------------------------------------------------------

def normalize(image):
    max = np.amax(image)
    min = np.amin(image)
    normalized_image = (image-min)/(max - min)
    return (normalized_image)

# --------------------------------------------------------------
# --------------------------------------------------------------

def thermal_to_rgb_image(image):
  norm_img = normalize(image = image)
  new_img  = np.round(norm_img * 255)
  u8 = new_img.astype(np.uint8)
  im_color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
  return(im_color)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
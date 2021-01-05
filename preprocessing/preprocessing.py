import os
import utils
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility purposes
np.random.seed(42)

# We obtain the path of all the original images
base_path = "./preprocessing/captcha_images/original"
images_path = list(map(lambda x: os.path.join(base_path, x), os.listdir(base_path)))

random_images = utils.get_random_images(images_path)

threshold_level = 99

# 1: Load the images
for image in random_images:
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = utils.threshold_image(image, threshold_level)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.1: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # 4.2: Contour drawing
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)

    threshold_level = 99

area_threshold = 100

# NEW: Kernel for the closing operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 1: Load the images
for image in random_images:
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = utils.threshold_image(image, threshold_level)
    
    # NEW: Apply a closing operation
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.1: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # 4.2: Contour drawing
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # NEW: Apply an area threshold
        if w * h > area_threshold:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    # 5: Visualization
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gray)
    plt.axis('off')
    
    plt.show()
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_random_images(images_path, n=5):
    image_collection = []
    
    for i in np.random.permutation(len(images_path))[:n]:
        image = cv2.imread(images_path[i])
        image_collection.append(image)
        
    return image_collection

def plot_histogram(image, colorspace):
    chans = cv2.split(image)
    
    plt.title(f"{colorspace} Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    for (chan_data, channel_name) in zip(chans, list(colorspace)):
        if colorspace == "BGR":
            hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])
            plt.plot(hist, label=channel_name, color=channel_name.lower())
        elif colorspace == "HSV":
            if channel_name == "H":
                hist = cv2.calcHist([chan_data], [0], None, [180], [0, 256])
            else:
                hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])
                
            plt.plot(hist, label=channel_name)
        
    
    plt.xlim([0, 256])
    plt.legend(loc="upper right")

def threshold_image(image, level):
    # image should be in BGR space
    
    # Conversion and extraction of `value` channel
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)[:, :, 2]
    
    # Thresholding at specified level
    _, thresh_img = cv2.threshold(image, level, 255, cv2.THRESH_BINARY_INV)
    
    return thresh_img

def get_letters(image, kernel, threshold_level=99, area_threshold=100):
    letters_bin = []
    rects = []
    
    # 2, 3: Convert to HSV, obtain the value channel and threshold it
    binarized = threshold_image(image, threshold_level)
    
    # 3.1: Apply a closing operation
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # 4: Obtain the contours over the binarized image
    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4.2: Create a copy of the binary image to draw over it
    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # Creation of bounding rects
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        # 4.4: Apply an area threshold
        if w * h > area_threshold:
            rects.append((x, y, w, h))
    
    # Sort by position along the x axis. This is necessary because the order does matter when
    # typing the answer to a captcha
    rects = sorted(rects, key=lambda x: x[0])
    
    for (x, y, w, h) in rects:
        # 4.5: Extract the letter and append it into an array
        letters_bin.append(binarized[y:(y + h), x:(x + w)])
            
            
    return letters_bin
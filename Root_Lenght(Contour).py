#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_longest_contour_below_horizontal(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and Canny Edge Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Print total number of contours
    print(f"Total number of contours: {len(contours)}")

    # Variables for the longest horizontal contour
    longest_horizontal_contour = None
    max_horizontal_length = 0
    horizontal_contour_y = 0

    # Find the longest horizontal contour
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        _, (w, h), angle = rect
        if -10 <= angle <= 10 or -100 <= angle <= -80 or 80 <= angle <= 100:
            length = max(w, h)
            if length > max_horizontal_length:
                max_horizontal_length = length
                longest_horizontal_contour = contour
                horizontal_contour_y = rect[0][1]

    # Crop the image below the longest horizontal contour
    if longest_horizontal_contour is not None:
        cropped_image = image[int(horizontal_contour_y):, :]
        cropped_gray = gray[int(horizontal_contour_y):, :]
    else:
        cropped_image = image.copy()
        cropped_gray = gray.copy()

    # Apply Canny Edge Detection to the cropped image
    edges_cropped = cv2.Canny(cropped_gray, 50, 150)

    # Find contours in the cropped image
    contours_cropped, _ = cv2.findContours(edges_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the longest contour in the cropped image
    longest_contour = None
    max_length = 0
    for contour in contours_cropped:
        length = cv2.arcLength(contour, True)
        if length > max_length:
            max_length = length
            longest_contour = contour

    # Draw the longest contour in the cropped image
    if longest_contour is not None:
        cv2.drawContours(cropped_image, [longest_contour], -1, (0, 255, 0), 3)  # Green color
        print(f"Length of the longest contour in the cropped image: {max_length}px")

    # Convert cropped image to RGB for Matplotlib and display
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# Replace with the path to your image
find_longest_contour_below_horizontal("C:\\My data\\Virginia tech\\Wright Labs\\Epithitis model\\Roots\\test_1.png")


# In[ ]:





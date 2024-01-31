#!/usr/bin/env python
# coding: utf-8

# In[84]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_horizontal_line_based_on_color_dip(image):
    """
    Find the y-coordinate of a horizontal line based on a dip in the average color of each row.
    """
    y_coordinate=np.diff((image[:,:,0].mean(1))).argmax() +4
    print(y_coordinate)

    
    return y_coordinate



def process_image(image_path):
    """
    Process the image to find the longest non-horizontal contour below the detected horizontal line,
    cropping everything above the line.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Find the horizontal line based on color dip
    horizontal_line_y = find_horizontal_line_based_on_color_dip(image)

    # Crop the image to keep only the part below the horizontal line
    cropped_image = image[horizontal_line_y:, :, :]

    # Convert cropped image to grayscale and find edges
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 35, 50)
    plt.imshow(edges)
    

    

    # Find contours in the cropped image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Print total number of contours
    print(f"Total number of contours: {len(contours)}")

    # Find the longest contour in the cropped image
    longest_contour = None
    max_length = 0
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length > max_length:
            max_length = length
            longest_contour = contour

    # Draw the longest contour in the cropped image
    if longest_contour is not None:
        cv2.drawContours(cropped_image, [longest_contour], -1, (0, 255, 0), 3)  # Green color
        print(f"Length of the longest contour in the cropped image: {max_length*0.00183364}cm")

    # Convert cropped image to RGB for Matplotlib and display
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()





# Replace 'path_to_image.png' with the path to your image
process_image("C:\\My data\\Virginia tech\\Wright Labs\\Epithitis model\\Roots\\Batch 1\\auxin100_A_col0.png")


# In[24]:


img = cv2.imread("C:\\My data\\Virginia tech\\Wright Labs\\Epithitis model\\Roots\\100auxin_A_afb1-3.png")


# In[25]:


img.shape


# In[18]:


plt.plot(img[:,:,0].mean(1))


# In[19]:


img[:,:,0].mean(1).argmin()


# In[54]:


plt.plot(np.diff((img[:,:,0].mean(1))))


# In[55]:


np.diff((img[:,:,0].mean(1))).armax()


# In[56]:


np.diff((img[:,:,0].mean(1))).argmax()


# In[ ]:




